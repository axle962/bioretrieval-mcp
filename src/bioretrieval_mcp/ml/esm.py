"""ESM-2 sequence embeddings with an on-disk cache.

Pooling matches the training pipeline of the two-tower model: truncate to the
first 1022 residues, take the final layer, mean over residue tokens (BOS/EOS
excluded). Keeping this identical is what makes the cached vectors reusable as
retrieval queries.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from bioretrieval_mcp.validation import ESM_MAX_RESIDUES

log = logging.getLogger(__name__)


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sequence_key(model_name: str, sequence: str) -> str:
    return hashlib.sha256(f"{model_name}:{sequence}".encode()).hexdigest()


class EsmEmbedder:
    """Lazy-loading ESM-2 embedder. Thread-safe; call from a worker thread."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        cache_dir: Path | None = None,
        device: str = "auto",
        memory_items: int = 256,
        cache_namespace: str | None = None,
    ):
        self.model_name = model_name
        # Distinct weights (e.g. fine-tuned ESM from a checkpoint) must not share cache entries.
        self.cache_namespace = cache_namespace or model_name
        self.cache_dir = cache_dir
        self.device_pref = device
        self._model: Any = None
        self._batch_converter: Any = None
        self._layer: int | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mem_items = memory_items
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, state_dict: dict[str, Any] | None = None) -> None:
        """Load weights. ``state_dict`` overrides them (e.g. a fine-tuned ESM from a checkpoint)."""
        with self._load_lock:
            if self._model is not None:
                return
            import esm

            log.info("Loading %s", self.model_name)
            model, alphabet = getattr(esm.pretrained, self.model_name)()
            if state_dict:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if unexpected or len(missing) > 4:  # contact head keys may legitimately be absent
                    raise RuntimeError(
                        f"ESM weights in checkpoint do not match {self.model_name}: "
                        f"{len(missing)} missing, {len(unexpected)} unexpected keys"
                    )
            self._device = resolve_device(self.device_pref)
            self._model = model.eval().to(self._device)
            self._batch_converter = alphabet.get_batch_converter()
            self._layer = len(model.layers)

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.npy" if self.cache_dir else None

    def lookup(self, sequence: str) -> np.ndarray | None:
        key = sequence_key(self.cache_namespace, sequence)
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        path = self._cache_path(key)
        if path is not None and path.exists():
            vec = np.load(path)
            self._remember(key, vec)
            return vec
        return None

    def _remember(self, key: str, vec: np.ndarray) -> None:
        self._mem[key] = vec
        self._mem.move_to_end(key)
        while len(self._mem) > self._mem_items:
            self._mem.popitem(last=False)

    def embed(self, sequence: str) -> tuple[np.ndarray, bool]:
        """Return (1280-d float32 vector, was_cached)."""
        if (hit := self.lookup(sequence)) is not None:
            return hit, True
        self.load()
        import torch

        seq = sequence[:ESM_MAX_RESIDUES]
        with self._run_lock, torch.no_grad():
            _, _, tokens = self._batch_converter([("q", seq)])
            out = self._model(tokens.to(self._device), repr_layers=[self._layer], return_contacts=False)
            vec = out["representations"][self._layer][0, 1 : len(seq) + 1].mean(0).float().cpu().numpy()
        vec = vec.astype(np.float32)
        key = sequence_key(self.cache_namespace, sequence)
        if (path := self._cache_path(key)) is not None:
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, vec)
            tmp.replace(path)
        self._remember(key, vec)
        return vec, False
