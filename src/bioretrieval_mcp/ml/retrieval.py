"""Sequence -> cryo-EM map retrieval with the two-tower (JEPA + InfoNCE) model.

Only the *sequence* tower runs at query time. Map embeddings are precomputed
offline by the volume tower and shipped as an index file (``scripts/build_map_index.py``):

    map_index.npz
      emdb_ids     (N,)   str
      embeddings   (N, D) float32   volume-tower outputs (re-normalized on load)
      uniprot_ids  (N,)   str       optional, the paired sequence of each map
      resolutions  (N,)   float     optional
      titles       (N,)   str       optional
      kind         ()     str       "two_tower" (default) or "esm_seq_baseline"
      model        ()     str       free-text model name reported to the agent

Supported checkpoints (autodetected):
  * ``mmprotein`` trainer: ``{"model_state_dict": ..., "config": ...}``, head ``seq_projector.net`` (ReLU)
  * node2/node3 E2E trainers: ``{"model": ...}``, head ``sequence_projection.net`` (GELU),
    optionally with (fine-tuned) ESM-2 weights under ``sequence_encoder.esm_model``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bioretrieval_mcp.ml.esm import EsmEmbedder

log = logging.getLogger(__name__)

# prefix -> activation used between LayerNorm and the output Linear
HEAD_PREFIXES = {
    "sequence_projection.net.": "gelu",
    "seq_projector.net.": "relu",
}
ESM_PREFIX = "sequence_encoder.esm_model."


@dataclass
class MapIndex:
    emdb_ids: np.ndarray
    embeddings: np.ndarray
    uniprot_ids: np.ndarray | None
    resolutions: np.ndarray | None
    titles: np.ndarray | None
    kind: str
    model: str

    @classmethod
    def load(cls, path: Path) -> MapIndex:
        with np.load(path, allow_pickle=False) as z:
            emb = z["embeddings"].astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
            ids = z["emdb_ids"].astype(str)
            if len(ids) != len(emb):
                raise ValueError(f"Map index {path}: {len(ids)} ids but {len(emb)} embeddings")
            opt = {k: z[k] for k in ("uniprot_ids", "resolutions", "titles") if k in z.files}
            return cls(
                emdb_ids=ids,
                embeddings=emb,
                uniprot_ids=opt["uniprot_ids"].astype(str) if "uniprot_ids" in opt else None,
                resolutions=opt["resolutions"].astype(float) if "resolutions" in opt else None,
                titles=opt["titles"].astype(str) if "titles" in opt else None,
                kind=str(z["kind"]) if "kind" in z.files else "two_tower",
                model=str(z["model"]) if "model" in z.files else path.stem,
            )

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def __len__(self) -> int:
        return len(self.emdb_ids)


def _unwrap_state_dict(ckpt: Any) -> dict[str, Any]:
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state dict")
    if all(k.startswith("module.") for k in state):  # DDP
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def build_sequence_head(state: dict[str, Any], activation: str | None = None):
    """Rebuild ``Linear -> LayerNorm -> act -> Linear`` from checkpoint weights."""
    import torch.nn as nn

    for prefix, default_act in HEAD_PREFIXES.items():  # noqa: B007 (used after break)
        if f"{prefix}0.weight" in state:
            break
    else:
        raise ValueError(f"No sequence projection head found (looked for {', '.join(HEAD_PREFIXES)})")
    w0, w3 = state[f"{prefix}0.weight"], state[f"{prefix}3.weight"]
    hidden, in_dim = w0.shape
    out_dim = w3.shape[0]
    act = {"gelu": nn.GELU, "relu": nn.ReLU}[(activation or default_act).lower()]
    head = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), act(), nn.Linear(hidden, out_dim))
    head.load_state_dict({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})
    return head.eval(), prefix


class Retriever:
    def __init__(
        self,
        index: MapIndex,
        embedder: EsmEmbedder,
        head: Any | None = None,
        model_name: str | None = None,
    ):
        self.index = index
        self.embedder = embedder
        self.head = head
        self.model_name = model_name or index.model
        if head is None and index.kind != "esm_seq_baseline":
            raise ValueError("A two-tower map index needs the checkpoint's sequence head; set the checkpoint path.")

    @classmethod
    def from_paths(
        cls,
        index_path: Path,
        checkpoint_path: Path | None,
        base_embedder: EsmEmbedder,
        activation: str | None = None,
    ) -> Retriever:
        index = MapIndex.load(index_path)
        if checkpoint_path is None:
            return cls(index, base_embedder)
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = _unwrap_state_dict(ckpt)
        head, prefix = build_sequence_head(state, activation)
        embedder = base_embedder
        esm_state = {k[len(ESM_PREFIX) :]: v for k, v in state.items() if k.startswith(ESM_PREFIX)}
        if esm_state:
            digest = hashlib.sha256(checkpoint_path.read_bytes()[:1 << 20]).hexdigest()[:12]
            embedder = EsmEmbedder(
                base_embedder.model_name,
                base_embedder.cache_dir,
                base_embedder.device_pref,
                cache_namespace=f"{base_embedder.model_name}+{checkpoint_path.name}:{digest}",
            )
            embedder.load(esm_state)
        log.info("Loaded sequence head %s from %s (ESM from checkpoint: %s)", prefix, checkpoint_path, bool(esm_state))
        return cls(index, embedder, head, model_name=f"{index.model} ({checkpoint_path.name})")

    def encode_query(self, sequence: str) -> np.ndarray:
        vec, _ = self.embedder.embed(sequence)
        if self.head is not None:
            import torch

            with torch.no_grad():
                vec = self.head(torch.from_numpy(vec)[None]).squeeze(0).numpy()
        vec = vec.astype(np.float32)
        if vec.shape[0] != self.index.dim:
            raise ValueError(f"Query dim {vec.shape[0]} does not match index dim {self.index.dim}")
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    def search(self, sequence: str, k: int) -> list[dict[str, Any]]:
        q = self.encode_query(sequence)
        scores = self.index.embeddings @ q
        order = np.argsort(-scores)
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for i in order:  # one row per map; keep its best-scoring row
            emdb_id = str(self.index.emdb_ids[i])
            if emdb_id in seen:
                continue
            seen.add(emdb_id)
            res = self.index.resolutions[i] if self.index.resolutions is not None else None
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "emdb_id": emdb_id,
                    "score": round(float(scores[i]), 4),
                    "uniprot_accession": str(self.index.uniprot_ids[i]) if self.index.uniprot_ids is not None else None,
                    "resolution_angstrom": None if res is None or np.isnan(res) else round(float(res), 3),
                    "title": (str(self.index.titles[i]) or None) if self.index.titles is not None else None,
                    "url": f"https://www.ebi.ac.uk/emdb/{emdb_id}",
                }
            )
            if len(hits) == k:
                break
        return hits
