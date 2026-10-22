"""Optional CpGPT sample embeddings.

CpGPT inference needs the sample's full beta-value vector plus the model and its
DNA-language-model locus embeddings, which is far too heavy to run per tool call.
Instead the server serves embeddings precomputed offline with CpGPT's own
``CpGPTTrainer.predict`` (see ``scripts/export_cpgpt_embeddings.py``):

    cpgpt_embeddings.npz
      sample_ids   (N,)   str   e.g. GSM accessions
      embeddings   (N, D) float32
      model        ()     str   e.g. "cpgpt-small"

The tool is only registered when ``BIORETRIEVAL_CPGPT_EMBEDDINGS`` points at such a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CpgptStore:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            ids = z["sample_ids"].astype(str)
            self.embeddings = z["embeddings"].astype(np.float32)
            self.model = str(z["model"]) if "model" in z.files else "cpgpt"
        self.row = {sid: i for i, sid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.row)

    def get(self, sample_id: str) -> np.ndarray:
        try:
            return self.embeddings[self.row[sample_id]]
        except KeyError:
            raise KeyError(
                f"Sample '{sample_id}' is not in the CpGPT embedding store ({len(self)} samples)."
            ) from None
