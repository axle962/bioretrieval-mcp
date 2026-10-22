"""Package CpGPT sample embeddings for the optional ``cpgpt_embed`` tool.

Run CpGPT inference in the CpGPT environment (its documented forward mode):

    preds = trainer.predict(model, dataloader, predict_mode="forward",
                            return_keys=["sample_embeddings"])
    torch.save(preds["sample_embeddings"], "sample_embeddings.pt")

and write the sample IDs, one per line in dataloader order, to ``sample_ids.txt``. Then:

    python scripts/export_cpgpt_embeddings.py --embeddings sample_embeddings.pt \\
        --ids sample_ids.txt --model cpgpt-small --out artifacts/cpgpt_embeddings.npz

and start the server with ``BIORETRIEVAL_CPGPT_EMBEDDINGS=artifacts/cpgpt_embeddings.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    import torch

    t = torch.load(path, map_location="cpu")
    return t.float().numpy() if hasattr(t, "numpy") else np.asarray(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", type=Path, required=True, help=".pt or .npy, shape (N, D)")
    ap.add_argument("--ids", type=Path, required=True, help="text file, one sample id per line")
    ap.add_argument("--model", default="cpgpt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    emb = load_matrix(args.embeddings)
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if emb.ndim != 2 or emb.shape[0] != len(ids):
        sys.exit(f"embeddings shape {emb.shape} does not match {len(ids)} sample ids")
    if len(set(ids)) != len(ids):
        sys.exit("sample ids are not unique")
    np.savez_compressed(args.out, sample_ids=np.array(ids), embeddings=emb.astype(np.float32), model=np.array(args.model))
    print(f"wrote {args.out}: {len(ids)} samples, dim {emb.shape[1]}", file=sys.stderr)


if __name__ == "__main__":
    main()
