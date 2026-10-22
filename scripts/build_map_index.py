"""Build the map index served by ``retrieve_maps``.

Two modes:

  two-tower     Package volume-tower embeddings you computed with the trained model
                (one row per map, same order as the metadata CSV):

                    python scripts/build_map_index.py two-tower \\
                        --embeddings map_emb.npy --metadata metadata_curated.csv \\
                        --model "JEPA+InfoNCE two-tower (human EMDB)" --out map_index.npz

  esm-baseline  No trained checkpoint needed. Each map is represented by the ESM-2
                embedding of its paired UniProt sequence, so retrieval reduces to
                sequence nearest-neighbour search. Useful as a baseline and for demos:

                    python scripts/build_map_index.py esm-baseline \\
                        --metadata metadata_curated.csv --out map_index_esm_baseline.npz

The metadata CSV needs columns ``emdb_id`` and ``uniprot_id``; ``resolution`` and
``title`` are used when present, ``seq`` is required for esm-baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import numpy as np


def read_metadata(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = {"emdb_id", "uniprot_id"} - set(rows[0] if rows else {})
    if missing:
        sys.exit(f"{path} is missing columns: {sorted(missing)}")
    return rows


async def fetch_titles(emdb_ids: list[str], batch: int = 80) -> dict[str, str]:
    from bioretrieval_mcp.http import HttpClient
    from bioretrieval_mcp.sources.emdb import SEARCH_URL

    http = HttpClient(timeout_s=60)
    titles: dict[str, str] = {}
    try:
        for i in range(0, len(emdb_ids), batch):
            chunk = emdb_ids[i : i + batch]
            q = "emdb_id:(" + " OR ".join(f'"{e}"' for e in chunk) + ")"
            text = await http.get(
                SEARCH_URL.format(query=q), params={"rows": len(chunk), "wt": "csv", "fl": "emdb_id,title"}, as_json=False
            )
            for row in csv.DictReader(text.splitlines()):
                titles[row["emdb_id"]] = row["title"]
            print(f"  titles {len(titles)}/{len(emdb_ids)}", file=sys.stderr)
    finally:
        await http.aclose()
    return titles


def save(out: Path, rows: list[dict[str, str]], emb: np.ndarray, kind: str, model: str, titles: dict[str, str]) -> None:
    def res(r: dict[str, str]) -> float:
        try:
            return float(r.get("resolution") or "nan")
        except ValueError:
            return float("nan")

    np.savez_compressed(
        out,
        emdb_ids=np.array([r["emdb_id"] for r in rows]),
        uniprot_ids=np.array([r["uniprot_id"] for r in rows]),
        resolutions=np.array([res(r) for r in rows], dtype=np.float32),
        titles=np.array([r.get("title") or titles.get(r["emdb_id"], "") for r in rows]),
        embeddings=emb.astype(np.float32),
        kind=np.array(kind),
        model=np.array(model),
    )
    print(f"wrote {out}: {len(rows)} maps, dim {emb.shape[1]}, kind={kind}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    tt = sub.add_parser("two-tower")
    tt.add_argument("--embeddings", type=Path, required=True)
    tt.add_argument("--model", default="two-tower")
    eb = sub.add_parser("esm-baseline")
    eb.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    eb.add_argument("--device", default="auto")
    eb.add_argument("--cache-dir", type=Path, default=Path("~/.cache/bioretrieval-mcp/esm").expanduser())
    for p in (tt, eb):
        p.add_argument("--metadata", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--fetch-titles", action="store_true", help="Look up map titles from the EMDB API")
    args = ap.parse_args()

    rows = read_metadata(args.metadata)
    if args.mode == "two-tower":
        emb = np.load(args.embeddings)
        if emb.ndim != 2 or emb.shape[0] != len(rows):
            sys.exit(f"embeddings shape {emb.shape} does not match {len(rows)} metadata rows")
        kind, model = "two_tower", args.model
    else:
        from bioretrieval_mcp.ml.esm import EsmEmbedder
        from bioretrieval_mcp.validation import normalize_sequence

        embedder = EsmEmbedder(args.esm_model, args.cache_dir, args.device)
        valid = []
        for r in rows:
            try:
                r["seq"] = normalize_sequence(r.get("seq") or "")
                valid.append(r)
            except ValueError:
                pass
        print(f"  skipped {len(rows) - len(valid)} rows without a usable sequence", file=sys.stderr)
        rows = valid
        uniq = sorted({r["seq"] for r in rows}, key=len)
        vecs: dict[str, np.ndarray] = {}
        t0 = time.time()
        for i, seq in enumerate(uniq, 1):
            vecs[seq] = embedder.embed(seq)[0]
            if i % 100 == 0 or i == len(uniq):
                print(f"  embedded {i}/{len(uniq)} ({time.time() - t0:.0f}s)", file=sys.stderr)
        emb = np.stack([vecs[r["seq"]] for r in rows])
        kind, model = "esm_seq_baseline", f"ESM-2 sequence nearest-neighbour baseline ({args.esm_model})"

    titles = asyncio.run(fetch_titles([r["emdb_id"] for r in rows])) if args.fetch_titles else {}
    save(args.out, rows, emb, kind, model, titles)


if __name__ == "__main__":
    main()
