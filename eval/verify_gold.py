"""Re-derive every gold fact from live tool output (no LLM involved).

For each question, runs its ``gold_source`` tool calls against the real server
(live EMDB / UniProt / GEO, real ESM-2, the configured map index) and checks that
every key-fact check passes on the concatenated tool output. This is how the gold
answers were grounded, and how to re-snapshot them later as databases grow.

    BIORETRIEVAL_MAP_INDEX=artifacts/map_index_esm_baseline.npz python eval/verify_gold.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client

sys.path.insert(0, str(Path(__file__).parent))
from checks import run_checks  # noqa: E402

from bioretrieval_mcp.server import create_server  # noqa: E402

HERE = Path(__file__).parent


async def resolve_args(client: Client, args: dict) -> dict:
    args = dict(args)
    if acc := args.pop("sequence_from", None):
        r = await client.call_tool("get_uniprot", {"accession": acc})
        args["sequence"] = r.structured_content["sequence"]
    return args


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    args = ap.parse_args()
    qs = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    failures = 0
    async with Client(create_server(), read_timeout_seconds=600) as client:
        available = {t.name for t in (await client.list_tools()).tools}
        for q in qs:
            texts, problems = [], []
            for src in q["gold_source"]:
                if src["tool"] not in available:
                    problems.append(f"tool {src['tool']} not enabled")
                    continue
                r = await client.call_tool(src["tool"], await resolve_args(client, src["args"]))
                if bool(r.is_error) != bool(src.get("expect_error")):
                    problems.append(f"{src['tool']} is_error={r.is_error}: {r.content[0].text[:120]}")
                texts.append(r.content[0].text)
            ok, detail = run_checks(q["checks"], "\n".join(texts))
            if not ok:
                problems.append(f"checks {detail}")
            failures += bool(problems)
            print(f"{q['id']} {'OK ' if not problems else 'FAIL'} {'; '.join(problems)}")
    print(f"\n{len(qs) - failures}/{len(qs)} gold answers grounded in current tool output")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
