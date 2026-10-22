"""Run the question set with Claude as the MCP client and record full traces.

Claude sees exactly the tool list the MCP server advertises (names, descriptions,
JSON schemas) and every tool call goes over a real MCP stdio session to
``bioretrieval-mcp``, so this measures the server as an agent would use it.

    python eval/run_eval.py --out eval/results/run.jsonl
    python eval/run_eval.py --only q01,q02 --out /tmp/smoke.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
from mcp import Client, StdioServerParameters

HERE = Path(__file__).parent
SYSTEM = """\
You are a research assistant for structural biology and DNA-methylation studies. You have tools \
for EMDB, UniProt, ESM-2 embeddings, sequence-to-cryo-EM-map retrieval, and GEO methylation series.
Ground every factual statement in tool output; do not rely on memory for accessions, counts, \
resolutions, or sample sizes. Cite the accession IDs you relied on. If a tool returns an error or \
nothing relevant, say so plainly instead of guessing. Keep the final answer concise."""
MAX_TOOL_RESULT_CHARS = 12000


def load_questions(path: Path, only: set[str] | None) -> list[dict[str, Any]]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [q for q in qs if not only or q["id"] in only]


def to_anthropic_tools(mcp_tools) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in mcp_tools]


def tool_result_text(result) -> str:
    text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + f"\n...[truncated {len(text) - MAX_TOOL_RESULT_CHARS} chars]"
    return text


async def run_one(
    q: dict[str, Any],
    claude: anthropic.AsyncAnthropic,
    mcp: Client,
    tools: list[dict[str, Any]],
    args: argparse.Namespace,
    on_call=None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": q["question"]}]
    calls: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    t0 = time.time()
    stop_reason = None
    answer = ""
    for turn in range(args.max_turns):
        response = await claude.messages.create(
            model=args.model,
            max_tokens=16000,
            system=SYSTEM,
            tools=tools,
            messages=messages,
            output_config={"effort": args.effort},
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        stop_reason = response.stop_reason
        # Append the full content (thinking blocks included) so history stays append-only.
        messages.append({"role": "assistant", "content": response.content})
        if stop_reason != "tool_use":
            answer = "\n".join(b.text for b in response.content if b.type == "text").strip()
            break
        uses = [b for b in response.content if b.type == "tool_use"]

        async def execute(block, turn=turn):
            started = time.time()
            try:
                r = await mcp.call_tool(block.name, block.input)
                text, is_error = tool_result_text(r), bool(r.is_error)
            except Exception as exc:  # transport-level failure: report to the model, keep going
                text, is_error = f"MCP call failed: {exc!r}", True
            calls.append(
                {"turn": turn, "name": block.name, "input": block.input, "is_error": is_error,
                 "output": text, "latency_s": round(time.time() - started, 3)}
            )
            if on_call:
                on_call(calls[-1])
            return {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error}

        results = await asyncio.gather(*(execute(b) for b in uses))
        messages.append({"role": "user", "content": list(results)})
    else:
        stop_reason = "max_turns"

    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "model": args.model,
        "effort": args.effort,
        "answer": answer,
        "stop_reason": stop_reason,
        "tool_calls": calls,
        "usage": usage,
        "latency_s": round(time.time() - t0, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "run.jsonl")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated question ids")
    args = ap.parse_args()

    questions = load_questions(args.questions, set(filter(None, args.only.split(","))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():  # resume
        done = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line.strip()}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(todo)} to run ({len(done)} already in {args.out})", file=sys.stderr)

    server = StdioServerParameters(
        command=sys.executable, args=["-m", "bioretrieval_mcp.server", "--log-level", "WARNING"], env=dict(os.environ)
    )
    claude = anthropic.AsyncAnthropic(max_retries=4)
    sem = asyncio.Semaphore(args.concurrency)
    async with Client(server, read_timeout_seconds=300) as mcp:
        tools = to_anthropic_tools((await mcp.list_tools()).tools)
        print("tools:", [t["name"] for t in tools], file=sys.stderr)

        async def guarded(q):
            async with sem:
                try:
                    rec = await run_one(q, claude, mcp, tools, args)
                except anthropic.APIStatusError as exc:
                    rec = {"id": q["id"], "category": q["category"], "question": q["question"], "answer": "",
                           "stop_reason": f"api_error_{exc.status_code}", "tool_calls": [], "error": str(exc)}
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{q['id']}: {rec['stop_reason']}, {len(rec['tool_calls'])} calls", file=sys.stderr)

        await asyncio.gather(*(guarded(q) for q in todo))


if __name__ == "__main__":
    asyncio.run(main())
