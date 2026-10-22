"""Ask one research question through Claude + bioretrieval-mcp and show the tool trace live.

    python eval/demo.py "Which cryo-EM maps best match the sequence of KRAS (P01116)?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import anthropic
from mcp import Client, StdioServerParameters

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import run_one, to_anthropic_tools  # noqa: E402

DIM, BOLD, CYAN, GREEN, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[31m", "\033[0m"


def show_call(call: dict) -> None:
    args = json.dumps(call["input"])
    if len(args) > 90:
        args = args[:87] + "..."
    status = f"{RED}error{RESET}" if call["is_error"] else f"{GREEN}ok{RESET}"
    print(f"  {CYAN}-> {call['name']}{RESET}{DIM}{args}{RESET}  {status} {DIM}{call['latency_s']:.2f}s{RESET}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-turns", type=int, default=10)
    args = ap.parse_args()
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "bioretrieval_mcp.server", "--log-level", "WARNING"], env=dict(os.environ)
    )
    async with Client(server, read_timeout_seconds=300) as mcp:
        tools = to_anthropic_tools((await mcp.list_tools()).tools)
        print(f"{DIM}bioretrieval-mcp tools: {', '.join(t['name'] for t in tools)}{RESET}")
        print(f"{BOLD}Q:{RESET} {args.question}")
        q = {"id": "demo", "category": "demo", "question": args.question}
        rec = await run_one(q, anthropic.AsyncAnthropic(), mcp, tools, args, on_call=show_call)
    print(f"\n{BOLD}A:{RESET} {rec['answer']}")
    print(f"{DIM}{len(rec['tool_calls'])} tool calls, {rec['latency_s']}s{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
