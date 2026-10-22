"""Score a run: tool-selection accuracy, answer correctness, Ragas-style faithfulness.

Metrics
-------
Tool selection   A question passes when every tool in ``expected_tools`` was called.
                 Also reported: first-call accuracy (the first tool called is expected) and
                 call precision (share of calls that went to expected/allowed tools).
Correctness      (a) key-fact checks: deterministic string/number checks against gold facts;
                 (b) LLM judge: Claude grades the answer against the gold answer as
                     correct / partially_correct / incorrect. Headline = judge "correct".
Faithfulness     Ragas definition: split the answer into atomic claims, then ask whether each
                 claim can be inferred from the retrieved context (here: the tool descriptions the
                 agent was shown plus the tool outputs it received). Score = supported / total.

    python eval/score.py eval/results/run.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import anthropic
from anthropic.lib._parse._transform import transform_schema
from mcp import Client, StdioServerParameters
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).parent))
from checks import run_checks  # noqa: E402

HERE = Path(__file__).parent
MAX_CONTEXT_CHARS = 60000


class Judgement(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    rationale: str = Field(description="One or two sentences")


class Claims(BaseModel):
    claims: list[str] = Field(description="Atomic, self-contained factual statements made in the answer")


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool = Field(description="True only if the claim can be directly inferred from the context")


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


JUDGE_PROMPT = """You are grading a research assistant's answer against a gold answer.
The gold answer was produced from the same databases on {snapshot}; databases grow over time, so
treat additional correct items beyond the gold as fine, but contradictions or missing key facts are not.

Question: {question}

Gold answer: {gold}

Assistant answer: {answer}

Grade "correct" if the answer contains the key facts of the gold answer with no material errors,
"partially_correct" if some key facts are right but others are missing or wrong, else "incorrect"."""

CLAIMS_PROMPT = """Break the answer below into atomic, self-contained factual claims (one fact each,
pronouns resolved). Skip hedges, offers of further help, and statements about what the assistant did.

Question: {question}

Answer: {answer}"""

VERIFY_PROMPT = """Context (tool descriptions the assistant was shown, then tool outputs it received):
<context>
{context}
</context>

For each claim, decide whether it can be directly inferred from the context above. General
background knowledge does not count as support. Return one verdict per claim, in order.

Claims:
{claims}"""


class JudgeError(RuntimeError):
    """The grader returned no structured output (refusal, max_tokens, ...)."""


# Opus 5.5 safety classifiers occasionally decline benign life-science text (here: protein/sequence
# facts), sometimes mid-output, which leaves truncated JSON. For the *grader* we opt into server-side
# fallbacks so a declined grading call is re-run on the category-appropriate fallback model; which
# model actually graded each item is recorded. The agent under test never uses fallbacks.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
GRADED_BY: dict[str, int] = defaultdict(int)


async def parse(client, model, prompt, schema, attempts: int = 2):
    fmt = {"type": "json_schema", "schema": transform_schema(schema)}
    reason = ""
    for _ in range(attempts):
        resp = await client.beta.messages.create(
            model=model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium", "format": fmt},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            reason = f"stop_reason=refusal category={category}"
            continue
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            parsed = schema.model_validate_json(text)
        except ValidationError:
            reason = f"stop_reason={resp.stop_reason}, unparseable output ({len(text)} chars)"
            continue
        GRADED_BY[resp.model] += 1
        return parsed
    raise JudgeError(f"{schema.__name__}: no structured output after {attempts} attempts ({reason})")


# The agent's grounding sources are the tool descriptions it was shown plus the tool outputs it received.
TOOL_DOCS = ""


async def load_tool_docs() -> str:
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "bioretrieval_mcp.server", "--log-level", "WARNING"], env=dict(os.environ)
    )
    async with Client(server, read_timeout_seconds=120) as mcp:
        tools = (await mcp.list_tools()).tools
    return "\n\n".join(f"[tool description: {t.name}]\n{t.description}" for t in tools)


def tool_context(rec) -> str:
    parts = [f"[{c['name']}({json.dumps(c['input'])})]\n{c['output']}" for c in rec["tool_calls"]]
    ctx = TOOL_DOCS + "\n\n" + "\n\n".join(parts)
    return ctx[:MAX_CONTEXT_CHARS]


async def faithfulness(client, model, rec) -> tuple[float | None, list[dict]]:
    if not rec["answer"]:
        return None, []
    claims = (await parse(client, model, CLAIMS_PROMPT.format(question=rec["question"], answer=rec["answer"]), Claims)).claims
    if not claims:
        return None, []
    listing = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    verdicts = (await parse(client, model, VERIFY_PROMPT.format(context=tool_context(rec), claims=listing), ClaimVerdicts)).verdicts
    supported = sum(v.supported for v in verdicts)
    return supported / max(len(verdicts), 1), [v.model_dump() for v in verdicts]


def tool_metrics(q, rec) -> dict:
    called = [c["name"] for c in rec["tool_calls"]]
    expected = set(q["expected_tools"])
    allowed = expected | set(q.get("allowed_tools", []))
    return {
        "called": called,
        "tool_selection_ok": expected <= set(called) if expected else not called,
        "first_call_ok": bool(called) and called[0] in expected if expected else not called,
        "call_precision": (sum(c in allowed for c in called) / len(called)) if called else (1.0 if not expected else 0.0),
        "n_calls": len(called),
        "n_tool_errors": sum(c["is_error"] for c in rec["tool_calls"]),
    }


async def score_one(client, model, q, rec, sem, snapshot):
    async with sem:
        out = {"id": q["id"], "category": q["category"], **tool_metrics(q, rec)}
        out["facts_ok"], out["facts_detail"] = run_checks(q["checks"], rec["answer"])
        out["scoring_errors"] = []
        if rec["answer"]:
            try:
                j = await parse(
                    client,
                    model,
                    JUDGE_PROMPT.format(snapshot=snapshot, question=q["question"], gold=q["gold_answer"], answer=rec["answer"]),
                    Judgement,
                )
                out["judge"], out["judge_rationale"] = j.verdict, j.rationale
            except JudgeError as exc:
                out["judge"], out["judge_rationale"] = "error", str(exc)
                out["scoring_errors"].append(str(exc))
        else:
            out["judge"], out["judge_rationale"] = "incorrect", f"no answer (stop_reason={rec.get('stop_reason')})"
        try:
            out["faithfulness"], out["claims"] = await faithfulness(client, model, rec)
        except JudgeError as exc:
            out["faithfulness"], out["claims"] = None, []
            out["scoring_errors"].append(str(exc))
        if out["scoring_errors"]:
            print(f"{q['id']}: scoring error: {'; '.join(out['scoring_errors'])}", file=sys.stderr)
        out["latency_s"] = rec.get("latency_s")
        out["usage"] = rec.get("usage")
        return out


def pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):.1f}%" if xs else "n/a"


def summarize(rows: list[dict], meta: dict) -> str:
    def block(rs):
        faith = [r["faithfulness"] for r in rs if r["faithfulness"] is not None]
        return {
            "n": len(rs),
            "tool_selection": pct(r["tool_selection_ok"] for r in rs),
            "first_call": pct(r["first_call_ok"] for r in rs),
            "call_precision": f"{100 * statistics.mean(r['call_precision'] for r in rs):.1f}%",
            "facts": pct(r["facts_ok"] for r in rs),
            "judge_correct": pct(r["judge"] == "correct" for r in rs if r["judge"] != "error"),
            "judge_correct_or_partial": pct(r["judge"] in ("correct", "partially_correct") for r in rs if r["judge"] != "error"),
            "faithfulness": f"{statistics.mean(faith):.3f}" if faith else "n/a",
        }

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cols = ["n", "tool_selection", "first_call", "call_precision", "facts", "judge_correct", "judge_correct_or_partial", "faithfulness"]
    lines = [
        f"Agent model: `{meta['agent_model']}` (effort {meta['effort']}), judge: `{meta['judge_model']}`, "
        f"gold snapshot {meta['snapshot']}.",
        "",
        "| slice | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for name, rs in [("**all**", rows)] + sorted(by_cat.items()):
        b = block(rs)
        lines.append(f"| {name} | " + " | ".join(str(b[c]) for c in cols) + " |")
    calls = [r["n_calls"] for r in rows]
    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    graded = meta.get("grading_calls_by_model") or {}
    if set(graded) - {meta["judge_model"]}:
        lines += ["", "Grading calls by model (server-side fallback on safety declines): "
                  + ", ".join(f"`{m}` {n}" for m, n in sorted(graded.items()))]
    n_err = sum(bool(r.get("scoring_errors")) for r in rows)
    if n_err:
        lines += ["", f"Scoring errors (grader returned no output; excluded from judge/faithfulness columns): {n_err}"]
    lines += ["", f"Mean tool calls/question: {statistics.mean(calls):.2f}; "
              f"median latency: {statistics.median(lat):.1f}s" if lat else ""]
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    ap.add_argument("--judge-model", default="claude-opus-5-5")
    ap.add_argument("--snapshot", default="2026-10-21")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    qs = {q["id"]: q for q in map(json.loads, filter(str.strip, args.questions.read_text().splitlines()))}
    recs = [json.loads(line) for line in args.run.read_text().splitlines() if line.strip()]
    global TOOL_DOCS
    TOOL_DOCS = await load_tool_docs()
    client = anthropic.AsyncAnthropic(max_retries=4)
    sem = asyncio.Semaphore(args.concurrency)
    # Resume: questions already scored cleanly are kept; the rest are (re)scored and saved as they finish.
    partial = args.run.with_suffix(".scores.partial.jsonl")
    done = {}
    if partial.exists():
        for line in filter(str.strip, partial.read_text().splitlines()):
            row = json.loads(line)
            if not row.get("scoring_errors"):
                done[row["id"]] = row

    async def score_and_save(r):
        row = await score_one(client, args.judge_model, qs[r["id"]], r, sem, args.snapshot)
        with partial.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    todo = [r for r in recs if r["id"] not in done]
    print(f"scoring {len(todo)} ({len(done)} already scored)", file=sys.stderr)
    rows = list(done.values()) + list(await asyncio.gather(*(score_and_save(r) for r in todo)))
    rows = sorted(rows, key=lambda r: r["id"])
    meta = {
        "agent_model": recs[0].get("model", "?"),
        "effort": recs[0].get("effort", "?"),
        "judge_model": args.judge_model,
        "snapshot": args.snapshot,
        "grading_calls_by_model": dict(GRADED_BY),
    }
    out = args.run.with_suffix(".scores.json")
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    md = summarize(rows, meta)
    args.run.with_suffix(".summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    asyncio.run(main())
