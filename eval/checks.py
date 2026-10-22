"""Deterministic answer checks shared by the scorer and the gold verifier.

A question's ``checks`` is a list of:
  {"type": "all_of", "values": [...]}             every string appears (case-insensitive)
  {"type": "any_of", "values": [...]}             at least one appears
  {"type": "number", "value": 1.15, "tol": 0.01}  some number in the text is within tol
"""

from __future__ import annotations

import re
from typing import Any

_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM.findall(text)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("å", "a").replace("ångström", "angstrom"))


def run_check(check: dict[str, Any], text: str) -> bool:
    t = _norm(text)
    kind = check["type"]
    if kind == "all_of":
        return all(_norm(v) in t for v in check["values"])
    if kind == "any_of":
        return any(_norm(v) in t for v in check["values"])
    if kind == "number":
        tol = check.get("tol", 0)
        return any(abs(n - check["value"]) <= tol for n in numbers(text))
    raise ValueError(f"unknown check type {kind}")


def run_checks(checks: list[dict[str, Any]], text: str) -> tuple[bool, list[bool]]:
    results = [run_check(c, text) for c in checks]
    return all(results), results
