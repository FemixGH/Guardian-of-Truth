"""Deterministic digest of the turn under review: what the turn does + the most relevant preconditions.

Appended after the full context (never instead of it), so the judge focuses on what is NEW in this turn.
"""
from __future__ import annotations

import json
import re

from .parse import parse_example
from .policy import TIME_RE


def _tool_schema(system: str, name: str, max_chars: int = 900) -> str:
    i = system.find("[AVAILABLE TOOLS]")
    if i < 0:
        return ""
    block = system[i:]
    m = re.search(r"^- %s\s+—.*?(?=^- [A-Za-z_]|\Z)" % re.escape(name), block, re.M | re.S)
    return m.group(0).strip()[:max_chars] if m else ""


def _short(obj, n: int) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n] + " …"


def turn_digest(prompt: str, response: str) -> str:
    try:
        ex = parse_example(prompt or "", response or "")
    except Exception:  # noqa: BLE001
        return ""
    lines = [f"Domain: {ex.domain}."]
    m = TIME_RE.search(ex.system)
    if m:
        lines.append(f"Current time stated in the policy: {m.group(1)} {m.group(2)}.")
    calls = ex.resp_calls
    if calls:
        lines.append("Tool calls made in this turn:")
        for c in calls:
            avail = "listed in [AVAILABLE TOOLS]" if c.name in ex.tools else "NOT listed in [AVAILABLE TOOLS]"
            lines.append(f"  * {c.name}({_short(c.args if c.args is not None else c.raw_args, 500)}) — {avail}")
            schema = _tool_schema(ex.system, c.name)
            if schema:
                lines.append("    schema: " + _short(schema, 700))
    else:
        lines.append("This turn makes no tool calls (text reply only).")
    if ex.resp_text:
        lines.append("Reply text length: %d characters." % len(ex.resp_text))
    if ex.last_user:
        lines.append("Last user message: " + _short(ex.last_user, 900))
    last_results = []
    for msg in reversed(ex.history):
        if msg.role == "ASSISTANT" and msg.responses:
            last_results = msg.responses
            break
    if last_results:
        lines.append("Tool results right before this turn:")
        for r in last_results[:4]:
            lines.append(f"  * {r.name}{' [ERROR]' if r.error else ''}: {_short(r.text, 600)}")
    return "\n".join(lines)
