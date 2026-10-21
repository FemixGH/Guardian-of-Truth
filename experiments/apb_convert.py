"""AgentProcessBench (τ²-Bench subset) -> Guardian transcript rows (prompt = context before the step, response = step).

Labels (human step labels of AgentProcessBench, mapped to "new hallucination in this turn"):
  first -1 of a trajectory  -> label 1 (candidate; re-checked manually, see apb_manual_labels.json)
  +1 steps before it        -> label 0
  neutral (0) steps, steps after the first error, the greeting -> dropped

python experiments/apb_convert.py --src ../data_ext/agentprocessbench/tau2/test.jsonl --out runs/apb_tau2.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

INSTRUCTIONS = """<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>"""


def _type(p: dict, defs: dict) -> tuple[str, dict | None]:
    if "$ref" in p:
        return "object", defs.get(p["$ref"].split("/")[-1])
    if "anyOf" in p:
        ts = [x for x in p["anyOf"] if x.get("type") != "null"]
        return _type(ts[0], defs) if ts else ("string", None)
    t = p.get("type", "string")
    if t == "array":
        it = p.get("items", {})
        if "$ref" in it:
            return "array", defs.get(it["$ref"].split("/")[-1])
        if it.get("type") == "object":
            return "array", it
    if t == "object" and "properties" in p:
        return "object", p
    return t, None


def tools_block(tools: list) -> str:
    out = []
    for t in tools:
        f = t.get("function", t)
        desc = (f.get("description") or "").strip().split("\n")
        out.append(f"- {f['name']} — {desc[0]}")
        out += ["    " + d for d in desc[1:] if d.strip()]
        params = f.get("parameters") or {}
        defs = params.get("$defs", {})
        req = set(params.get("required", []))
        for name, p in (params.get("properties") or {}).items():
            t_, sub = _type(p, defs)
            enum = f" [enum: {'|'.join(map(str, p['enum']))}]" if p.get("enum") else ""
            out.append(f"    {name}: {t_}{'!' if name in req else ''}{enum} — {(p.get('description') or '').strip()}")
            if sub:
                sreq = set(sub.get("required", []))
                for cn, cp in (sub.get("properties") or {}).items():
                    ct, _ = _type(cp, defs)
                    out.append(f"        · {cn}: {ct}{'!' if cn in sreq else ''} — {(cp.get('description') or '').strip()}")
    return "\n".join(out)


def _args(a) -> str:
    try:
        return json.dumps(json.loads(a) if isinstance(a, str) else a, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(a)


def _calls_and_results(msg: dict, results: dict) -> list[str]:
    lines, res = [], []
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function", tc)
        lines.append(f"\t→ TOOL_CALL {f['name']}: {_args(f.get('arguments'))}")
        r = results.get(tc.get("id"))
        if r is not None:
            text = str(r.get("content") or "")
            err = " [ERROR]" if text.lower().startswith(("error", '{"error"')) else ""
            body = text.split("\n")
            res.append(f"\t← TOOL_RESPONSE {f['name']}{err}: {body[0]}")
            res += ["\t  " + b for b in body[1:]]
    return lines + res


def convert(traj: dict) -> list[dict]:
    msgs = traj["messages"]
    results = {m.get("tool_call_id"): m for m in msgs if m["role"] == "tool"}
    labels = {int(k): v for k, v in traj["step_labels"].items()}
    first_err = min((k for k, v in labels.items() if v == -1), default=None)
    dom = traj["data_source"].replace("tau2_", "")
    tid = f"apb_{dom}_{traj['query_index']}_{traj['sample_index']}"
    blocks, rows, turn = [], [], 0
    for i, m in enumerate(msgs):
        if m["role"] == "system":
            blocks.append(f"⟦SYSTEM⟧\n{INSTRUCTIONS}\n<policy>\n{m['content'].strip()}\n</policy>\n\n"
                          f"[AVAILABLE TOOLS]\n{tools_block(traj['tools'])}\n\n")
        elif m["role"] == "user":
            body = (m.get("content") or "").strip()
            blocks.append("\n".join(["⟦USER⟧", body] + _calls_and_results(m, results)) + "\n\n")
        elif m["role"] == "assistant":
            content = (m.get("content") or "").strip()
            if turn == 0 and i <= 2 and not m.get("tool_calls"):  # greeting, unnumbered as in the dataset
                blocks.append(f"⟦ASSISTANT⟧\n{content}\n\n")
                turn = 0.5
                continue
            turn = int(turn) + 1
            head = f"⟦ASSISTANT · ход {turn}⟧"
            call_lines = [f"\t→ TOOL_CALL {tc.get('function', tc)['name']}: {_args(tc.get('function', tc).get('arguments'))}"
                          for tc in (m.get("tool_calls") or [])]
            response = "\n".join([head] + ([content] if content else []) + call_lines) + "\n"
            lab = labels.get(i)
            keep = None
            if lab is not None:
                if first_err is not None and i == first_err:
                    keep = 1
                elif lab == 1 and (first_err is None or i < first_err):
                    keep = 0
            if keep is not None:
                rows.append(dict(id=f"{tid}::t{turn}", traj=tid, task=f"{dom}_{traj['query_index']}", domain=dom,
                                 prompt="".join(blocks), response=response, label=keep, step=i))
            blocks.append("\n".join([head] + ([content] if content else []) + _calls_and_results(m, results)) + "\n\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[2] / "data_ext/agentprocessbench/tau2/test.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/apb_tau2.parquet"))
    args = ap.parse_args()
    rows = []
    for line in open(args.src):
        rows += convert(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(len(df), "rows |", df.label.value_counts().to_dict(), "| by domain:",
          df.groupby(["domain", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
