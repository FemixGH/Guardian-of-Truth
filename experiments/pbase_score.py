"""P_base: the organisers' baseline prompt, one token without reasoning, P('1') from top logprobs.
Exact baseline layout (system = baseline_prompt.md, user = <prompt>…</prompt>\n<response>…</response>) or
layout "tail" (our judge's shared context prefix + baseline instructions at the end -> prefix-cache hit).
    python experiments/pbase_score.py --data X.parquet --out X.jsonl [--layout exact|tail] [--concurrency 96]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardian.judge import SHARED_SYSTEM  # noqa: E402

SYSTEM = (ROOT / "experiments" / "baseline_prompt.md").read_text(encoding="utf-8").strip()


def messages(prompt: str, response: str, layout: str) -> list[dict]:
    if layout == "exact":
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<prompt>\n{prompt}\n</prompt>\n<response>\n{response}\n</response>"}]
    user = (f"<context>\n{prompt}\n</context>\n\n<turn_under_review>\n{response}\n</turn_under_review>\n"
            "\n=== AUDIT INSTRUCTIONS ===\n" + SYSTEM)
    return [{"role": "system", "content": SHARED_SYSTEM}, {"role": "user", "content": user}]


def p_one(top) -> float | None:
    """P('1') renormalised over the digits 0/1 among the top logprobs."""
    pr = {"0": 0.0, "1": 0.0}
    for t in top:
        k = t.token.strip()
        if k in pr:
            pr[k] += math.exp(t.logprob)
    s = pr["0"] + pr["1"]
    return pr["1"] / s if s > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="judge")
    ap.add_argument("--layout", default="exact", choices=["exact", "tail"])
    ap.add_argument("--concurrency", type=int, default=96)
    a = ap.parse_args()
    client = openai.OpenAI(base_url=a.base_url, api_key="x", timeout=600)
    df = pd.read_parquet(a.data)
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()

    def work(row):
        if row.id in done:
            return
        try:
            r = client.chat.completions.create(
                model=a.model, messages=messages(row.prompt or "", row.response or "", a.layout), temperature=0.0,
                max_tokens=1, logprobs=True, top_logprobs=20, seed=1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            lp = r.choices[0].logprobs.content
            p = p_one(lp[0].top_logprobs) if lp else None
            rec = {"id": row.id, "p": p, "prompt_tokens": r.usage.prompt_tokens}
        except Exception as e:  # noqa: BLE001
            rec = {"id": row.id, "p": None, "error": str(e)[:200]}
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(work, df.itertuples(index=False)))
    print("done", a.out)


if __name__ == "__main__":
    main()
