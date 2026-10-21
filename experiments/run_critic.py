"""Run the critic on rows flagged by judge passes (excluding hard-flag rows) and store its verdicts.

python experiments/run_critic.py --data rows.parquet --flags v3b.jsonl v2.jsonl@0 --out critic.jsonl --base-url ...
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian import combine  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian.judge import Judge, finding_of  # noqa: E402
from guardian.parse import parse_example  # noqa: E402


def load(spec):
    path, _, ps = spec.partition("@")
    out = {}
    for line in open(path):
        r = json.loads(line)
        if ps and str(r.get("pass")) != ps:
            continue
        v = r.get("verdict")
        flagged = v == 1 if v is not None else (r.get("p") or 0) >= 0.5
        if flagged and r["id"] not in out:
            out[r["id"]] = finding_of(r.get("reasoning") or "", r.get("text") or "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--flags", nargs="+", required=True, help="judge jsonl files (first has priority for the finding)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--prompt", default=None)
    a = ap.parse_args()
    df = pd.read_parquet(a.data).set_index("id")
    cfg = combine.load(ROOT / "model" / "calibration.json")
    findings = {}
    for spec in a.flags:
        for i, f in load(spec).items():
            findings.setdefault(i, f)
    todo = [i for i in findings if i in df.index
            and combine.final_p(flags(parse_example(df.prompt[i], df.response[i]))[0], 0.0, cfg) < 1.0]
    print(f"{len(findings)} flagged rows, {len(todo)} without hard flags -> critic", flush=True)
    judge = Judge(a.base_url, "qwen", mode="think", max_tokens=4096, layout="head")

    def work(i):
        t0 = time.time()
        r = judge.critic(df.prompt[i], df.response[i], findings[i], critic_path=a.prompt)
        rec = {"id": i, "label": int(df.label[i]) if "label" in df.columns else None, "p": r.get("p"),
               "verdict": r.get("verdict"), "category": r.get("category"), "finding": findings[i], "reasoning": r.get("reasoning"),
               "sec": round(time.time() - t0, 1)}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(work, todo))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
