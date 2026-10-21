"""Run the LLM judge over a labelled parquet, cache outputs to JSONL, report metrics.

python experiments/run_judge.py --data ../valid.parquet --out runs/think_v1.jsonl --base-url http://127.0.0.1:8080/v1
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
from guardian.features import flags  # noqa: E402
from guardian.judge import Judge  # noqa: E402
from guardian.parse import parse_example  # noqa: E402


def metrics(y, p, thr=0.5):
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b >= thr)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b >= thr)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b < thr)
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return dict(f1=round(2 * pr * rc / max(1e-9, pr + rc), 4), p=round(pr, 3), r=round(rc, 3), tp=tp, fp=fp, fn=fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "valid.parquet"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--mode", default="think", choices=["think", "single"])
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-ctx-chars", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--no-thinking", action="store_true", help="disable model's native <think> block")
    ap.add_argument("--prompt", default=None, help="judge system prompt file")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--layout", default="tail", choices=["tail", "head"])
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--extra", default=None, help="JSON merged into the request body (model-specific fields)")
    ap.add_argument("--stop-verdict", action="store_true", help="stop generation at the first written verdict")
    ap.add_argument("--force-mode", default="turn", choices=["turn", "prefill"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated row indices")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.ids:
        df = df.iloc[[int(x) for x in args.ids.split(",")]]
    if args.limit:
        df = df.head(args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("p") is not None:
                done[r["id"]] = r
    judge = Judge(args.base_url, args.model, mode=args.mode, max_tokens=args.max_tokens,
                  enable_thinking=not args.no_thinking, prompt_path=args.prompt,
                  temperature=args.temperature, layout=args.layout, digest=args.digest,
                  extra_body=json.loads(args.extra) if args.extra else None, force_mode=args.force_mode, stop_at_verdict=args.stop_verdict)

    def work(row):
        if row.id in done:
            return done[row.id]
        notes = None if args.no_notes else flags(parse_example(row.prompt, row.response))[1]
        t0 = time.time()
        r = judge.score(row.prompt, row.response, notes, args.max_ctx_chars)
        r.update(id=row.id, label=int(row.label) if "label" in row._fields else None, sec=round(time.time() - t0, 1))
        with open(out, "a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{row.id:60s} y={r['label']} p={r['p']} v={r['verdict']} {r['sec']}s "
              f"ptok={r.get('prompt_tokens')} ctok={r.get('completion_tokens')}", flush=True)
        return r

    rows = list(df.itertuples(index=False))
    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(work, rows))
    y = [r["label"] for r in res]
    p = [0.5 if r["p"] is None else r["p"] for r in res]
    print("N", len(res), "failed", sum(r["p"] is None for r in res))
    print("metrics@0.5", metrics(y, p))


if __name__ == "__main__":
    main()
