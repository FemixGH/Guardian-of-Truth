"""Metrics of one judge model (run_judge.py outputs) on our offline checks.

valid (46 labelled turns of timing_431) TP/FP/FN, flag rate on the 385 unlabelled early turns (predicts the test flag
rate), APB-ru eval half (417 rows, 49 hallucinations): recall, FPR and F1 projected to the test positive rate 23.7%,
judge-only and with the deterministic hard flags (as in the submission); reasoning length.

python experiments/lora/eval_report.py --name it1 --t431 it1_t431.jsonl --apb it1_apb.jsonl [--out report.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from guardian import combine  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian.parse import parse_example  # noqa: E402

BASE_RATE = 0.237


def load(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--t431", required=True)
    ap.add_argument("--apb", required=True)
    ap.add_argument("--t431-data", default=str(ROOT / "runs/timing_431_labeled.parquet"))
    ap.add_argument("--apb-data", default=str(ROOT / "runs/lora/apb_eval2_ru.parquet"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = combine.load(str(ROOT / "model/calibration.json"))
    rep = {"name": a.name}
    t = pd.read_parquet(a.t431_data)
    r = load(a.t431)
    fl = {i: (r[i].get("p") or 0) >= 0.5 for i in r}
    lab = dict(zip(t.id, t.label))
    v = [(lab[i], fl[i]) for i in fl if lab.get(i, -9) >= 0]
    rep["valid_tp"] = sum(y == 1 and f for y, f in v)
    rep["valid_fp"] = sum(y == 0 and f for y, f in v)
    rep["valid_fn"] = sum(y == 1 and not f for y, f in v)
    early = [fl[i] for i in fl if lab.get(i) == -1]
    rep["early_flag_rate"] = round(float(np.mean(early)), 3) if early else None
    rep["t431_no_verdict"] = sum(1 for x in r.values() if x.get("p") is None)
    d = pd.read_parquet(a.apb_data)
    q = load(a.apb)
    hard = {x.id for x in d.itertuples() if combine.final_p(flags(parse_example(x.prompt, x.response))[0], 0.0, cfg) >= 1.0}
    for tag, use_hard in (("judge", False), ("with_hard", True)):
        f = {i: ((q.get(i, {}).get("p") or 0) >= 0.5) or (use_hard and i in hard) for i in d.id}
        pos = [i for i in d.id if dict(zip(d.id, d.label))[i] == 1]
        neg = [i for i in d.id if dict(zip(d.id, d.label))[i] == 0]
        rec = sum(f[i] for i in pos) / len(pos)
        fpr = sum(f[i] for i in neg) / len(neg)
        prec = rec * BASE_RATE / max(1e-9, rec * BASE_RATE + fpr * (1 - BASE_RATE))
        rep[f"apb_{tag}_recall"] = round(rec, 3)
        rep[f"apb_{tag}_fpr"] = round(fpr, 3)
        rep[f"apb_{tag}_f1@23.7"] = round(2 * prec * rec / max(1e-9, prec + rec), 3)
    toks = [x.get("completion_tokens") or 0 for x in list(r.values()) + list(q.values())]
    rep["median_completion_tokens"] = int(np.median(toks))
    rep["p90_completion_tokens"] = int(np.percentile(toks, 90))
    rep["forced_share"] = round(float(np.mean([bool(x.get("forced")) for x in list(r.values()) + list(q.values())])), 3)
    print(json.dumps(rep, ensure_ascii=False))
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rep, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
