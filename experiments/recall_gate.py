"""Gate for recall passes: precision of the flags a pass ADDED (rows flagged only by that pass: no hard flag, no other
pass >= 0.5), on the 385 early turns (Claude labels), valid (organisers) and APB-ru (human).
Pre-registered: precision >= 0.5 on early turns AND <= +5 false alarms on APB.
    python experiments/recall_gate.py <dir with {name}_{t431,apb}.{csv,jsonl}> <name> <pass index or name> [baseline pass]
e.g. textmiss specialist of s11r2 = pass 3 (0 v3f, 1 v2, 2 numbers, 3 textmiss); DBYJ of s11j = pass 1.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from calib_eval import labels  # noqa: E402


def added(d: Path, name: str, st: str, k: int):
    votes = defaultdict(dict)
    for r in map(json.loads, open(d / f"{name}_{st}.jsonl")):
        if r.get("p") is not None:
            votes[r["id"]][r["pass"]] = r["p"]
    fin = dict(pd.read_csv(d / f"{name}_{st}.csv")[["id", "label"]].values)
    out = []
    for i, v in votes.items():
        if fin.get(i) == 1 and v.get(k, 0) >= 0.5 and not any(p >= 0.5 for kk, p in v.items() if kk != k):
            out.append(i)
    ran = sum(1 for v in votes.values() if k in v)
    return out, ran


def main():
    d, name, k = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    lab = labels()
    t = pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")
    valid = {i: int(l) for i, l in zip(t.id, t.label) if l >= 0}
    early = {i: v for i, v in lab.items() if i not in valid}
    a = pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet")
    apb = {i: int(l) for i, l in zip(a.id, a.label) if l >= 0}
    res = {}
    for st, golds in [("t431", [("early385", early), ("valid", valid)]), ("apb", [("APB", apb)])]:
        if not (d / f"{name}_{st}.csv").exists():
            continue
        ids, ran = added(d, name, st, k)
        for gname, g in golds:
            x = [i for i in ids if i in g]
            tp = sum(g[i] for i in x)
            res[gname] = (tp, len(x) - tp)
            print(f"{name} pass {k} on {gname}: ran on {ran} rows, added {len(x)} flags: {tp} real / {len(x) - tp} false"
                  f" -> precision {tp / max(len(x), 1):.2f}")
    e, ap = res.get("early385"), res.get("APB")
    if e and ap:
        prec = e[0] / max(sum(e), 1)
        ok = prec >= 0.5 and ap[1] <= 5
        print(f"GATE: early precision {prec:.2f} (need >=0.5), APB added false alarms {ap[1]} (need <=5) -> "
              f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
