"""Offline proxy vs real test F1 for all past submissions (runs/calib/out/<sub>_<set>_<run>.csv).

Proxy sets:
  T431 : valid (organisers' labels) + 385 early turns (blind Claude labels: audit_labels_*.json + labels2_*.json)
  APB  : APB-ru (human labels), F1 rescaled to the test positive rate 23.7%
Prints per-submission metrics, correlation with the real test F1 and a leave-one-out prediction error.

python experiments/calib_eval.py [--dir runs/calib/out]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST = {"sub1": 0.5075, "sub4": 0.5116, "sub5": 0.5985, "sub7": 0.5923, "sub8": 0.6023, "sub11": 0.6124, "cand12": 0.576}
B = 0.237


def labels():
    t = pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")
    lab = {r.id: int(r.label) for r in t.itertuples() if r.label >= 0}
    k1 = json.load(open(ROOT / "runs/audit/audit_key.json"))
    for f in glob.glob(str(ROOT / "runs/audit/audit_labels_*.json")):
        for x in json.load(open(f)):
            lab[k1[x["row"]]["id"]] = int(x["label"])
    k2p = ROOT / "runs/audit/audit_key2.json"
    if k2p.exists():
        k2 = json.load(open(k2p))
        for x in json.load(open(ROOT / "runs/audit/labels2_merged.json")):
            lab[k2[x["row"]]] = int(x["label"])
    return lab


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs/calib/out"))
    a = ap.parse_args()
    lab = labels()
    apb = pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet"); al = dict(zip(apb.id, apb.label))
    rows = []
    for f in sorted(glob.glob(f"{a.dir}/*.csv")):
        name = Path(f).stem; sub, st, run = name.rsplit("_", 2)
        p = pd.read_csv(f); pr = dict(zip(p.id, p.label))
        if st == "t431":
            ids = [i for i in pr if i in lab]
            tp = sum(pr[i] == 1 and lab[i] == 1 for i in ids); fp = sum(pr[i] == 1 and lab[i] == 0 for i in ids)
            fn = sum(pr[i] == 0 and lab[i] == 1 for i in ids)
            v = [i for i in ids if "::t" in i]
            vtp = sum(pr[i] == 1 and lab[i] == 1 for i in v); vfp = sum(pr[i] == 1 and lab[i] == 0 for i in v)
            rows.append(dict(sub=sub, set=st, run=run, n=len(ids), F1=f1(tp, fp, fn), P=tp / max(1, tp + fp), R=tp / max(1, tp + fn),
                             flag=np.mean([pr[i] for i in pr]), valid=f"{vtp}/{vfp}"))
        else:
            pos = [i for i in pr if al.get(i) == 1]; neg = [i for i in pr if al.get(i) == 0]
            R = np.mean([pr[i] for i in pos]); F = np.mean([pr[i] for i in neg]); P = R * B / (R * B + F * (1 - B))
            rows.append(dict(sub=sub, set=st, run=run, n=len(pr), F1=2 * P * R / (P + R), P=P, R=R, flag=np.mean(list(pr.values())), valid=""))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    for st in ("t431", "apb"):
        m = d[d.set == st].groupby("sub")[["F1", "flag"]].mean()
        m["test"] = [TEST.get(s, np.nan) for s in m.index]
        k = m.dropna()
        if len(k) >= 3:
            r = np.corrcoef(k.F1, k.test)[0, 1]
            rk = pd.Series(k.F1).rank().corr(pd.Series(k.test).rank())
            # leave-one-out linear prediction of the test F1
            err = []
            for s in k.index:
                o = k.drop(s); b = np.polyfit(o.F1, o.test, 1); err.append(np.polyval(b, k.F1[s]) - k.test[s])
            print(f"\n[{st}] Pearson {r:.2f}  Spearman {rk:.2f}  LOO abs err mean {np.mean(np.abs(err)):.4f} max {np.max(np.abs(err)):.4f}")
        b = np.polyfit(k.F1, k.test, 1) if len(k) >= 2 else None
        if b is not None:
            m["pred_test"] = np.polyval(b, m.F1)
        print(m.round(4).to_string())


if __name__ == "__main__":
    main()
