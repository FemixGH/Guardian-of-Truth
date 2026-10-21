"""Compare judge models (s11r pipeline) on APB-ru and t431 (valid + Claude-labelled early turns).
APB F1 is re-weighted to the test positive rate 23.7%, as in earlier reports.
    python experiments/models_eval.py runs/h4/pull/eval
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from calib_eval import labels  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/h4/pull/eval")
apb = pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet")[["id", "label"]].rename(columns={"label": "y"})
apb = apb[apb.y >= 0]
lab = labels()
t431 = pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")[["id", "label"]]
t431["y"] = t431.id.map(lab)
valid = t431[t431.label >= 0][["id", "label"]].rename(columns={"label": "y"})
t431 = t431[["id", "y"]]


def scores(pred, gold, rate=None):
    d = gold.merge(pred[["id", "label"]], on="id").dropna(subset=["y"])
    tp = ((d.y == 1) & (d.label == 1)).sum(); fn = ((d.y == 1) & (d.label == 0)).sum()
    fp = ((d.y == 0) & (d.label == 1)).sum(); tn = ((d.y == 0) & (d.label == 0)).sum()
    rec = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    if rate:  # F1 at the test positive rate
        tp_, fp_ = rate * rec, (1 - rate) * fpr
        f1 = 2 * tp_ / (2 * tp_ + fp_ + rate * (1 - rec))
    else:
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, rec, fpr, len(d)


rows = []
names = sorted({f.stem.rsplit("_", 1)[0] for f in [*D.glob("*_apb.csv"), *D.glob("*_t431.csv")]} | {"base"})
for n in names:
    r = {"model": n}
    f = D / f"{n}_apb.csv"
    if f.exists():
        r["apb_f1"], r["apb_R"], r["apb_FPR"], r["apb_n"] = scores(pd.read_csv(f), apb, 0.237)
    t = [D / f"{n}_t431.csv"] if n != "base" else [ROOT / f"runs/calib/out/s11r_t431_{i}.csv" for i in (1, 2)]
    for i, f in enumerate(x for x in t if x.exists()):
        p = pd.read_csv(f)
        r[f"t431_f1_{i}"] = scores(p, t431)[0]
        v = scores(p, valid)
        r[f"valid_TPR_FPR_{i}"] = f"{v[1]:.2f}/{v[2]:.2f}"
        r[f"flag_rate_{i}"] = p.label.mean()
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
