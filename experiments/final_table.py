"""One table of every configuration measured on 19.10: our labelled set (valid 46 + 385 early turns), APB-ru, flag rate,
wall time. Missing runs are skipped.   python experiments/final_table.py [--md out.md]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from calib_eval import labels  # noqa: E402

H4, HY, CAL = ROOT / "runs/h4/pull/eval", ROOT / "runs/hyb/pull/eval", ROOT / "runs/calib/out"
# (name, what, t431 csv, apb csv, server)
RUNS = [
    ("s11r #1", "base (sub11 + 3 rules)", CAL / "s11r_t431_1.csv", None, "calibration 14.10"),
    ("s11r #2", "base, repeat", CAL / "s11r_t431_2.csv", None, "calibration 14.10"),
    ("s11r #3", "base, LoRA-enabled server", HY / "s11r_t431.csv", None, "RTX PRO 10183"),
    ("s11r #4", "base, plain server", HY / "s11r0_t431.csv", H4 / "base_apb.csv", "10183 / 10174"),
    ("s11r #5", "base, repeat APB", None, HY / "s11r_apb2.csv", "RTX PRO 10183"),
    ("empero", "Qwen3.8-35B-A3B distill (BF16)", H4 / "empero_t431.csv", H4 / "empero_apb.csv", "RTX PRO 10174"),
    ("bigbang", "BigBang-v1 (BF16)", H4 / "bigbang_t431.csv", H4 / "bigbang_apb.csv", "RTX PRO 10174"),
    ("occamy", "occamy-1.0 (BF16)", H4 / "occamy_t431.csv", H4 / "occamy_apb.csv", "RTX PRO 10174"),
    ("opus47", "Claude-4.7-Opus distill (BF16)", H4 / "lordx_t431.csv", None, "RTX PRO 10174"),
    ("hybrid", "Opus-4.7 LoRA on FP8 + base on capped rows", HY / "hyb_t431.csv", HY / "hyb_apb.csv", "RTX PRO 10183"),
    ("s11f", "s11r + valid fixes (refusal, transfer, phrasing, evidence)", HY / "s11f_t431.csv", HY / "s11f_apb.csv", "RTX PRO 10183"),
    ("s11f #2", "s11f, repeat", HY / "s11f_t431b.csv", HY / "s11f_apb2.csv", "RTX PRO 10183"),
    ("s11g", "s11f + early-turn fixes (airline cabin, retail, banking)", HY / "s11g_t431.csv", HY / "s11g_apb.csv", "RTX PRO 10183"),
    ("s11h", "s11g + stop at verdict", HY / "s11h_t431.csv", HY / "s11h_apb.csv", "RTX PRO 10183"),
]


def prf(pred: dict, gold: dict, rate: float | None = None):
    ids = [i for i in gold if i in pred]
    tp = sum(1 for i in ids if gold[i] == 1 and pred[i] == 1)
    fp = sum(1 for i in ids if gold[i] == 0 and pred[i] == 1)
    fn = sum(1 for i in ids if gold[i] == 1 and pred[i] == 0)
    tn = len(ids) - tp - fp - fn
    if rate:  # F1 at the test positive rate (APB has 12% positives, the test 23.7%)
        r, fpr = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
        return 2 * rate * r / (2 * rate * r + (1 - rate) * fpr + rate * (1 - r)), tp, fp, fn
    return 2 * tp / max(2 * tp + fp + fn, 1), tp, fp, fn


def wall(csv: Path):
    log = csv.with_suffix(".log")
    if not log.exists():
        return None
    for ln in reversed(log.read_text(errors="ignore").splitlines()):
        if "wrote" in ln and "predictions" in ln:
            return float(ln.split("s]")[0].split()[-1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    lab = labels()
    t = pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")
    valid = {i: int(l) for i, l in zip(t.id, t.label) if l >= 0}
    early = {i: v for i, v in lab.items() if i not in valid}
    apb = pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet")
    apb = {i: int(l) for i, l in zip(apb.id, apb.label) if l >= 0}
    rows = []
    for name, what, tc, ac, srv in RUNS:
        r = {"run": name, "what": what}
        if tc and tc.exists() and wall(tc) is None and tc.parent != CAL:
            r["early385 F1"] = "running"
        elif tc and tc.exists():
            p = dict(pd.read_csv(tc)[["id", "label"]].values)
            f, tp, fp, fn = prf(p, early)
            r["early385 F1"], r["early TP/FP/FN"] = round(f, 3), f"{tp}/{fp}/{fn}"
            f, tp, fp, fn = prf(p, valid)
            r["valid F1"], r["valid TP/FP/FN"] = round(f, 3), f"{tp}/{fp}/{fn}"
            r["flagged"] = f"{sum(p.values()) / len(p):.1%}"
            r["t431 s"] = wall(tc)
        if ac and ac.exists() and wall(ac) is None:
            r["APB F1@23.7%"] = "running"
        elif ac and ac.exists():
            p = dict(pd.read_csv(ac)[["id", "label"]].values)
            f, tp, fp, fn = prf(p, apb, 0.237)
            r["APB F1@23.7%"], r["APB TP/FP/FN"] = round(f, 3), f"{tp}/{fp}/{fn}"
            r["APB s"] = wall(ac)
        if len(r) > 2:
            r["server"] = srv
            rows.append(r)
    df = pd.DataFrame(rows).fillna("")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    if a.md:
        Path(a.md).write_text(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
