"""Final 19.10 summary: candidates with run means, plus the exact effect of the payment_schema hard rule (deterministic,
added after the runs). Writes runs/final_table_1019.md.   python experiments/final_summary.py"""
import statistics as st
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "dist/s11h/src"))
from calib_eval import labels  # noqa: E402
from guardian.parse import parse_example  # noqa: E402
from guardian.rules2 import payment_schema  # noqa: E402

H4, HY, CAL = ROOT / "runs/h4/pull/eval", ROOT / "runs/hyb/pull/eval", ROOT / "runs/calib/out"
C = {"s11r (base)": ([CAL / "s11r_t431_1.csv", CAL / "s11r_t431_2.csv", HY / "s11r_t431.csv", HY / "s11r0_t431.csv"],
                     [H4 / "base_apb.csv", HY / "s11r_apb2.csv"], [582, 599, 628, 638], [1012, 1153]),
     "s11f": ([HY / "s11f_t431.csv", HY / "s11f_t431b.csv"], [HY / "s11f_apb.csv", HY / "s11f_apb2.csv"], [639, 651], [1153, 1157]),
     "s11g": ([HY / "s11g_t431.csv"], [HY / "s11g_apb.csv"], [657], [1197]),
     "s11h": ([HY / "s11h_t431.csv"], [HY / "s11h_apb.csv"], [444], [726]),
     "hybrid": ([HY / "hyb_t431.csv"], [HY / "hyb_apb.csv"], [653], [1107])}


def sc(p, g, rate=None):
    ids = [i for i in g if i in p]
    tp = sum(1 for i in ids if g[i] == 1 and p[i] == 1)
    fp = sum(1 for i in ids if g[i] == 0 and p[i] == 1)
    fn = sum(1 for i in ids if g[i] == 1 and p[i] == 0)
    tn = len(ids) - tp - fp - fn
    if rate:
        r, fpr = tp / (tp + fn), fp / (fp + tn)
        return 2 * rate * r / (2 * rate * r + (1 - rate) * fpr + rate * (1 - r)), tp, fp, fn
    return 2 * tp / (2 * tp + fp + fn), tp, fp, fn


def main():
    lab = labels()
    t = pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")
    valid = {i: int(l) for i, l in zip(t.id, t.label) if l >= 0}
    early = {i: v for i, v in lab.items() if i not in valid}
    a = pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet")
    apb = {i: int(l) for i, l in zip(a.id, a.label) if l >= 0}
    ps = {r.id for df in (t, a) for r in df.itertuples() if payment_schema(parse_example(r.prompt, r.response))}
    rows = []
    for n, (tf, af, tt, ta) in C.items():
        for adj in ([False, True] if n != "hybrid" else [False]):
            def load(f):
                p = dict(pd.read_csv(f)[["id", "label"]].values)
                return {i: (1 if adj and i in ps else v) for i, v in p.items()}
            T, A = [load(f) for f in tf], [load(f) for f in af]
            e, v, ap = [sc(p, early) for p in T], [sc(p, valid) for p in T], [sc(p, apb, 0.237) for p in A]
            rows.append({
                "candidate": n + (" + payment_schema" if adj else ""),
                "runs 431/APB": f"{len(T)}/{len(A)}",
                "early385 F1 mean [min-max]": f"{st.mean(x[0] for x in e):.3f} [{min(x[0] for x in e):.3f}-{max(x[0] for x in e):.3f}]",
                "valid caught/false/missed per run": ", ".join(f"{x[1]}/{x[2]}/{x[3]}" for x in v),
                "APB F1 mean [runs]": f"{st.mean(x[0] for x in ap):.3f} [{', '.join(f'{x[0]:.3f}' for x in ap)}]",
                "APB caught/false/missed (mean)": f"{st.mean(x[1] for x in ap):.0f}/{st.mean(x[2] for x in ap):.0f}/{st.mean(x[3] for x in ap):.0f}",
                "time 431 / APB, s (RTX PRO)": f"{st.mean(tt):.0f} / {st.mean(ta):.0f}"})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(df.to_string(index=False))
    (ROOT / "runs/final_table_1019.md").write_text(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
