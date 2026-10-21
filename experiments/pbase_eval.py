"""Does adding top text-only turns by P_base (organisers' baseline prompt) to s11r flags pay off?
Pre-registered rule: added rows need precision >= 0.48 on our labels (~0.31 on the public standard, using the
Claude->public ratio 0.47/0.73 measured on text turns by another team), else F1 does not improve.
    python experiments/pbase_eval.py runs/pbase/pull
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from calib_eval import labels  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/pbase/pull/eval")


def load_p(name):
    f = D / name
    if not f.exists():
        return None
    return {r["id"]: r["p"] for r in map(json.loads, f.read_text().splitlines())}


sets = {
    "t431": (pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet"),
             [ROOT / "runs/calib/out/s11r_t431_1.csv", ROOT / "runs/calib/out/s11r_t431_2.csv"]),
    "apb": (pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet"), [D / "s11r_apb.csv"]),
}
lab = labels()
for name, (df, flag_files) in sets.items():
    df = df.copy()
    df["text"] = ~df.response.str.contains("TOOL_CALL")
    df["y"] = df.id.map(lab) if name == "t431" else df.label.where(df.label >= 0)
    for lay in ("exact", "tail"):
        p = load_p(f"pbase_{lay}_{name}.jsonl")
        if p is None:
            continue
        df["p"] = df.id.map(p)
        t = df[df.text & df.y.notna() & df.p.notna()]
        print(f"\n== {name} {lay}: text rows {len(t)}, base {t.y.mean():.3f}; share p>=0.8 {(t.p >= .8).mean():.2f}")
        for c in (0.8, 0.5):
            s = t[t.p >= c]
            print(f"   P_base>={c}: n={len(s)} precision {s.y.mean():.2f}")
        for ff in flag_files:
            if not ff.exists():
                continue
            fl = dict(pd.read_csv(ff)[["id", "label"]].values)
            u = t[t.id.map(fl) == 0].sort_values("p", ascending=False)
            print(f"   {ff.name}: text unflagged {len(u)}, base {u.y.mean():.3f}")
            for k in (5, 10, 15, 20, 30, 40):
                s = u.head(k)
                print(f"     add top-{k:2d} (p>={s.p.min():.2f}): precision {s.y.mean():.2f}  "
                      f"[{int(s.y.sum())}/{k}] {'PASS' if s.y.mean() >= 0.48 else 'fail'}")
    if "p" in df:
        del df["p"]
