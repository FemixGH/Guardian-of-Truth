"""Combine per-pass judge outputs into pipeline variants (hard rules OR pass 1 OR pass 2) and score them.

python experiments/pipeline_eval.py --dir runs/rtx/eval
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardian import combine  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian.parse import parse_example  # noqa: E402

B = 0.237


def load(path):
    out = {}
    if Path(path).exists():
        for line in open(path):
            r = json.loads(line)
            out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs/rtx/eval"))
    a = ap.parse_args()
    D = Path(a.dir)
    cfg = combine.load(str(ROOT / "model/calibration.json"))
    cfg_old = dict(cfg, hard_flags=[h for h in cfg["hard_flags"] if h != "rules2_hard"])
    sets = {"apb": pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet"),
            "t431": pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")}
    hard = {}
    for s, df in sets.items():
        for r in df.itertuples():
            f = flags(parse_example(r.prompt, r.response))[0]
            hard[r.id] = (combine.final_p(f, 0.0, cfg_old) >= 1, combine.final_p(f, 0.0, cfg) >= 1)
    variants = {
        "s11-like: old hard | v3f | v2": ("old", "E0_v3f", "E4_v2_stop"),
        "old hard | v3f (no 2nd pass)": ("old", "E0_v3f", None),
        "old hard | v3f+stop": ("old", "E1_v3f_stop", None),
        "old hard | v3f+stop | v2+stop": ("old", "E1_v3f_stop", "E4_v2_stop"),
        "NEW hard | v3f+stop | v2+stop": ("new", "E1_v3f_stop", "E4_v2_stop"),
        "NEW hard | v3g+stop+kb": ("new", "E2_v3g_stop_kb", None),
        "NEW hard | v3g+stop+kb | recall+stop": ("new", "E2_v3g_stop_kb", "E3_recall_stop"),
    }
    for name, (h, p1, p2) in variants.items():
        rep = []
        secs = 0
        for s, df in sets.items():
            j1 = load(D / f"{p1}_{s}.jsonl")
            j2 = load(D / f"{p2}_{s}.jsonl") if p2 else {}
            if not j1:
                continue
            secs += sum(x.get("sec") or 0 for x in list(j1.values()) + list(j2.values()))
            fl = {}
            for r in df.itertuples():
                hd = hard[r.id][1 if h == "new" else 0]
                v1 = (j1.get(r.id, {}).get("p") or 0) >= 0.5
                v2 = (j2.get(r.id, {}).get("p") or 0) >= 0.5
                fl[r.id] = hd or v1 or v2
            lab = dict(zip(df.id, df.label))
            if s == "apb":
                pos = [i for i in fl if lab[i] == 1]; neg = [i for i in fl if lab[i] == 0]
                R = sum(fl[i] for i in pos) / len(pos); F = sum(fl[i] for i in neg) / len(neg)
                P = R * B / (R * B + F * (1 - B))
                rep.append(f"APB R {R:.3f} FPR {F:.3f} F1 {2 * P * R / (P + R):.3f}")
            else:
                tp = sum(fl[i] for i in fl if lab[i] == 1); fp = sum(fl[i] for i in fl if lab[i] == 0)
                fn = sum(not fl[i] for i in fl if lab[i] == 1)
                early = [fl[i] for i in fl if lab[i] == -1]
                rep.append(f"valid {tp}/{fp}/{fn} early {sum(early) / len(early):.3f}")
        if rep:
            print(f"{name:40s} | " + " | ".join(rep) + f" | judge GPU-sec sum {secs:,.0f}")


if __name__ == "__main__":
    main()
