"""Score OR-combinations of judge passes (+ new hard rules) on APB-ru and t431: lever experiments on top of candidate 12.

python experiments/lever_eval.py --dir runs/rtx/eval
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
    sets = {"apb": pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet"),
            "t431": pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")}
    hard = {r.id: combine.final_p(flags(parse_example(r.prompt, r.response))[0], 0.0, cfg) >= 1
            for df in sets.values() for r in df.itertuples()}
    # each variant: list of pass names OR-ed (file <name>_<set>.jsonl); "L2" replaces E2 on long rows by the 160k run
    variants = {
        "cand12 (E2)": ["E2_v3g_stop_kb"],
        "+ L1 T0.7 sample": ["E2_v3g_stop_kb", "L1_t07"],
        "+ L3 args": ["E2_v3g_stop_kb", "L3_args"],
        "+ L3 escalation": ["E2_v3g_stop_kb", "L3_esc"],
        "+ L3 args + esc": ["E2_v3g_stop_kb", "L3_args", "L3_esc"],
        "+ L1 + L3 args": ["E2_v3g_stop_kb", "L1_t07", "L3_args"],
        "+ L1 + L3 args + esc": ["E2_v3g_stop_kb", "L1_t07", "L3_args", "L3_esc"],
        "L2: long rows at 160k": ["E2_v3g_stop_kb", "@L2_long"],
    }
    for name, passes in variants.items():
        rep = []
        secs = 0
        for s, df in sets.items():
            js = []
            for p in passes:
                if p.startswith("@"):
                    continue
                j = load(D / f"{p}_{s}.jsonl")
                if not j:
                    break
                js.append(j)
            else:
                rep_s = None
                long_new = load(D / f"{passes[-1][1:]}_{s}.jsonl") if passes[-1].startswith("@") else {}
                if passes[-1].startswith("@") and not long_new:
                    continue
                secs += sum(x.get("sec") or 0 for j in js for x in j.values())
                fl = {}
                for r in df.itertuples():
                    v = [(j.get(r.id, {}).get("p") or 0) >= 0.5 for j in js]
                    if r.id in long_new:
                        v[0] = (long_new[r.id].get("p") or 0) >= 0.5
                    fl[r.id] = hard[r.id] or any(v)
                lab = dict(zip(df.id, df.label))
                if s == "apb":
                    pos = [i for i in fl if lab[i] == 1]; neg = [i for i in fl if lab[i] == 0]
                    R = sum(fl[i] for i in pos) / len(pos); F = sum(fl[i] for i in neg) / len(neg)
                    P = R * B / (R * B + F * (1 - B))
                    rep_s = f"APB R {R:.3f} FPR {F:.3f} F1 {2 * P * R / (P + R):.3f}"
                else:
                    tp = sum(fl[i] for i in fl if lab[i] == 1); fp = sum(fl[i] for i in fl if lab[i] == 0)
                    fn = sum(not fl[i] for i in fl if lab[i] == 1)
                    early = [fl[i] for i in fl if lab[i] == -1]
                    rep_s = f"valid {tp}/{fp}/{fn} early {sum(early) / len(early):.3f}"
                rep.append(rep_s)
        if rep:
            print(f"{name:28s} | " + " | ".join(rep) + f" | judge GPU-sec {secs:,.0f}")


if __name__ == "__main__":
    main()
