"""Voting over prompt variants (v3g x2 runs, v3h, v3i, v3j) + new hard rules: APB-ru, valid, early, blind audit.

python experiments/vote_eval.py
"""
import itertools
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
RUNS = {"g1": "runs/rtx/eval/E2_v3g_stop_kb", "g2": "runs/rtx2/eval/G_v3g", "h": "runs/rtx2/eval/H_v3h",
        "i": "runs/rtx3/eval/I_v3i", "j": "runs/rtx3/eval/J_v3j"}


def load(p):
    return {json.loads(l)["id"]: (json.loads(l).get("p") or 0) >= .5 for l in open(p)}


def main():
    cfg = combine.load(str(ROOT / "experiments/cal_cand12.json"))
    apb = pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet"); t = pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")
    hard = {r.id: combine.final_p(flags(parse_example(r.prompt, r.response))[0], 0, cfg) >= 1
            for df in (apb, t) for r in df.itertuples()}
    J = {k: (load(ROOT / f"{v}_apb.jsonl"), load(ROOT / f"{v}_t431.jsonl")) for k, v in RUNS.items()}
    key = json.load(open(ROOT / "runs/audit/audit_key.json"))
    lab = {key[x["row"]]["id"]: (x["label"], key[x["row"]]["grp"]) for i in range(6)
           for x in json.load(open(ROOT / f"runs/audit/audit_labels_{i}.json"))}
    aud_pos = [i for i, x in lab.items() if x[0] == 1]; aud_neg = [i for i, x in lab.items() if x[0] == 0]

    def score(members, k):
        fa = {i: hard[i] or sum(J[m][0].get(i, False) for m in members) >= k for i in apb.id}
        ft = {i: hard[i] or sum(J[m][1].get(i, False) for m in members) >= k for i in t.id}
        pos = apb.id[apb.label == 1]; neg = apb.id[apb.label == 0]
        R = sum(fa[i] for i in pos) / len(pos); F = sum(fa[i] for i in neg) / len(neg); P = R * B / (R * B + F * (1 - B))
        v = t[t.label >= 0]; e = t[t.label == -1]
        tp = sum(ft[i] for i in v.id[v.label == 1]); fp = sum(ft[i] for i in v.id[v.label == 0])
        atp = sum(ft[i] for i in aud_pos); afp = sum(ft[i] for i in aud_neg)
        ap = atp / max(1, atp + afp); ar = atp / len(aud_pos)
        return (2 * P * R / (P + R), R, F, tp, fp, sum(ft[i] for i in e.id) / len(e), atp, afp, 2 * ap * ar / max(1e-9, ap + ar))

    rows = []
    for n in (1, 2, 3, 4, 5):
        for members in itertools.combinations(RUNS, n):
            for k in range(1, n + 1):
                s = score(members, k)
                rows.append(("+".join(members), f"{k}/{n}") + s)
    df = pd.DataFrame(rows, columns=["members", "vote", "apbF1", "apbR", "apbFPR", "vTP", "vFP", "early", "audTP", "audFP", "audF1"])
    pd.set_option("display.width", 200)
    print(df[df.vote.isin(["1/1"])].round(3).to_string(index=False))
    print("\nTOP by APB F1 (multi-member):")
    print(df[~df.vote.isin(["1/1"])].sort_values("apbF1", ascending=False).head(15).round(3).to_string(index=False))
    print("\nTOP by audit F1 (all 145 audited early turns, Claude labels):")
    print(df.sort_values("audF1", ascending=False).head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
