"""v3h vs v3g (same server) + earlier v3g run: APB-ru, valid, early-turn flag rate and the blind Claude audit.

python experiments/v3h_eval.py --dir runs/rtx2/eval
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


def load(p):
    if not Path(p).exists():
        return {}
    return {json.loads(l)["id"]: (json.loads(l).get("p") or 0) >= .5 for l in open(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs/rtx2/eval"))
    a = ap.parse_args()
    D = Path(a.dir)
    cfg = combine.load(str(ROOT / "experiments/cal_cand12.json"))
    apb = pd.read_parquet(ROOT / "runs/cc/apb_eval_ru.parquet"); t = pd.read_parquet(ROOT / "runs/timing_431_labeled.parquet")
    hard = {r.id: combine.final_p(flags(parse_example(r.prompt, r.response))[0], 0, cfg) >= 1
            for df in (apb, t) for r in df.itertuples()}
    key = json.load(open(ROOT / "runs/audit/audit_key.json"))
    lab = {key[x["row"]]["id"]: (x["label"], key[x["row"]]["grp"], x["confidence"])
           for i in range(6) for x in json.load(open(ROOT / f"runs/audit/audit_labels_{i}.json"))}
    runs = {"v3g (earlier server, E2)": (ROOT / "runs/rtx/eval/E2_v3g_stop_kb_apb.jsonl", ROOT / "runs/rtx/eval/E2_v3g_stop_kb_t431.jsonl"),
            "v3g (server 2)": (D / "G_v3g_apb.jsonl", D / "G_v3g_t431.jsonl"),
            "v3h (server 2)": (D / "H_v3h_apb.jsonl", D / "H_v3h_t431.jsonl"),
            "v3i claims table": (ROOT / "runs/rtx3/eval/I_v3i_apb.jsonl", ROOT / "runs/rtx3/eval/I_v3i_t431.jsonl"),
            "v3j expected turn": (ROOT / "runs/rtx3/eval/J_v3j_apb.jsonl", ROOT / "runs/rtx3/eval/J_v3j_t431.jsonl")}
    for name, (pa, pt) in runs.items():
        ja, jt = load(pa), load(pt)
        out = [f"{name:26s}"]
        if ja:
            fl = {i: hard[i] or ja.get(i, False) for i in apb.id}
            pos = apb.id[apb.label == 1]; neg = apb.id[apb.label == 0]
            R = sum(fl[i] for i in pos) / len(pos); F = sum(fl[i] for i in neg) / len(neg); P = R * B / (R * B + F * (1 - B))
            out.append(f"APB R {R:.3f} FPR {F:.3f} F1 {2 * P * R / (P + R):.3f}")
        if jt:
            fl = {i: hard[i] or jt.get(i, False) for i in t.id}
            v = t[t.label >= 0]; e = t[t.label == -1]
            tp = sum(fl[i] for i in v.id[v.label == 1]); fp = sum(fl[i] for i in v.id[v.label == 0])
            out.append(f"valid {tp}/{fp}/{23 - tp} early {sum(fl[i] for i in e.id) / len(e):.3f}")
            # audit: agent label vs flag, cand12-unflagged groups only
            un = {i: x for i, x in lab.items() if x[1] != "flagged_ctrl"}
            m_pos = [i for i, x in un.items() if x[0] == 1]; m_neg = [i for i, x in un.items() if x[0] == 0]
            ctrl = [i for i, x in lab.items() if x[1] == "flagged_ctrl"]
            out.append(f"audit misses caught {sum(fl[i] for i in m_pos)}/{len(m_pos)}, audit-clean flagged {sum(fl[i] for i in m_neg)}/{len(m_neg)}, "
                       f"ctrl kept {sum(fl[i] for i in ctrl)}/{len(ctrl)}")
        print(" | ".join(out))


if __name__ == "__main__":
    main()
