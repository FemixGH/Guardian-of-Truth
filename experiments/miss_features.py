"""Among rows candidate 12 leaves unflagged (no hard flag, judge p<0.5): how often does each deterministic signal fire on
true hallucinations vs clean rows?  -> candidates for new high-precision rules.

python experiments/miss_features.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardian import combine  # noqa: E402
from guardian.features import ID_RE, flags  # noqa: E402
from guardian.parse import parse_example  # noqa: E402

TEXT_ID_RE = re.compile(r"(?<![\w@.])#?([A-Z]{1,3}\d{4,}|[A-Z0-9]{6}|\d{6,})(?![\w@])")


def novel_text_ids(ex, prompt):
    t = ex.resp_text or ""
    out = []
    for m in TEXT_ID_RE.finditer(t):
        v = m.group(1)
        if not re.search(r"\d", v) or not re.search(r"[A-Za-z]", v) and len(v) < 6:
            continue
        if v not in prompt and v.lstrip("#") not in prompt:
            out.append(v)
    return out


def load(p):
    return {json.loads(l)["id"]: (json.loads(l).get("p") or 0) >= .5 for l in open(p)}


def main():
    cfg = combine.load(str(ROOT / "model/calibration.json"))
    sets = {"apb": (ROOT / "runs/cc/apb_eval_ru.parquet", ROOT / "runs/rtx/eval/E2_v3g_stop_kb_apb.jsonl"),
            "t431": (ROOT / "runs/timing_431_labeled.parquet", ROOT / "runs/rtx/eval/E2_v3g_stop_kb_t431.jsonl")}
    for s, (pq, js) in sets.items():
        d = pd.read_parquet(pq); j = load(js)
        cnt = Counter(); tot = Counter()
        for r in d.itertuples():
            ex = parse_example(r.prompt, r.response)
            f = flags(ex)[0]
            if combine.final_p(f, 0, cfg) >= 1 or j.get(r.id):
                continue
            key = {1: "pos", 0: "neg"}.get(int(r.label), "other")
            tot[key] += 1
            sig = {"novel_text_nums": f.get("n_novel_text_nums", 0) > 0,
                   "novel_arg_nums": f.get("n_novel_arg_nums", 0) > 0,
                   "action_claim_no_call": f.get("action_claim_no_call", 0) > 0,
                   "novel_text_ids": bool(novel_text_ids(ex, r.prompt)),
                   "write_no_confirm": f.get("write_no_confirm", 0) > 0,
                   "trace_write_unconfirmed": f.get("trace_write_unconfirmed", 0) > 0,
                   "trace_no_status_check": f.get("trace_no_status_check", 0) > 0,
                   "policy_soft": f.get("policy_soft", 0) > 0,
                   "transfer": f.get("n_transfer", 0) > 0}
            for k, v in sig.items():
                if v:
                    cnt[(k, key)] += 1
        print(f"== {s}: unflagged rows {dict(tot)}")
        for k in sorted({k for k, _ in cnt}):
            print(f"   {k:24s} pos {cnt[(k, 'pos')]:3d}  neg {cnt[(k, 'neg')]:3d}  other {cnt[(k, 'other')]:3d}")


if __name__ == "__main__":
    main()
