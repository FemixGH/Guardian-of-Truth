"""Compare pass combinations on valid (labelled) + history turns (unlabelled FP-cost proxy).

python experiments/combos.py name1=path1.jsonl name2=path2.jsonl ...
A predict.py dump (with a "pass" field) can be split: name=path.jsonl@0 takes only pass 0.
"""
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian import combine  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian.parse import parse_example  # noqa: E402


def load(spec):
    path, _, ps = spec.partition("@")
    out = {}
    for line in open(path):
        r = json.loads(line)
        if ps and str(r.get("pass")) != ps:
            continue
        p, v = r.get("p"), r.get("verdict")
        if v is not None and (p is None or int(p >= 0.5) != v):
            p = 0.97 if v == 1 else 0.03
        if p is not None:
            out[r["id"]] = p
    return out


def main(args):
    runs = {}
    for a in args:
        name, _, spec = a.partition("=")
        runs[name] = load(spec)
    v = pd.read_parquet(ROOT.parent / "valid.parquet").set_index("id")
    h = pd.read_parquet(ROOT / "experiments" / "history_turns.parquet").set_index("id")
    cfg = combine.load(ROOT / "model" / "calibration.json")
    fv = {i: flags(parse_example(v.prompt[i], v.response[i]))[0] for i in v.index}
    fh = {i: flags(parse_example(h.prompt[i], h.response[i]))[0] for i in h.index}
    names = list(runs)
    print(f"{'combo':34s} {'tp':>3s} {'fp':>3s} {'fn':>3s} {'F1':>6s} {'R':>5s} {'hist%':>6s} {'cov':>5s}")
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            tp = fp = fn = 0
            for i in v.index:
                vs = [runs[c][i] for c in combo if i in runs[c]]
                y = combine.final_label(fv[i], combine.aggregate(vs, cfg) if vs else None, cfg)
                tp += y and v.label[i]
                fp += y and not v.label[i]
                fn += (not y) and v.label[i]
            hp = 0
            cov = 0
            for i in h.index:
                vs = [runs[c][i] for c in combo if i in runs[c]]
                cov += bool(vs)
                hp += combine.final_label(fh[i], combine.aggregate(vs, cfg) if vs else None, cfg)
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            print(f"{'+'.join(combo):34s} {tp:3d} {fp:3d} {fn:3d} {f1:6.3f} {tp / 23:5.2f} {hp / len(h):6.1%} {cov:5d}")


if __name__ == "__main__":
    main(sys.argv[1:])
