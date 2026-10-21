"""Validate the rules2 checks on every labelled set we have: per rule, fires on hallucinations / clean turns / APB
process errors / unlabelled early turns; prints the clean-turn fires for manual inspection.

python experiments/rules2_eval.py [--show 6]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guardian import rules2  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian import combine  # noqa: E402
from guardian.parse import parse_example  # noqa: E402

SETS = {
    "valid": ROOT.parent / "valid.parquet",
    "apb_en": ROOT / "runs/apb_tau2_labeled.parquet",
    "apb_ru": ROOT / "runs/cc/apb_eval_ru.parquet",
    "early": ROOT / "runs/timing_431_labeled.parquet",
    "apb_train_ru": ROOT / "runs/lora_h200/data/apb_train_ru.parquet",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    cfg = combine.load(str(ROOT / "model/calibration.json"))
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # rule -> set/label -> [fires, already_hard]
    examples = defaultdict(list)
    for sname, path in SETS.items():
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if sname == "early":
            df = df[df.label == -1]
        for r in df.itertuples(index=False):
            ex = parse_example(r.prompt, r.response)
            res = rules2.check(ex)
            hard = combine.final_p(flags(ex)[0], 0.0, cfg) >= 1.0
            lab = int(r.label)
            key = f"{sname}:{'pos' if lab == 1 else 'neg' if lab == 0 else 'B' if lab == -1 and sname.startswith('apb') else 'unl' if sname == 'early' else 'other'}"
            for name in rules2.RULES:
                st = stats[name][key]
                if name in res:
                    st[0] += 1
                    st[1] += int(hard)
                    if lab != 1:
                        examples[name].append((r.id, key, res[name][0][:220], (r.response or "")[:200].replace("\n", " ")))
                st[1] += 0
            stats["_rows"][key][0] += 1
    keys = sorted(stats["_rows"])
    print("rows per set:", {k: stats["_rows"][k][0] for k in keys})
    for name in rules2.RULES:
        cells = [f"{k}={stats[name][k][0]}(hard already {stats[name][k][1]})" for k in keys if stats[name][k][0]]
        print(f"\n== {name}: " + ", ".join(cells))
        for e in examples[name][:a.show]:
            print("   NON-POS FIRE", e[0], e[1], "|", e[2], "| RESP:", e[3])


if __name__ == "__main__":
    main()
