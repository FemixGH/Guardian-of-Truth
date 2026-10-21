"""After the base-model rollouts on the whole pool: decide which rows are trustworthy and useful for training.

- orig (noisy label 0): keep only if the base model votes 0 in >= --orig-min-zero of the samples (a turn the
  base model itself mostly considers clean); otherwise the "clean" label is unreliable -> dropped.
- every row: informative = samples disagree (some right, some wrong) -> GSPO signal.
- all-wrong reliable positives (apb_pos, syn_*): candidates for STaR rationalisation (hinted rollouts) for SFT.
- class balance of the GSPO set: positives capped to --max-pos-share.

Writes ids_sft_rows.json (rows with >=1 correct sample), ids_gspo.json, ids_star.json and prints a summary.

python experiments/lora/select_rows.py --pool pool.parquet --summary rollouts_base_summary.csv --outdir sel/
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--orig-min-zero", type=float, default=0.75)
    ap.add_argument("--max-pos-share", type=float, default=0.40)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(a.pool)[["id", "label", "kind", "domain"]]
    s = pd.read_csv(a.summary).merge(pool[["id", "kind", "domain"]], on="id")
    s["n"] = s.votes0 + s.votes1 + s.none
    s["correct"] = s.apply(lambda r: r.votes1 if r.label == 1 else r.votes0, axis=1)
    s["acc"] = s.correct / s.n
    # unverified label-0 rows (original τ² turns and their paraphrases): kept only if the base model mostly agrees
    drop_orig = s.kind.isin(["orig", "para"]) & (s.votes0 / s.n < a.orig_min_zero)
    print(f"orig/para rows dropped as unreliable (base votes 0 in < {a.orig_min_zero:.0%}): {int(drop_orig.sum())} of "
          f"{int(s.kind.isin(['orig', 'para']).sum())}")
    s = s[~drop_orig]
    print("\nbase accuracy by kind:\n" + s.groupby("kind").agg(rows=("id", "size"), acc=("acc", "mean"),
                                                                 all_right=("acc", lambda x: (x == 1).mean()),
                                                                 all_wrong=("acc", lambda x: (x == 0).mean())).round(3).to_string())
    sft = s[s.correct > 0].id.tolist()
    inf = s[(s.correct > 0) & (s.correct < s.n)]
    pos, neg = inf[inf.label == 1], inf[inf.label == 0]
    cap = int(a.max_pos_share / (1 - a.max_pos_share) * len(neg))
    if len(pos) > cap:
        pos = pos.sample(cap, random_state=1)
    gspo = pd.concat([pos, neg]).sample(frac=1, random_state=1).id.tolist()
    # STaR rationalisation only for MANUALLY CONFIRMED positives the base model never finds: a synthetic positive the
    # model never flags may be a failed injection (no real error), rationalising it would teach false alarms
    star = s[(s.correct == 0) & (s.kind == "apb_pos")].id.tolist()
    sft = sorted(set(sft) | set(star))  # STaR rows have no correct base sample: their hinted samples are added
    json.dump(sft, open(out / "ids_sft_rows.json", "w"))
    json.dump(gspo, open(out / "ids_gspo.json", "w"))
    json.dump(star, open(out / "ids_star.json", "w"))
    print(f"\nSFT rows {len(sft)} | GSPO informative rows {len(gspo)} (pos {len(pos)}, neg {len(neg)}) | STaR rows {len(star)}")


if __name__ == "__main__":
    main()
