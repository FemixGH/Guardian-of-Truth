"""Assemble the training pool (Russian, test-like format) from the translated sources.

kinds:
  apb_pos   - AgentProcessBench first errors, manually confirmed hallucinations (label 1, reliable)
  apb_neg   - AgentProcessBench clean steps before the first error (label 0, ~5% noise)
  syn_<t>   - τ² turns with ONE injected hallucination of type t (label 1, reliable by construction)
  para      - meaning-preserving paraphrase of a τ² turn (label 0, inherits the noise of the original turn)
  orig      - original τ² turn from a successful simulation (label 0, NOISY: kept later only if the base model
              itself mostly says 0, see select.py)

python experiments/lora/build_pool.py --apb apb_train_ru.parquet --seeds tau2_seeds_ru.parquet \
    --synth tau2_synth_ru.parquet --out pool.parquet
"""
import argparse
import re

import pandas as pd

NUM = re.compile(r"\d+(?:[.,]\d+)?")
CALL = re.compile(r"^\t→ TOOL_CALL .*$", re.M)


def _nums(t: str) -> list[str]:
    body = t.split("\n", 1)[1] if "\n" in t else ""  # skip the "⟦ASSISTANT · ход N⟧" header
    return sorted(x.replace(",", ".") for x in NUM.findall(body))


def para_ok(orig: str, new: str) -> bool:
    """A paraphrase is a valid label-0 example only if nothing factual changed: identical tool calls, no tool results
    inside the turn, the same numbers/ids, comparable length."""
    if "← TOOL_RESPONSE" in new or CALL.findall(orig) != CALL.findall(new) or _nums(orig) != _nums(new):
        return False
    lo, ln = len(orig), len(new)
    return 0.6 * lo <= ln <= 1.7 * lo


def syn_ok(orig: str, new: str) -> bool:
    return "← TOOL_RESPONSE" not in new and new.strip() != orig.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apb", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--apb-neg", type=int, default=450, help="max APB clean rows")
    ap.add_argument("--orig", type=int, default=600, help="max original τ² turns")
    ap.add_argument("--max-type-share", type=float, default=0.25, help="cap of one error type among synthetic positives")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = ["id", "prompt", "response", "label", "kind", "domain"]
    apb = pd.read_parquet(a.apb)
    apb["kind"] = apb.label.map({1: "apb_pos", 0: "apb_neg"})
    apb = pd.concat([apb[apb.label == 1], apb[apb.label == 0].sample(min(a.apb_neg, (apb.label == 0).sum()), random_state=1)])
    seeds = pd.read_parquet(a.seeds)
    seeds["kind"] = "orig"
    seeds = seeds.sample(min(a.orig, len(seeds)), random_state=1)
    syn = pd.read_parquet(a.synth)
    syn["kind"] = ["para" if m == "neg_paraphrase" else "syn_" + m for m in syn.mtype]
    orig_all = dict(zip(pd.read_parquet(a.seeds).src, pd.read_parquet(a.seeds).response))
    keep = [(para_ok if k == "para" else syn_ok)(orig_all.get(src, ""), resp)
            for k, src, resp in zip(syn.kind, syn.src, syn.response)]
    print("synthetic rows rejected by the consistency checks:",
          syn[[not x for x in keep]].kind.value_counts().to_dict())
    syn = syn[keep]
    pos = syn[syn.label == 1]
    cap = int(a.max_type_share * len(pos))
    pos = pd.concat([g.sample(min(len(g), max(cap, 1)), random_state=1) for _, g in pos.groupby("kind")]) if len(pos) else pos
    syn = pd.concat([pos, syn[syn.label == 0]])
    print("synthetic positives per type after the cap:", pos.kind.value_counts().to_dict())
    if "domain" not in syn:
        syn["domain"] = syn.id.str.extract(r"^t2_([a-z]+)_")[0]
    pool = pd.concat([apb[cols], seeds[cols], syn[cols]], ignore_index=True).drop_duplicates("id")
    pool.to_parquet(a.out, index=False)
    print(len(pool), "rows; positives", int(pool.label.sum()), f"({pool.label.mean():.2f})")
    print(pool.groupby(["kind", "label"]).size().to_string())
    print(pool.groupby("domain").label.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
