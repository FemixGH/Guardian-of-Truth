"""Evaluate pipeline variants on the re-labelled AgentProcessBench τ² set (runs/apb_eval.parquet).

Components (jsonl from experiments/run_judge.py): Qwen v3f, Qwen v2, numbers specialist, Gemma v3f; hard flags are
computed here. A variant = OR of a subset of components. Reports recall on the 97 confirmed hallucinations, false
positive rate on clean steps, flags on B (process error, not hallucination) / C (ambiguous), and a projected F1 at the
test base rate (23.7%, measured with the all-ones probe).

python experiments/apb_eval.py --dir runs/cc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian import combine  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian.parse import parse_example  # noqa: E402

BASE_RATE = 0.237


def load(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("p") is not None:
                out[r["id"]] = r["p"] >= 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs" / "cc"))
    ap.add_argument("--eval", default=str(ROOT / "runs" / "apb_eval.parquet"))
    args = ap.parse_args()
    d = pd.read_parquet(args.eval)
    cfg = combine.load(str(ROOT / "model" / "calibration.json"))
    hard = {r.id for r in d.itertuples() if combine.final_p(flags(parse_example(r.prompt, r.response))[0], 0.0, cfg) >= 1.0}
    comp = {"hard": {i: (i in hard) for i in d.id}}
    for name, f in [("v3f", "apb_qwen_v3f"), ("v2", "apb_qwen_v2"), ("num", "apb_qwen_num"), ("gemma", "apb_gemma_v3f")]:
        comp[name] = load(Path(args.dir) / f"{f}.jsonl")
    lab = dict(zip(d.id, d.label))
    variants = {
        "hard only": ["hard"],
        "Qwen v3f + hard": ["hard", "v3f"],
        "sub11 (hard+v3f+v2+num)": ["hard", "v3f", "v2", "num"],
        "sub12 (hard+v3f+num+gemma)": ["hard", "v3f", "num", "gemma"],
        "all (hard+v3f+v2+num+gemma)": ["hard", "v3f", "v2", "num", "gemma"],
        "Gemma alone + hard": ["hard", "gemma"],
    }
    print(f"rows: pos={sum(v == 1 for v in lab.values())} neg={sum(v == 0 for v in lab.values())} "
          f"B={sum(v == -1 for v in lab.values())} C={sum(v == -2 for v in lab.values())}; "
          f"coverage: " + ", ".join(f"{k}={len(v)}" for k, v in comp.items()))
    for name, parts in variants.items():
        if any(len(comp[p]) == 0 for p in parts):
            continue
        flag = {i: any(comp[p].get(i, False) for p in parts) for i in d.id}
        pos = [i for i in d.id if lab[i] == 1]
        neg = [i for i in d.id if lab[i] == 0]
        r = sum(flag[i] for i in pos) / len(pos)
        fpr = sum(flag[i] for i in neg) / len(neg)
        prec = r * BASE_RATE / max(1e-9, r * BASE_RATE + fpr * (1 - BASE_RATE))
        f1 = 2 * prec * r / max(1e-9, prec + r)
        b = sum(flag[i] for i in d.id if lab[i] == -1)
        c = sum(flag[i] for i in d.id if lab[i] == -2)
        print(f"{name:32s} recall {r:.3f} ({sum(flag[i] for i in pos)}/{len(pos)})  FPR {fpr:.3f} "
              f"({sum(flag[i] for i in neg)}/{len(neg)})  B {b}/33  C {c}/13  | @23.7%: P {prec:.3f} F1 {f1:.3f}")
    # by domain for the two main candidates
    for name in ["sub11 (hard+v3f+v2+num)", "sub12 (hard+v3f+num+gemma)"]:
        parts = variants[name]
        if any(len(comp[p]) == 0 for p in parts):
            continue
        flag = {i: any(comp[p].get(i, False) for p in parts) for i in d.id}
        for dom, g in d.groupby("domain"):
            pos = [i for i in g.id if lab[i] == 1]
            neg = [i for i in g.id if lab[i] == 0]
            print(f"   {name[:5]} {dom:8s} recall {sum(flag[i] for i in pos)}/{len(pos)}  FP {sum(flag[i] for i in neg)}/{len(neg)}")
    # unique contributions
    base = {i: comp["hard"].get(i) or comp["v3f"].get(i, False) or comp["num"].get(i, False) for i in d.id}
    for extra in ["v2", "gemma"]:
        if not comp[extra]:
            continue
        add = [i for i in d.id if comp[extra].get(i) and not base[i]]
        tp = sum(lab[i] == 1 for i in add)
        fp = sum(lab[i] == 0 for i in add)
        print(f"unique adds of {extra:6s} over hard+v3f+num: {len(add)} (true {tp}, clean {fp}, B {sum(lab[i] == -1 for i in add)}, "
              f"C {sum(lab[i] == -2 for i in add)}) -> precision on labelled {tp / max(1, tp + fp):.2f}")


if __name__ == "__main__":
    main()
