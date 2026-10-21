"""LoRA training of the judge on vLLM rollouts: rejection-sampling SFT or GSPO (sequence-level policy optimisation).

SFT  : NLL of correct sampled reasonings (reward 1), at most --sft-per-row per row, assistant tokens only.
GSPO : group-normalised advantages per row (8 samples), sequence-level importance ratio
       s = exp(mean_t log pi_new - mean_t log pi_old), clipped objective  -min(s*A, clip(s, 1-el, 1+eh)*A).
       pi_old = the policy that generated the rollouts, recomputed with transformers before the update (--old hf),
       so vLLM/transformers numeric mismatch does not enter the ratio; the mismatch is corrected with a capped
       sequence-level importance weight w = min(exp(mean(lp_old_hf - lp_vllm)), --tis-cap) (detached).
       One optimizer step per mini-batch of --prompts-per-step rows.

python experiments/lora/train.py --mode gspo --base /workspace/models/qwen36_bf16 --adapter runs/it0 \
    --rollouts rollouts_it1.pkl --data pool.parquet --out runs/it1
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chat_text, completion_logprobs, load_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "gspo"], required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="continue from this LoRA adapter")
    ap.add_argument("--rollouts", required=True, nargs="+")
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--sft-per-row", type=int, default=2)
    ap.add_argument("--sft-max-tokens", type=int, default=6000)
    ap.add_argument("--clip-low", type=float, default=3e-4)
    ap.add_argument("--clip-high", type=float, default=4e-4)
    ap.add_argument("--tis-cap", type=float, default=2.0)
    ap.add_argument("--old", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ids", default=None, help="json list: train only on these row ids (selection of select_rows.py)")
    ap.add_argument("--sft-max-neg-share", type=float, default=0.6, help="cap of label-0 samples in SFT")
    ap.add_argument("--no-experts", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(p) for p in a.data], ignore_index=True).drop_duplicates("id").set_index("id")
    ro = [x for f in a.rollouts for x in pickle.load(open(f, "rb"))]
    if a.ids:
        keep = set(json.load(open(a.ids)))
        ro = [x for x in ro if x["id"] in keep]
    tok = AutoTokenizer.from_pretrained(a.base)
    model = load_policy(a.base, a.adapter, trainable=True, experts=not a.no_experts)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = a.lr or (5e-5 if a.mode == "sft" else 1e-5)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    prompt_cache: dict[str, list[int]] = {}

    def prompt_ids(rid):
        if rid not in prompt_cache:
            r = df.loc[rid]
            prompt_cache[rid] = tok(chat_text(tok, r.prompt, r.response), add_special_tokens=False).input_ids
        return prompt_cache[rid]

    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    step = 0
    if a.mode == "sft":
        by = defaultdict(list)
        for x in ro:
            if x["reward"] == 1.0 and x["n_tokens"] <= a.sft_max_tokens and x["id"] in df.index:
                by[x["id"]].append(x)
        items = [s for v in by.values() for s in random.sample(v, min(a.sft_per_row, len(v)))]
        pos = [x for x in items if x["label"] == 1]
        neg = [x for x in items if x["label"] == 0]
        cap = int(a.sft_max_neg_share / (1 - a.sft_max_neg_share) * len(pos))
        if len(neg) > cap:  # do not let the (easy, plentiful) clean rows pull the judge towards "no hallucination"
            neg = random.sample(neg, cap)
        items = pos + neg
        print(f"SFT: {len(items)} samples (pos {len(pos)}, neg {len(neg)}) from {len(by)} rows, lr {lr}", flush=True)
        bs = a.prompts_per_step
        for ep in range(a.epochs):
            random.shuffle(items)
            for i in range(0, len(items), bs):
                batch = items[i:i + bs]
                ntok = sum(len(x["completion_ids"]) for x in batch)
                tot = 0.0
                for x in batch:
                    lp = completion_logprobs(model, prompt_ids(x["id"]), list(map(int, x["completion_ids"])))
                    loss = -lp.sum() / ntok
                    loss.backward()
                    tot += float(loss)
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = dict(step=step, ep=ep, loss=tot, grad_norm=float(gn), sec=round(time.time() - t0))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
                if a.max_steps and step >= a.max_steps:
                    break
    else:
        groups = defaultdict(list)
        for x in ro:
            if x["id"] in df.index:
                groups[x["id"]].append(x)
        usable = []
        for rid, g in groups.items():
            r = np.array([x["reward"] for x in g], dtype=np.float32)
            if r.std() < 1e-6:
                continue  # all samples agree: no learning signal
            adv = (r - r.mean()) / (r.std() + 1e-4)
            for x, aa in zip(g, adv):
                x["adv"] = float(aa)
            usable.append(rid)
        random.shuffle(usable)
        if a.max_steps:
            usable = usable[:a.max_steps * a.prompts_per_step]
        print(f"GSPO: {len(usable)} informative rows of {len(groups)}, lr {lr}", flush=True)
        # pi_old = the rollout policy (parameters BEFORE any update of this iteration), recomputed with transformers
        # in one no-grad pass, so that later mini-batches get a proper off-policy ratio
        for rid in usable:
            for x in groups[rid]:
                if a.old == "hf" and len(x["completion_ids"]):
                    with torch.no_grad():
                        x["lp_old"] = completion_logprobs(model, prompt_ids(x["id"]),
                                                          list(map(int, x["completion_ids"]))).float().cpu().numpy()
                else:
                    x["lp_old"] = x["vllm_lp"]
        print(f"old log-probs done {(time.time() - t0) / 60:.1f} min", flush=True)
        for i in range(0, len(usable), a.prompts_per_step):
            rows = usable[i:i + a.prompts_per_step]
            samples = [x for rid in rows for x in groups[rid]]
            stats = defaultdict(float)
            n = len(samples)
            for x in samples:
                ids = list(map(int, x["completion_ids"]))
                if not ids:
                    continue
                lp = completion_logprobs(model, prompt_ids(x["id"]), ids)
                old = torch.tensor(x["lp_old"], device=lp.device, dtype=torch.float32)
                s = torch.exp(lp.float().mean() - old.mean())
                A = x["adv"]
                obj = torch.minimum(s * A, torch.clamp(s, 1 - a.clip_low, 1 + a.clip_high) * A)
                mism = float(np.mean(x["lp_old"]) - np.mean(x["vllm_lp"])) if len(x["vllm_lp"]) else 0.0
                w = min(math.exp(mism), a.tis_cap)
                loss = -(w * obj) / n
                loss.backward()
                stats["ratio"] += float(s) / n
                stats["clipped"] += float(abs(float(s) - 1) > max(a.clip_low, a.clip_high)) / n
                stats["mismatch_abs"] += abs(mism) / n
                stats["k3_vllm_hf"] += float(np.mean(np.exp(x["vllm_lp"] - x["lp_old"]) - 1 - (x["vllm_lp"] - x["lp_old"]))) / n
                stats["reward"] += x["reward"] / n
                stats["len"] += len(ids) / n
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = dict(step=step, rows=len(rows), samples=n, grad_norm=float(gn), sec=round(time.time() - t0),
                       **{k: round(v, 5) for k, v in stats.items()})
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            if a.max_steps and step >= a.max_steps:
                break
    model.save_pretrained(out / "adapter")
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    print("saved", out / "adapter", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
