"""LoRA fine-tuning of a small causal LM as a fast hallucination classifier (answer = one token "1"/"0").

Train on synthetic rows (experiments/synth*.py) of the dialogues in --train-folds, evaluate P(1) on the real rows
(timing_431_labeled: valid + early turns) of the held-out dialogues. Fold = hash of the dialogue id.

python experiments/train_cls.py --model /workspace/g/models/q35_4b --data synth_det.parquet,synth_llm.parquet \
    --fold 0 --out /workspace/g/exp/cls_f0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian.judge import fit_context  # noqa: E402

QUESTION = ("\n\nDoes the agent turn above contain a contextual hallucination (an unsupported or contradicting "
            "statement or action, a call to an unavailable tool, wrong arguments, a policy violation, a false "
            "refusal)? Answer 1 or 0.\nAnswer:")


def fold_of(traj: str, k: int = 2) -> int:
    return int(hashlib.md5(traj.encode()).hexdigest(), 16) % k


def make_text(prompt: str, response: str, max_chars: int) -> str:
    ctx = fit_context(prompt, max_chars) if len(prompt) > max_chars else prompt
    return f"<context>\n{ctx}\n</context>\n<turn_under_review>\n{response}\n</turn_under_review>{QUESTION}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="comma-separated synthetic parquet files")
    ap.add_argument("--eval", default=str(ROOT / "timing_431_labeled.parquet"))
    ap.add_argument("--fold", type=int, default=-1, help="held-out fold (-1 = train on everything)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    args = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    id1, id0 = tok.encode(" 1", add_special_tokens=False)[-1], tok.encode(" 0", add_special_tokens=False)[-1]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    syn = pd.concat([pd.read_parquet(p) for p in args.data.split(",")], ignore_index=True)
    if args.fold >= 0:
        syn = syn[syn.traj.map(fold_of) != args.fold]
    rnd = random.Random(0)
    train = syn.sample(frac=1.0, random_state=0).reset_index(drop=True)
    print("train rows", len(train), train.label.value_counts().to_dict(), flush=True)

    def score_logits(text):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=16000).input_ids.cuda()
        logits = model(input_ids=ids).logits[0, -1]
        return logits[[id0, id1]]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 30) * max(0.05, 1 - s / max(1, steps // args.grad_accum)))
    model.train()
    t0, run_loss = time.time(), 0.0
    for step in range(steps):
        r = train.iloc[step % len(train)]
        two = score_logits(make_text(r.prompt, r.response, args.max_chars))
        w = args.pos_weight if r.label == 1 else 1.0
        loss = w * torch.nn.functional.cross_entropy(two[None].float(), torch.tensor([int(r.label)]).cuda())
        (loss / args.grad_accum).backward()
        run_loss += loss.item()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{steps} loss {run_loss / 100:.4f} {time.time() - t0:.0f}s", flush=True)
            run_loss = 0.0
    model.save_pretrained(out / "adapter")

    # evaluation on the real rows of the held-out dialogues (or all rows when fold == -1)
    model.eval()
    ev = pd.read_parquet(args.eval)
    ev["traj"] = ev.id.str.split("::").str[0]
    if args.fold >= 0:
        ev = ev[ev.traj.map(fold_of) == args.fold]
    res = []
    with torch.no_grad():
        for r in ev.itertuples(index=False):
            two = score_logits(make_text(r.prompt, r.response, args.max_chars)).float()
            p1 = torch.softmax(two, -1)[1].item()
            res.append({"id": r.id, "label": int(r.label), "p": p1})
    with open(out / "eval.jsonl", "w") as fh:
        for x in res:
            fh.write(json.dumps(x) + "\n")
    v = [x for x in res if x["label"] >= 0]
    for thr in (0.3, 0.5, 0.7, 0.9):
        tp = sum(1 for x in v if x["label"] == 1 and x["p"] >= thr)
        fp = sum(1 for x in v if x["label"] == 0 and x["p"] >= thr)
        fn = sum(1 for x in v if x["label"] == 1 and x["p"] < thr)
        h = [x for x in res if x["label"] < 0]
        print(f"thr {thr}: valid tp/fp/fn {tp}/{fp}/{fn} | history flagged "
              f"{sum(1 for x in h if x['p'] >= thr)}/{len(h)}", flush=True)


if __name__ == "__main__":
    main()
