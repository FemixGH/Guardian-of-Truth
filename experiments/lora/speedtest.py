"""Go / no-go measurement before any long run: training-step cost of the real model.

Loads the policy (bf16 + LoRA, grouped_mm experts, gradient checkpointing) and times forward+backward of one
sequence for several (prompt, completion) lengths taken from real pool rows; reports peak GPU memory. With
--rollouts it also measures the vLLM vs transformers log-prob mismatch on real samples.

python experiments/lora/speedtest.py --base /workspace/g/models/qwen36_bf16 --data pool.parquet [--rollouts ro.pkl]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chat_text, completion_logprobs, load_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--completion", type=int, nargs="+", default=[1500, 4000])
    ap.add_argument("--no-experts", action="store_true", help="LoRA without the routed MoE experts")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    df = pd.read_parquet(a.data)
    t0 = time.time()
    model = load_policy(a.base, None, trainable=True, experts=not a.no_experts)
    print(f"load {time.time() - t0:.0f}s, mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    lens = df.prompt.str.len()
    picks = [df.iloc[(lens - q).abs().argmin()] for q in (lens.quantile(0.5), lens.quantile(0.95))]
    for r in picks:
        pids = tok(chat_text(tok, r.prompt, r.response), add_special_tokens=False).input_ids
        for c in a.completion:
            comp = np.random.randint(1000, 50000, c).tolist()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); t = time.time()
            with torch.no_grad():
                completion_logprobs(model, pids, comp)
            torch.cuda.synchronize(); tf = time.time() - t
            t = time.time()
            lp = completion_logprobs(model, pids, comp)
            (-lp.mean()).backward()
            torch.cuda.synchronize(); tb = time.time() - t
            model.zero_grad(set_to_none=True)
            print(f"prompt {len(pids):6d} tok + completion {c:5d}: no-grad fwd {tf:5.1f}s | fwd+bwd {tb:5.1f}s | "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if a.rollouts:
        ro = pickle.load(open(a.rollouts, "rb"))[:16]
        rows = df.set_index("id")
        d = []
        for x in ro:
            r = rows.loc[x["id"]]
            pids = tok(chat_text(tok, r.prompt, r.response), add_special_tokens=False).input_ids
            with torch.no_grad():
                hf = completion_logprobs(model, pids, list(map(int, x["completion_ids"]))).float().cpu().numpy()
            v = x["vllm_lp"]
            d.append((np.mean(np.abs(hf - v)), np.mean(np.exp(v - hf) - 1 - (v - hf))))
        print(f"vLLM vs HF on {len(d)} real samples: mean|dlp| {np.mean([x[0] for x in d]):.4f}  "
              f"k3 {np.mean([x[1] for x in d]):.2e}  (Qwen3-30B-A3B reference ~1.5e-3)", flush=True)


if __name__ == "__main__":
    main()
