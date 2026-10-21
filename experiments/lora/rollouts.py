"""Sample N judge reasonings per row with vLLM (offline engine) and score them against the labels.

Output (pickle): list of dicts, one per sample: id, label, k, completion_ids, vllm_lp (per-token logprobs of the
sampled tokens), verdict, reward, truncated, n_tokens; plus a short text summary per row.

python experiments/lora/rollouts.py --model /workspace/models/merged_it0 --data pool.parquet --ids ids.json \
    --n 8 --out rollouts_it1.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chat_text, reward, verdict_of  # noqa: E402

import re  # noqa: E402

LEAK_RE = re.compile(r"(?i)reference annotation|reference verdict|annotation note|this note|the note says|"
                     r"эталон|разметк|подсказк|аннотац")
# no JIT-compiled FlashInfer sampler (needs nvcc + a matching toolchain; plain sampling is enough here)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--ids", default=None, help="json list of row ids to use (default: all rows)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None, help="LoRA adapter applied on top of --model (e.g. FP8 base + adapter)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs for generation")
    ap.add_argument("--moe-backend", default="triton", help="bf16 MoE kernels: triton (no nvcc JIT) or auto")
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs / compilation (local smoke tests)")
    ap.add_argument("--hint", action="store_true", help="STaR rationalization: tell the model the gold verdict")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    df = pd.concat([pd.read_parquet(p) for p in a.data], ignore_index=True).drop_duplicates("id")
    if a.ids:
        keep = set(json.load(open(a.ids)))
        df = df[df.id.isin(keep)]
    if df.empty:
        raise SystemExit("rollouts: no rows selected (empty --ids?)")
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    prompts = []
    for r in df.itertuples(index=False):
        t = chat_text(tok, r.prompt, r.response)
        if a.hint:  # gold label as privileged hint; the training target is later re-attached to the plain prompt
            hint = ("(Reference annotation for this turn: VERDICT " + str(int(r.label)) + ". Explain from the "
                    "context why this is the correct verdict, as an independent audit, without mentioning this note.)")
            t = t.replace("<|im_end|>\n<|im_start|>assistant\n", "\n" + hint + "<|im_end|>\n<|im_start|>assistant\n", 1)
        prompts.append(tok(t, add_special_tokens=False).input_ids)
    lens = [len(p) for p in prompts]
    print(f"{len(prompts)} prompts, prompt tokens median {int(np.median(lens))} max {max(lens)}", flush=True)
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, language_model_only=True,
              max_num_seqs=128, seed=a.seed, enforce_eager=a.eager, moe_backend=a.moe_backend,
              tensor_parallel_size=a.tp, **({"enable_lora": True, "max_lora_rank": 64, "max_loras": 1} if a.lora else {}))
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, logprobs=0,
                        seed=a.seed)
    t0 = time.time()
    lr = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lr = LoRARequest("adapter", 1, a.lora)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, lora_request=lr)
    dt = time.time() - t0
    res, summ = [], []
    for r, o in zip(df.itertuples(index=False), outs):
        vs = []
        for k, c in enumerate(o.outputs):
            ids = list(c.token_ids)
            lp = [next(iter(d.values())).logprob if d else 0.0 for d in (c.logprobs or [])]
            text = tok.decode(ids, skip_special_tokens=False)
            v = verdict_of(text)
            trunc = c.finish_reason == "length"
            leak = a.hint and bool(LEAK_RE.search(text))  # rationale cites the hint -> unusable as a training target
            res.append(dict(id=r.id, label=int(r.label), k=k, completion_ids=np.array(ids, dtype=np.int32),
                            vllm_lp=np.array(lp, dtype=np.float32), verdict=v, truncated=trunc, n_tokens=len(ids),
                            reward=0.0 if leak else reward(v, int(r.label), trunc), leak=leak,
                            text=text if k < 2 else None))
            vs.append(v)
        summ.append(dict(id=r.id, label=int(r.label), votes1=sum(1 for v in vs if v == 1),
                         votes0=sum(1 for v in vs if v == 0), none=sum(1 for v in vs if v is None)))
    pickle.dump(res, open(a.out, "wb"))
    s = pd.DataFrame(summ)
    s.to_csv(a.out.replace(".pkl", "_summary.csv"), index=False)
    gen = sum(x["n_tokens"] for x in res)
    acc = np.mean([x["reward"] for x in res])
    inform = ((s.votes1 > 0) & (s.votes0 > 0)).mean()
    if a.hint:
        print(f"hinted samples citing the hint (dropped): {sum(x['leak'] for x in res)} of {len(res)}", flush=True)
    print(f"done in {dt / 60:.1f} min, {gen / dt:.0f} gen tok/s, mean reward {acc:.3f}, informative rows {inform:.2f}, "
          f"truncated {np.mean([x['truncated'] for x in res]):.3f}, median len {int(np.median([x['n_tokens'] for x in res]))}",
          flush=True)


if __name__ == "__main__":
    main()
