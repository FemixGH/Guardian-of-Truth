"""Block-FP8 checkpoint -> bf16 checkpoint with EXACTLY the FP8 weight values (w = q * scale_inv per 128x128 block).

Training the LoRA on this base makes the adapter correct the model that is actually deployed (FP8 + adapter in vLLM),
instead of the original bf16 weights (an adapter trained on bf16 lost most of its effect on the FP8 base).

python experiments/lora/dequant_fp8.py --fp8 /workspace/g/models/qwen36_fp8 --out /workspace/g/models/qwen36_fp8deq
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant(q: torch.Tensor, s: torch.Tensor, block: int = 128) -> torch.Tensor:
    o, i = q.shape
    sb = s.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:o, :i]
    return (q.float() * sb).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.fp8), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(src / "model.safetensors.index.json"))
    by = defaultdict(list)
    for k, f in idx["weight_map"].items():
        by[f].append(k)
    new_map, n = {}, 0
    for f, keys in sorted(by.items()):
        with safe_open(src / f, "pt", device="cpu") as h:
            t = {k: h.get_tensor(k) for k in keys}
        res = {}
        for k, v in t.items():
            if k.endswith("_scale_inv"):
                continue
            if k + "_scale_inv" in t:
                v = dequant(v, t[k + "_scale_inv"])
                n += 1
            res[k] = v.contiguous()
            new_map[k] = f
        save_file(res, out / f, metadata={"format": "pt"})
        print(f, len(res), flush=True)
    json.dump({"metadata": {}, "weight_map": new_map}, open(out / "model.safetensors.index.json", "w"), indent=1)
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json":
            shutil.copy(p, out / p.name)
    cfg = json.load(open(out / "config.json"))
    cfg.pop("quantization_config", None)
    if "text_config" in cfg:
        cfg["text_config"].pop("quantization_config", None)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print("dequantised tensors:", n, "->", out)


if __name__ == "__main__":
    main()
