"""Build an 8-bit K2-Horizon checkpoint (< 40 GB) for vLLM 0.19.1 from the official BF16 weights.
- routed experts, shared experts, attention projections, dense MLPs: symmetric int8, per-output-channel scales,
  compressed-tensors pack-quantized (4 values per int32, stored as q + 128) -> Marlin W8A16 on A100.
- MoVA value experts: raw int8 [N, K] + float32 per-channel scale [N] under `...v_experts.<e>.weight_int8` /
  `...weight_scale`, consumed by the K2 port's int8 MoVA path (fused_mova_impl use_int8_w8a16).
- embeddings, lm_head, norms, routers, biases: BF16.
    python quant_int8.py <bf16_dir> <out_dir> [cuda device]"""
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, OUT = sys.argv[1], sys.argv[2]
DEV = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
os.makedirs(OUT, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]
LIN = re.compile(r"\.(mlp\.experts\.\d+\.(gate|up|down)_proj|mlp\.shared_experts\.(gate|up|down)_proj|"
                 r"mlp\.(gate|up|down)_proj|self_attn\.(q|k|v|o|gate)_proj)\.weight$")
VEXP = re.compile(r"\.self_attn\.v_experts\.\d+\.weight$")


def q8(w):
    x = w.to(DEV, torch.float32)
    s = (x.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-10)
    s = s.to(torch.bfloat16).to(torch.float32)
    q = torch.clamp(torch.round(x / s), -128, 127)
    return q, s


def pack8(q):  # (out, in) in [-128, 127] -> int32 (out, in/4), stored unsigned as q + 128
    u = (q.to(torch.int64) + 128).reshape(q.shape[0], -1, 4)
    sh = torch.arange(0, 32, 8, device=u.device, dtype=torch.int64)
    p = (u << sh).sum(-1)
    return torch.where(p >= 2 ** 31, p - 2 ** 32, p).to(torch.int32).cpu()


def unpack8(p):
    sh = torch.arange(0, 32, 8, dtype=torch.int32)
    return (((p.unsqueeze(-1) >> sh) & 0xFF).reshape(p.shape[0], -1) - 128).float()


quantized, checked, new_wm = [], 0, {}
for shard in sorted(set(wm.values())):
    out = {}
    with safe_open(f"{SRC}/{shard}", "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if LIN.search(k):
                q, s = q8(t)
                packed = pack8(q)
                m = k[: -len(".weight")]
                if checked < 2:
                    rel = ((unpack8(packed) * s.cpu() - t.float()).norm() / t.float().norm()).item()
                    print(f"check {m}: rel. error {rel:.5f}", flush=True)
                    assert rel < 0.05, rel
                    checked += 1
                out[m + ".weight_packed"] = packed
                out[m + ".weight_scale"] = s.to(torch.bfloat16).cpu()
                out[m + ".weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
                quantized.append(m)
            elif VEXP.search(k):
                q, s = q8(t)
                out[k[: -len(".weight")] + ".weight_int8"] = q.to(torch.int8).cpu()
                out[k[: -len(".weight")] + ".weight_scale"] = s.reshape(-1).cpu()
            else:
                out[k] = t.contiguous()
    save_file(out, f"{OUT}/{shard}", metadata={"format": "pt"})
    new_wm.update({k: shard for k in out})
    print(shard, len(out), flush=True)

json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_wm}, open(f"{OUT}/model.safetensors.index.json", "w"),
          indent=2)
cfg = json.load(open(f"{SRC}/config.json"))
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
    "config_groups": {"group_0": {"targets": ["Linear"], "format": "pack-quantized", "input_activations": None,
                                  "output_activations": None,
                                  "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
                                              "group_size": None, "dynamic": False, "actorder": None,
                                              "block_structure": None, "observer": "minmax", "observer_kwargs": {}}}},
    # everything not quantized above: routers, lm_head, value experts (own int8 path in the port)
    "ignore": ["lm_head", "re:.*\\.mlp\\.gate$", "re:.*v_router.*", "re:.*v_experts.*"],
    "kv_cache_scheme": None, "sparsity_config": {}, "transform_config": {}, "global_compression_ratio": None,
}
cfg["mova_v_experts_int8"] = True
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2)
for fn in os.listdir(SRC):
    if not fn.endswith(".safetensors") and fn not in ("model.safetensors.index.json", "config.json") \
            and os.path.isfile(f"{SRC}/{fn}"):
        shutil.copy(f"{SRC}/{fn}", f"{OUT}/{fn}")
print("DONE", len(quantized), "linear modules int8;", len(new_wm), "tensors")
