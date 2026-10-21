"""Build a SYMMETRIC int4 (group 32, compressed-tensors pack-quantized) K2-Horizon checkpoint from the official BF16
weights, quantizing exactly the modules the cyankiwi AWQ build quantized (routed experts + q/k/o/gate_proj of the MoVA
layers). vLLM 0.19.1 (the platform) only runs symmetric int4 MoE; the cyankiwi build is asymmetric. Per group the clip
ratio with the lowest squared error is chosen (like the MSE observer the cyankiwi build used). MoVA v_experts and
everything in cyankiwi's ignore list stay BF16.
    python quant_sym.py <bf16_dir> <awq_dir (module list + quant config)> <out_dir> [cuda device]"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, AWQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEV = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"
G = 32
CLIPS = torch.tensor([1.0, 0.95, 0.9, 0.85, 0.8, 0.75])

os.makedirs(OUT, exist_ok=True)
wa = json.load(open(f"{AWQ}/model.safetensors.index.json"))["weight_map"]
qmods = {k[: -len(".weight_packed")] for k in wa if k.endswith(".weight_packed")}
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]


def quant(w: torch.Tensor):
    """w (out, in) bf16 -> packed int32 (out, in/8), scale bf16 (out, in/G); q in [-8, 7], stored as q + 8."""
    out_f, in_f = w.shape
    x = w.to(DEV, torch.float32).reshape(out_f, in_f // G, G)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    best_err, best_s = None, None
    for c in CLIPS.tolist():
        s = amax * c / 7.0
        q = torch.clamp(torch.round(x / s), -8, 7)
        err = ((q * s - x) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_s = err, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s, best_s)
    s = best_s.to(torch.bfloat16).to(torch.float32)  # quantize against the scale that is actually stored
    q = torch.clamp(torch.round(x / s), -8, 7).reshape(out_f, in_f).to(torch.int64) + 8
    q = q.reshape(out_f, in_f // 8, 8)
    shifts = torch.arange(0, 32, 4, device=DEV, dtype=torch.int64)
    packed = (q << shifts).sum(-1)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    return packed.cpu(), s.reshape(out_f, in_f // G).to(torch.bfloat16).cpu()


def unpack(p: torch.Tensor):
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    u = (p.unsqueeze(-1) >> sh) & 0xF
    return (u.reshape(p.shape[0], -1) - 8).to(torch.int16)


new_wm, checked = {}, 0
for shard in sorted(set(wm.values())):
    tensors = {}
    with safe_open(f"{SRC}/{shard}", "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            mod = k[: -len(".weight")] if k.endswith(".weight") else None
            if mod in qmods:
                packed, scale = quant(t)
                if checked < 3:  # round-trip check of the packing
                    deq = unpack(packed).float() * scale.float().repeat_interleave(G, 1)
                    rel = ((deq - t.float()).norm() / t.float().norm()).item()
                    print(f"check {mod}: rel. error {rel:.4f}", flush=True)
                    assert rel < 0.2, rel
                    checked += 1
                tensors[mod + ".weight_packed"] = packed
                tensors[mod + ".weight_scale"] = scale
                tensors[mod + ".weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
            else:
                tensors[k] = t.contiguous()
    save_file(tensors, f"{OUT}/{shard}", metadata={"format": "pt"})
    for k in tensors:
        new_wm[k] = shard
    print(shard, len(tensors), "tensors", flush=True)

json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_wm}, open(f"{OUT}/model.safetensors.index.json", "w"),
          indent=2)
cfg = json.load(open(f"{SRC}/config.json"))
qc = json.load(open(f"{AWQ}/config.json"))["quantization_config"]
w = qc["config_groups"]["group_0"]["weights"]
w["symmetric"] = True
w.pop("zp_dtype", None)
w["observer"] = "mse"
if "re:.*v_experts.*" not in qc["ignore"]:
    qc["ignore"].append("re:.*v_experts.*")
qc["ignore"] = sorted(set(qc["ignore"]) - {m for m in qmods})  # never ignore a module we quantized
cfg["quantization_config"] = qc
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2)
for fn in os.listdir(SRC):
    if not fn.endswith(".safetensors") and fn not in ("model.safetensors.index.json", "config.json") \
            and os.path.isfile(f"{SRC}/{fn}"):
        shutil.copy(f"{SRC}/{fn}", f"{OUT}/{fn}")
print("DONE", len(new_wm), "tensors")
