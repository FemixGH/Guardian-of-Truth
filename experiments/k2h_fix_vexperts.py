"""K2-Horizon AWQ (compressed-tensors pack-quantized int4, asym, group 32): vLLM 0.31's MoVA path reads
v_experts[*].weight directly, which fails on quantized experts. Dequantize all v_experts to bf16 in place and add
them to the quantization ignore list.  python k2h_fix_vexperts.py <model_dir>"""
import json, os, re, sys
import torch
from safetensors.torch import load_file, save_file

d = sys.argv[1]
idx_p = os.path.join(d, "model.safetensors.index.json")
idx = json.load(open(idx_p))
wm = idx["weight_map"]


def unpack(p, dim):  # int32 -> 8 int4 values along dim, signed via offset 8
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    if dim == 1:
        u = (p.unsqueeze(-1) >> sh) & 0xF
        return (u.reshape(p.shape[0], -1) - 8).to(torch.int16)
    u = (p.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF
    return (u.reshape(-1, p.shape[1]) - 8).to(torch.int16)


shards = sorted({f for k, f in wm.items() if ".v_experts." in k})
done_layers = []
for sh in shards:
    t = load_file(os.path.join(d, sh))
    pref = sorted({k.rsplit(".", 1)[0] for k in t if ".v_experts." in k and k.endswith("weight_packed")})
    for pfx in pref:
        q = unpack(t[pfx + ".weight_packed"], 1)
        out, inn = t[pfx + ".weight_shape"].tolist()
        q = q[:out, :inn].float()
        zp = unpack(t[pfx + ".weight_zero_point"], 0)[:out].float()
        sc = t[pfx + ".weight_scale"].float()
        g = inn // sc.shape[1]
        w = (q - zp.repeat_interleave(g, 1)) * sc.repeat_interleave(g, 1)
        for s in ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point"):
            del t[pfx + "." + s]
            wm.pop(pfx + "." + s, None)
        t[pfx + ".weight"] = w.to(torch.bfloat16).contiguous()
        wm[pfx + ".weight"] = sh
    save_file(t, os.path.join(d, sh), metadata={"format": "pt"})
    print(sh, len(pref), "experts dequantized", flush=True)
json.dump(idx, open(idx_p, "w"), indent=2)
c = json.load(open(os.path.join(d, "config.json")))
ig = c["quantization_config"]["ignore"]
if "re:.*v_experts.*" not in ig:
    ig.append("re:.*v_experts.*")
json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)
print("ok; sample weight std", float(w.std()), "absmax", float(w.abs().max()))
