"""Copy a HF checkpoint dropping tensors whose names start with the given prefixes (vision tower, MTP head).

python experiments/strip_weights.py SRC DST model.visual,mtp
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

src, dst, prefixes = sys.argv[1], sys.argv[2], tuple(sys.argv[3].split(","))
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if os.path.isfile(p) and not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        shutil.copy2(p, os.path.join(dst, f))
weight_map, dropped, kept = {}, 0, 0
for f in sorted(f for f in os.listdir(src) if f.endswith(".safetensors")):
    out = {}
    with safe_open(os.path.join(src, f), "pt") as fh:
        meta = fh.metadata()
        for k in fh.keys():
            if k.startswith(prefixes):
                dropped += 1
                continue
            out[k] = fh.get_tensor(k)
    if out:
        save_file(out, os.path.join(dst, f), metadata=meta or {"format": "pt"})
        weight_map.update({k: f for k in out})
        kept += len(out)
total = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst) if f.endswith(".safetensors"))
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
print(f"{src} -> {dst}: kept {kept}, dropped {dropped}, {total / 1e9:.2f} GB")
