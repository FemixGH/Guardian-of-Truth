"""Merge a LoRA adapter into the checkpoint files.

bf16 : base = Qwen/Qwen3.6-35B-A3B (bf16). The adapter is merged with PEFT in transformers, every tensor that changed
       is written back under its checkpoint name; unchanged shards are hard-linked. Output loads in vLLM as before.
fp8  : base = Qwen/Qwen3.6-35B-A3B-FP8 (block FP8 128x128, weight * weight_scale_inv). Changed tensors are taken from
       a merged bf16 directory (--merged), re-quantised per block; fused experts gate_up_proj / down_proj are split
       into the per-expert gate/up/down tensors of the FP8 checkpoint. Other tensors are copied unchanged.

python experiments/lora/merge.py bf16 --base qwen36_bf16 --adapter runs/it1/adapter --out merged_it1
python experiments/lora/merge.py fp8  --base qwen36_bf16 --merged merged_it1 --fp8 qwen36_fp8 --out final_fp8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

LM = "model.language_model."


def ckpt_name(hf_name: str) -> str:
    """Parameter name of the text-only CausalLM (model.layers...) -> name in the multimodal checkpoint."""
    if hf_name.startswith("model.") and not hf_name.startswith(LM):
        return LM + hf_name[len("model."):]
    return hf_name


def expand(name: str, t: torch.Tensor):
    """A merged tensor under its fused name AND, for MoE experts, under per-expert names
    (checkpoints store experts either fused - mlp.experts.gate_up_proj [E, 2I, H] - or per expert)."""
    yield name, t
    m = re.match(r"(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$", name)
    if m:
        pre, kind = m.groups()
        for e in range(t.shape[0]):
            if kind == "gate_up_proj":
                half = t.shape[1] // 2
                yield f"{pre}.{e}.gate_proj.weight", t[e, :half]
                yield f"{pre}.{e}.up_proj.weight", t[e, half:]
            else:
                yield f"{pre}.{e}.down_proj.weight", t[e]


def copy_aux(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, dst / f.name)


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def merge_bf16(a):
    from common import load_policy
    base, out = Path(a.base), Path(a.out)
    model = load_policy(str(base), a.adapter, trainable=False)
    model = model.merge_and_unload()
    from common import LORA_TARGET_MODULES
    tmod = re.compile(LORA_TARGET_MODULES)

    def is_target(name: str) -> bool:
        # ONLY tensors that carry a LoRA delta are written back; everything else stays bit-identical (the checkpoint
        # keeps e.g. linear_attn.norm in float32 - a bf16 round-trip would silently degrade it)
        if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            return True
        return name.endswith(".weight") and bool(tmod.fullmatch(name[:-len(".weight")]))

    sd = {}
    for k, v in model.state_dict().items():
        if not is_target(k):
            continue
        for kk in {k, ckpt_name(k)}:
            for name, t in expand(kk, v):
                sd[name] = t
    idx = json.load(open(base / "model.safetensors.index.json"))
    by_shard = defaultdict(list)
    for k, f in idx["weight_map"].items():
        by_shard[f].append(k)
    copy_aux(base, out)
    shutil.copy(base / "model.safetensors.index.json", out / "model.safetensors.index.json")
    changed_total = 0
    for f, keys in sorted(by_shard.items()):
        tensors, changed = {}, 0
        with safe_open(base / f, "pt", device="cpu") as fh:
            meta = fh.metadata()
            for k in keys:
                t = fh.get_tensor(k)
                m = sd.get(k)
                if m is not None and m.shape == t.shape and not torch.equal(m.to(t.dtype).cpu(), t):
                    t = m.to(t.dtype).cpu().contiguous()
                    changed += 1
                tensors[k] = t
        if changed:
            save_file(tensors, out / f, metadata=meta or {"format": "pt"})
        else:
            link_or_copy(base / f, out / f)
        changed_total += changed
        print(f"{f}: {changed}/{len(keys)} changed", flush=True)
    print("changed tensors:", changed_total, "->", out)


def quant_block(w: torch.Tensor, block: int = 128):
    """bf16 [out, in] -> (float8_e4m3fn [out, in], float32 scale_inv [ceil(out/b), ceil(in/b)])."""
    w = w.float().cuda()
    o, i = w.shape
    po, pi = (-o) % block, (-i) % block
    wp = torch.nn.functional.pad(w, (0, pi, 0, po))
    blocks = wp.view(wp.shape[0] // block, block, wp.shape[1] // block, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / 448.0
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q = q.view(wp.shape)[:o, :i].contiguous()
    return q.cpu(), scale.float().cpu()


def merge_fp8(a):
    bf, mg, fp, out = Path(a.base), Path(a.merged), Path(a.fp8), Path(a.out)
    # which bf16 tensors changed?
    bidx = json.load(open(bf / "model.safetensors.index.json"))["weight_map"]
    midx = json.load(open(mg / "model.safetensors.index.json"))["weight_map"]
    changed = {}
    for k, f in midx.items():
        if (mg / f).samefile(bf / bidx[k]) if (mg / f).exists() and (bf / bidx[k]).exists() else False:
            continue
        with safe_open(mg / f, "pt", device="cpu") as h1, safe_open(bf / bidx[k], "pt", device="cpu") as h2:
            t1, t2 = h1.get_tensor(k), h2.get_tensor(k)
            if not torch.equal(t1, t2):
                changed[k] = t1
    print("changed bf16 tensors:", len(changed), flush=True)
    # map to FP8 tensor names (FP8 checkpoint: experts per expert)
    fnames = set(json.load(open(fp / "model.safetensors.index.json"))["weight_map"])
    target = {}
    for k, t in changed.items():
        for name, tt in expand(k, t):
            if name in fnames:
                target[name] = tt
    fidx = json.load(open(fp / "model.safetensors.index.json"))
    by_shard = defaultdict(list)
    for k, f in fidx["weight_map"].items():
        by_shard[f].append(k)
    lost = [k for k in changed if not any(n in target for n, _ in expand(k, changed[k]))]
    if lost:
        raise SystemExit(f"{len(lost)} changed tensors have no FP8 counterpart, e.g. {lost[:3]}")
    copy_aux(fp, out)
    shutil.copy(fp / "model.safetensors.index.json", out / "model.safetensors.index.json")
    n = 0
    for f, keys in sorted(by_shard.items()):
        if not any(k in target for k in keys):
            link_or_copy(fp / f, out / f)
            continue
        tensors = {}
        with safe_open(fp / f, "pt", device="cpu") as fh:
            meta = fh.metadata()
            for k in keys:
                tensors[k] = fh.get_tensor(k)
        for k in keys:
            if k not in target:
                continue
            new = target[k]
            sk = k + "_scale_inv"
            if sk in fidx["weight_map"]:
                assert fidx["weight_map"][sk] == f, f"scale of {k} in another shard"
                q, s = quant_block(new)
                assert q.shape == tensors[k].shape and s.shape == tensors[sk].shape, (k, q.shape, s.shape)
                tensors[k], tensors[sk] = q, s.to(tensors[sk].dtype)
            else:
                tensors[k] = new.to(tensors[k].dtype)
            n += 1
        save_file(tensors, out / f, metadata=meta or {"format": "pt"})
        print(f"{f}: rewritten", flush=True)
    print("re-quantised / replaced tensors:", n, "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bf16", "fp8"])
    ap.add_argument("--base", required=True, help="bf16 checkpoint dir")
    ap.add_argument("--adapter")
    ap.add_argument("--merged", help="(fp8) merged bf16 dir")
    ap.add_argument("--fp8", help="(fp8) FP8 checkpoint dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge_bf16(a) if a.mode == "bf16" else merge_fp8(a)


if __name__ == "__main__":
    main()
