"""Shared pieces of the judge fine-tuning pipeline (rollouts with vLLM, SFT / GSPO with transformers + PEFT).

The training prompt is built EXACTLY as at inference in the submission (scripts/predict.py, first pass):
judge_prompt_v3f.md with domain rules, checker notes from the deterministic layer, context fitted to 76000 chars,
layout "head", thinking enabled.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from guardian.features import flags  # noqa: E402
from guardian.judge import build_messages  # noqa: E402
from guardian.parse import parse_example  # noqa: E402

PROMPT = ROOT / "src/guardian/judge_prompt_v3f.md"
MAX_CTX_CHARS = 76000
VERDICT_RE = re.compile(r"VERDICT\s*[:：]\s*\**\s*([01])")

# LoRA targets for Qwen3.6-35B-A3B (qwen3_5_moe). Router (mlp.gate), shared_expert_gate, norms, conv1d untouched.
LORA_TARGET_MODULES = (r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
                       r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|"
                       r"mlp\.shared_expert\.(gate_proj|up_proj|down_proj))")
LORA_TARGET_PARAMETERS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def judge_messages(prompt: str, response: str) -> list[dict]:
    notes = flags(parse_example(prompt, response))[1]
    return build_messages(prompt, response, notes, "think", MAX_CTX_CHARS, PROMPT, "head")


def chat_text(tok, prompt: str, response: str) -> str:
    return tok.apply_chat_template(judge_messages(prompt, response), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)


def verdict_of(completion: str) -> int | None:
    """Verdict from the final answer (after the thinking block); falls back to the last VERDICT anywhere."""
    tail = completion.split("</think>")[-1]
    m = VERDICT_RE.findall(tail) or VERDICT_RE.findall(completion)
    return int(m[-1]) if m else None


def reward(verdict: int | None, label: int, truncated: bool) -> float:
    if verdict is None or truncated:
        return 0.0
    return 1.0 if verdict == label else 0.0


def load_policy(base: str, adapter: str | None = None, trainable: bool = True, rank: int = 32, alpha: int = 64,
                expert_rank: int = 4, experts: bool = True, device: str = "cuda"):
    """Base model (bf16, text-only) + LoRA. experts_implementation="grouped_mm" is ~10x faster than the loop."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers import AutoConfig
    cfg_ = AutoConfig.from_pretrained(base)
    moe = "num_experts" in json.dumps(cfg_.to_dict())
    kw = {"experts_implementation": "grouped_mm"} if moe else {}
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map=device, **kw)
    m.config.use_cache = False
    if experts and not any(".mlp.experts." in n for n, _ in m.named_parameters()):
        experts = False  # dense model (local tests with Qwen3.5-0.8B)
    if adapter:
        pm = PeftModel.from_pretrained(m, adapter, is_trainable=trainable)
    else:
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=LORA_TARGET_MODULES,
                         target_parameters=LORA_TARGET_PARAMETERS if experts else None,
                         rank_pattern={"gate_up_proj": expert_rank, "down_proj": expert_rank} if experts else {})
        pm = get_peft_model(m, cfg)
    if trainable:
        pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        pm.enable_input_require_grads()
        pm.train()  # transformers applies gradient checkpointing only in training mode (all dropouts are 0 here)
    else:
        pm.eval()
    return pm


def completion_logprobs(model, prompt_ids: list[int], completion_ids: list[int], chunk: int = 1024):
    """Per-token log p(completion_t | prefix) under `model` (PEFT-wrapped CausalLM), computed from the last hidden
    states with the LM head applied in chunks (vocab 248k: full logits would not fit)."""
    import torch
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids).last_hidden_state[0]  # [T, H]
    start = len(prompt_ids) - 1  # hidden at position t predicts token t+1
    h = hidden[start:start + len(completion_ids)]
    tgt = ids[0, len(prompt_ids):]
    from torch.utils.checkpoint import checkpoint

    def chunk_lp(hc, tc):
        return torch.log_softmax(base.lm_head(hc).float(), -1).gather(-1, tc[:, None])[:, 0]

    out = []
    for i in range(0, h.shape[0], chunk):
        hc, tc = h[i:i + chunk], tgt[i:i + chunk]
        # with gradients: recompute the [chunk x 248k] logits in backward instead of keeping them (GBs per sample)
        out.append(checkpoint(chunk_lp, hc, tc, use_reentrant=False) if torch.is_grad_enabled() else chunk_lp(hc, tc))
    return torch.cat(out)
