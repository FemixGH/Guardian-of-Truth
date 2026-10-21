#!/bin/bash
# Python environment for the judge fine-tuning (same on the local test machine and on the rented GPU).
#   bash setup_env.sh <venv_dir>
# vLLM 0.31.0 (as in the submission image), transformers >= 5.2 (qwen3_5_moe config; older versions silently build
# a wrong config), peft (LoRA incl. MoE expert parameters), flash-linear-attention (fast Gated DeltaNet in training;
# without it transformers falls back to a ~2x slower torch path with no error).
set -euo pipefail
V=${1:-/workspace/venv}
python3 -m venv "$V" 2>/dev/null || { apt-get install -y -qq python3-venv >/dev/null && python3 -m venv "$V"; }
"$V/bin/pip" install -q -U pip
"$V/bin/pip" install -q "vllm==0.31.0"
"$V/bin/pip" install -q "transformers>=5.2" "peft>=0.21" pandas pyarrow huggingface_hub hf_transfer openai
"$V/bin/pip" install -q flash-linear-attention || echo "WARN: flash-linear-attention not installed"
"$V/bin/pip" install -q --no-build-isolation causal-conv1d || echo "WARN: causal-conv1d not installed (training conv falls back to torch)"
"$V/bin/python" - <<'E'
import torch, vllm, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__, "| transformers",
      transformers.__version__, "| peft", peft.__version__)
try:
    import fla  # noqa: F401
    print("flash-linear-attention OK")
except Exception as e:  # noqa: BLE001
    print("flash-linear-attention MISSING:", e)
from packaging.version import Version
assert Version(transformers.__version__) >= Version("5.2"), "transformers too old for qwen3_5_moe"
E
