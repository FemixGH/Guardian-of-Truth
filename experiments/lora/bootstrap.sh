#!/bin/bash
# First thing on a fresh GPU server (NVIDIA CUDA template). Idempotent.
#   bash bootstrap.sh            -> checks, venv (setup_env.sh), model downloads in parallel
# Needs: driver with CUDA >= 13.0 (torch cu130 wheels of vLLM 0.31), nvcc (vLLM JIT pieces), ~500 GB disk.
set -u
G=/workspace/g; M=$G/models; V=/workspace/venv
mkdir -p "$M" "$G/lora"
echo "== GPU / driver"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CUDA_DRV=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}'); echo "driver CUDA $CUDA_DRV"
python3 - "$CUDA_DRV" <<'E' || { echo "FATAL: driver CUDA < 13.0 - pick another server"; exit 1; }
import sys; a = (tuple(int(x) for x in sys.argv[1].split(".") if x.isdigit()) + (0, 0))[:2]; sys.exit(0 if a >= (13, 0) else 1)
E
echo "== nvcc"; (command -v nvcc || ls /usr/local/cuda/bin/nvcc) 2>/dev/null || echo "WARN: nvcc not found (vLLM may need it)"
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
echo "== disk"; df -h /workspace | tail -1
# venv with the downloader first (system pip may be "externally managed"), downloads in background,
# then the full environment is installed into the same venv while the models download
[ -x "$V/bin/python" ] || python3 -m venv "$V" || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv "$V"; }
"$V/bin/pip" install -q -U pip huggingface_hub hf_transfer
dl() { HF_HUB_ENABLE_HF_TRANSFER=1 "$V/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('$1', local_dir='$M/$2')
print('DLDONE')" > "$G/dl_$2.log" 2>&1; }
MODELS=${MODELS:-"qwen36_bf16 qwen36_fp8 tpro"}  # e.g. MODELS=qwen36_fp8 for inference-only work
declare -A REPO=([qwen36_bf16]=Qwen/Qwen3.6-35B-A3B [qwen36_fp8]=Qwen/Qwen3.6-35B-A3B-FP8 [tpro]=t-tech/T-pro-it-2.1-FP8 [k2bf16]=IFM/K2-Horizon-MoVA-36B-A4B [k2awq]=cyankiwi/K2-Horizon-MoVA-36B-A4B-AWQ-INT4 [glm47f]=zai-org/GLM-4.7-Flash)
for m in $MODELS; do grep -qs DLDONE "$G/dl_$m.log" || dl "${REPO[$m]}" "$m" & done
[ -x "$V/bin/python" ] && "$V/bin/python" -c "import vllm, peft" 2>/dev/null || bash "$G/guardian/experiments/lora/setup_env.sh" "$V"
wait
for d in $MODELS; do printf "%-12s %s %s\n" $d "$(du -sh $M/$d 2>/dev/null | cut -f1)" "$(grep -c DLDONE $G/dl_$d.log 2>/dev/null)"; done
echo BOOTSTRAP_DONE
