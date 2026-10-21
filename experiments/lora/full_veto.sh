#!/bin/bash
# End-to-end predict.py: candidate 12 + veto (v3j) + 160K context, A100 numerics (Marlin), 431-row set.
cd /workspace/g/guardian
source /workspace/venv/bin/activate
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
export GUARDIAN_VLLM_EXTRA="--moe-backend marlin"
O=/workspace/g/lora/full; mkdir -p $O
t0=$(date +%s)
python scripts/predict.py --input runs/timing_431_labeled.parquet --output $O/veto.csv \
  --model-path /workspace/g/models/qwen36_fp8 --calibration experiments/cal_cand12_veto.json --dump $O/veto.jsonl > $O/veto.log 2>&1
echo "veto exit=$? wall=$(( $(date +%s) - t0 ))s" >> $O/summary.txt
echo DONE >> $O/summary.txt
