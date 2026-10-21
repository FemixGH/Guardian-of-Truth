#!/bin/bash
# Full predict.py runs (candidate 12 vs submission-11 config) on the 431-row timing set, A100 numerics (Marlin).
cd /workspace/g/guardian
source /workspace/venv/bin/activate
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
export GUARDIAN_VLLM_EXTRA="--moe-backend marlin"
O=/workspace/g/lora/full; mkdir -p $O
for c in cand12 s11_rules; do
  t0=$(date +%s)
  python scripts/predict.py --input runs/timing_431_labeled.parquet --output $O/$c.csv \
    --model-path /workspace/g/models/qwen36_fp8 --calibration experiments/cal_$c.json --dump $O/$c.jsonl > $O/$c.log 2>&1
  echo "$c exit=$? wall=$(( $(date +%s) - t0 ))s" >> $O/summary.txt
done
echo DONE >> $O/summary.txt
