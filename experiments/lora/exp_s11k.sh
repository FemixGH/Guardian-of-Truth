#!/bin/bash
# s11k = s11h + exchange_false_refusal hard rule (the only difference). End-to-end check on a fresh plain FP8 server
# with A100 numerics (Marlin). Outputs: /workspace/g/lora/eval/s11k_{t431,apb}.{csv,jsonl,log}
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; PY=/workspace/venv/bin/python
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q BOOTSTRAP_DONE $G/bootstrap.log 2>/dev/null; do sleep 20; done
log "S11K START"
VENV=/workspace/venv LOG=$O/vllm.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/exp.log \
  || { log "s11k serve FAILED"; exit 1; }
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet
run() {  # code_dir name set
  local data=$T431; [ $3 = apb ] && data=$APB; local t0=$(date +%s)
  (cd $1 && $PY scripts/predict.py --input $data --output $O/$2_$3.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/$2_$3.jsonl > $O/$2_$3.log 2>&1)
  log "$2 $3 exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$2_$3.log | tail -1)"
}
run $R/dist/s11k s11k t431
run $R/dist/s11k s11k apb
kill $(cat /tmp/vllm_8000.pid)
log "S11K DONE"
