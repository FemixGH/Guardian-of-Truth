#!/bin/bash
# Hypothesis "s11r + top text turns by the organisers' baseline prompt (P_base)": score P_base (exact and tail layout)
# on t431 / APB-ru / valid and run s11r end-to-end on APB-ru, A100 numerics (Marlin W8A16).
# Outputs: /workspace/g/pbase/*.{jsonl,csv,log}
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; PY=/workspace/venv/bin/python
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q BOOTSTRAP_DONE $G/bootstrap.log 2>/dev/null; do sleep 30; done
log "START"
VENV=/workspace/venv LOG=$O/vllm.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/exp.log || { log "serve FAILED"; exit 1; }
grep -E "Selected .*Fp8LinearMethod|MoE backend" $O/vllm.log | head -2 >> $O/exp.log
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet; VAL=$R/runs/valid.parquet
for set in t431 apb valid; do
  case $set in t431) D=$T431;; apb) D=$APB;; valid) D=$VAL;; esac
  for lay in exact tail; do
    t0=$(date +%s); $PY $R/experiments/pbase_score.py --data $D --out $O/pbase_${lay}_${set}.jsonl --layout $lay > /dev/null 2>> $O/exp.log
    log "pbase $lay $set $(( $(date +%s) - t0 ))s rows=$(wc -l < $O/pbase_${lay}_${set}.jsonl)"
  done
done
t0=$(date +%s)
(cd $R/runs/calib/subs/s11r && $PY scripts/predict.py --input $APB --output $O/s11r_apb.csv --base-url http://127.0.0.1:8000/v1 \
   --served-name judge --dump $O/s11r_apb.jsonl > $O/s11r_apb.log 2>&1)
log "s11r apb exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/s11r_apb.log | tail -1)"
kill $(cat /tmp/vllm_8000.pid)
log "DONE"
