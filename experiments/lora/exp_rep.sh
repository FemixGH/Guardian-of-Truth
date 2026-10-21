#!/bin/bash
# Repeat runs to separate s11f from s11r beyond vLLM run-to-run noise: s11r on APB (2nd base APB run), s11f on APB
# (2nd), s11f on t431 (2nd). Runs after exp_s11h.sh on a fresh plain FP8 server; stops itself at 18:50 MSK so the
# watchdog's final pull at 19:00 gets complete files. Outputs: /workspace/g/lora/eval/{s11r_apb2,s11f_apb2,s11f_t431b}.*
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; PY=/workspace/venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q "S11H DONE" $O/exp.log 2>/dev/null; do sleep 15; done
sleep 10
log "REP START"
VENV=/workspace/venv LOG=$O/vllm6.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/exp.log \
  || { log "rep serve FAILED"; exit 1; }
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet
STOP_AT=$(TZ=Europe/Moscow date -d "today 18:50" +%s)
run() {  # code_dir name data
  local t0=$(date +%s) left=$(( STOP_AT - $(date +%s) ))
  [ $left -lt 300 ] && { log "$2 skipped (no time)"; return; }
  (cd $1 && timeout $left $PY scripts/predict.py --input $3 --output $O/$2.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/$2.jsonl > $O/$2.log 2>&1)
  log "$2 exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$2.log | tail -1)"
}
run $R/runs/calib/subs/s11r s11r_apb2 $APB
run $R/dist/s11f s11f_apb2 $APB
run $R/dist/s11f s11f_t431b $T431
kill $(cat /tmp/vllm_8000.pid)
log "REP DONE"
