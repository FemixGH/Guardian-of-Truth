#!/bin/bash
# s11u = s11q with the main judge prompt v3l (lists every PROBLEM, then VERDICT: 1 if the list is non-empty; FINAL
# paper "binarized" format) instead of v3f's verdict-only output. A100. Order: s11u t431, s11u apb, then a second s11q
# t431 draw (baseline noise; the first s11q draw is runs/a100b). Outputs: /workspace/g/a100/{s11u,s11q2}_*.{csv,jsonl,log}
G=/workspace/g; O=$G/a100; R=$G/guardian; PY=/workspace/venv/bin/python
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/a100.log; }
until grep -q "BOOTSTRAP_DONE" $G/bootstrap.log 2>/dev/null; do grep -q FATAL $G/bootstrap.log && { log "bootstrap FATAL"; exit 1; }; sleep 15; done
log "S11U START"
VENV=/workspace/venv LOG=$O/vllm_u.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/a100.log \
  || { log "s11u serve FAILED"; exit 1; }
run() {  # code_dir name set
  local data=$R/runs/timing_431_labeled.parquet; [ $3 = apb ] && data=$R/runs/lora/apb_eval_ru.parquet; local t0=$(date +%s)
  (cd $R/dist/$1 && $PY scripts/predict.py --input $data --output $O/$2_$3.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/$2_$3.jsonl > $O/$2_$3.log 2>&1)
  log "$2 $3 exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$2_$3.log | tail -1)"
}
run s11u s11u t431
run s11u s11u apb
run s11q s11q2 t431
kill $(cat /tmp/vllm_8000.pid)
log "S11U DONE"
