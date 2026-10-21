#!/bin/bash
# Recall experiment on the A100 after exp_a100.sh: s11r2 (= s11h + text-reply "textmiss" specialist on unflagged text
# replies) on t431 and APB-ru against an external vLLM server (native A100 FP8). The specialist's additions are
# isolated from the dump (pass index of the textmiss specialist). Outputs: /workspace/g/a100/s11r2_{t431,apb}.*
G=/workspace/g; O=$G/a100; R=$G/guardian; PY=/workspace/venv/bin/python
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/a100.log; }
until grep -q "A100 DONE" $O/a100.log 2>/dev/null; do sleep 15; done
sleep 10; log "RECALL3 START"
VENV=/workspace/venv LOG=$O/vllm_r3.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/a100.log \
  || { log "recall3 serve FAILED"; exit 1; }
for set in t431 apb; do
  data=$R/runs/timing_431_labeled.parquet; [ $set = apb ] && data=$R/runs/lora/apb_eval_ru.parquet; t0=$(date +%s)
  (cd $R/dist/s11r2 && $PY scripts/predict.py --input $data --output $O/s11r2_$set.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/s11r2_$set.jsonl > $O/s11r2_$set.log 2>&1)
  log "s11r2 $set exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/s11r2_$set.log | tail -1) $(grep -o "specialist 'textmiss': [0-9]* rows" $O/s11r2_$set.log)"
done
kill $(cat /tmp/vllm_8000.pid)
log "RECALL3 DONE"
