#!/bin/bash
# Do-Before-You-Judge second pass: s11j (= s11h with the v2 second pass replaced by judge_prompt_dbyj.md, expected turn
# first, then diff) on t431 and APB-ru, after exp_recall3.sh. Outputs: /workspace/g/a100/s11j_{t431,apb}.*
G=/workspace/g; O=$G/a100; R=$G/guardian; PY=/workspace/venv/bin/python
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/a100.log; }
until grep -q "RECALL3 DONE" $O/a100.log 2>/dev/null; do sleep 15; done
sleep 10; log "DBYJ START"
VENV=/workspace/venv LOG=$O/vllm_dbyj.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/a100.log \
  || { log "dbyj serve FAILED"; exit 1; }
for set in t431 apb; do
  data=$R/runs/timing_431_labeled.parquet; [ $set = apb ] && data=$R/runs/lora/apb_eval_ru.parquet; t0=$(date +%s)
  (cd $R/dist/s11j && $PY scripts/predict.py --input $data --output $O/s11j_$set.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/s11j_$set.jsonl > $O/s11j_$set.log 2>&1)
  log "s11j $set exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/s11j_$set.log | tail -1) "
done
kill $(cat /tmp/vllm_8000.pid)
log "DBYJ DONE"
