#!/bin/bash
# Ideas test on the A100 after the s11p platform timing: stop the timing queue (s11q timing is not needed), then on one
# vLLM server run s11q (baseline, same machine) and s11t (= s11q + v3k carry-over rule + textproc specialist on text-only
# replies) on t431 and APB-ru. Outputs: /workspace/g/a100/{s11q,s11t}_{t431,apb}.{csv,jsonl,log}
G=/workspace/g; O=$G/a100; R=$G/guardian; PY=/workspace/venv/bin/python
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/a100.log; }
until grep -q "s11p exit=" $O/a100.log 2>/dev/null; do sleep 5; done
for pid in $(pgrep -f "experiments/lora/[e]xp_a100.sh"); do kill $pid; done
sleep 25; pkill -f "vllm.entrypoints" ; pkill -f "scripts/[p]redict.py"; sleep 15
log "IDEAS START (timing queue stopped after s11p)"
VENV=/workspace/venv LOG=$O/vllm_ideas.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/a100.log \
  || { log "ideas serve FAILED"; exit 1; }
for c in s11q s11t; do
  for set in t431 apb; do
    data=$R/runs/timing_431_labeled.parquet; [ $set = apb ] && data=$R/runs/lora/apb_eval_ru.parquet; t0=$(date +%s)
    (cd $R/dist/$c && $PY scripts/predict.py --input $data --output $O/${c}_$set.csv --base-url http://127.0.0.1:8000/v1 \
       --served-name judge --dump $O/${c}_$set.jsonl > $O/${c}_$set.log 2>&1)
    log "$c $set exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/${c}_$set.log | tail -1)"
  done
done
kill $(cat /tmp/vllm_8000.pid)
log "IDEAS DONE"
