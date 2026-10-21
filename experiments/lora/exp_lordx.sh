#!/bin/bash
# Replace the queued "opus" (hesamation 4.6 distill) in exp_models.sh by lordx64's Claude-4.7-Opus distill:
# kill the opus download when it starts (exp_models.sh then logs it as failed and finishes), download lordx64 in
# parallel with the running model, then evaluate it the same way after exp_models.sh exits.
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; M=$G/models; PY=/workspace/venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
# wait until exp_models.sh has moved on to occamy (bigbang deleted) and started the opus download, then cancel it
until grep -q "occamy downloaded" $O/exp.log 2>/dev/null; do sleep 10; done
until ps -eo args | grep -q "[h]esamation"; do sleep 5; done
for p in $(ps -eo pid,args | awk '/[h]esamation/ {print $1}'); do kill $p; done; sleep 5; rm -rf $M/opus
log "opus (4.6) download cancelled; downloading lordx (Claude 4.7 Opus distill)"
HF_HUB_ENABLE_HF_TRANSFER=1 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled', local_dir='$M/lordx',
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','*.py'])
print('DLDONE')" > $G/dl_lordx.log 2>&1
grep -q DLDONE $G/dl_lordx.log || { log "lordx download FAILED"; exit 1; }
while pgrep -f exp_models.sh > /dev/null; do sleep 15; done
rm -rf $M/occamy $M/bigbang $M/empero
log "lordx downloaded ($(du -sh $M/lordx | cut -f1)), serving"
VENV=/workspace/venv LOG=$O/vllm_lordx.log GPU_MEM=0.93 bash $R/experiments/lora/serve.sh $M/lordx judge 8000 131072 >> $O/exp.log \
  || { log "lordx serve FAILED"; exit 1; }
for set in t431 apb; do
  data=$R/runs/timing_431_labeled.parquet; [ $set = apb ] && data=$R/runs/lora/apb_eval_ru.parquet; t0=$(date +%s)
  (cd $R/runs/calib/subs/s11r && $PY scripts/predict.py --input $data --output $O/lordx_$set.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/lordx_$set.jsonl > $O/lordx_$set.log 2>&1)
  log "lordx $set exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/lordx_$set.log | tail -1)"
done
kill $(cat /tmp/vllm_8000.pid)
log "ALL_DONE"
