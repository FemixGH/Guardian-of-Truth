#!/bin/bash
# Qwen3.6-35B-A3B fine-tunes as the judge: full s11r pipeline (predict.py) on t431 and APB-ru per model.
# Base = organisers' FP8 in A100 numerics (Marlin); fine-tunes in BF16 (they ship BF16 only). Next model is
# downloaded while the current one runs. Outputs: /workspace/g/lora/eval/<model>_<set>.{csv,jsonl,log}
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; M=$G/models; PY=/workspace/venv/bin/python
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet
declare -A REPO=([empero]=empero-ai/Qwen3.8-35B-A3B-Distill [bigbang]=endless-frontier/BigBang-v1
                 [occamy]=Accio-Lab/occamy-1.0 [opus]=hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled)
ORDER="empero bigbang occamy opus"
dl() { HF_HUB_ENABLE_HF_TRANSFER=1 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('${REPO[$1]}', local_dir='$M/$1', allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','*.py'])
print('DLDONE')" > $G/dl_$1.log 2>&1; }
run() {  # name set
  local data=$T431; [ $2 = apb ] && data=$APB; local t0=$(date +%s)
  (cd $R/runs/calib/subs/s11r && $PY scripts/predict.py --input $data --output $O/$1_$2.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/$1_$2.jsonl > $O/$1_$2.log 2>&1)
  log "$1 $2 exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$1_$2.log | tail -1)"
}
serve() {  # dir [extra]
  local d=$1; shift
  VENV=/workspace/venv LOG=$O/vllm_$(basename $d).log GPU_MEM=0.93 bash $R/experiments/lora/serve.sh $d judge 8000 131072 "$@" >> $O/exp.log
}
until grep -q BOOTSTRAP_DONE $G/bootstrap.log 2>/dev/null; do sleep 20; done
log "START"
first=${ORDER%% *}; dl $first &
serve $M/qwen36_fp8 --moe-backend marlin || { log "base serve FAILED"; exit 1; }
run base apb
kill $(cat /tmp/vllm_8000.pid); sleep 15
prev=""
for m in $ORDER; do
  wait; grep -q DLDONE $G/dl_$m.log || { log "$m download FAILED: $(tail -2 $G/dl_$m.log)"; continue; }
  [ -n "$prev" ] && rm -rf $M/$prev
  nxt=$(echo $ORDER | tr ' ' '\n' | grep -A1 -x $m | sed -n 2p); [ -n "$nxt" ] && dl $nxt &
  log "$m downloaded ($(du -sh $M/$m | cut -f1)), serving"
  if serve $M/$m; then run $m t431; run $m apb; kill $(cat /tmp/vllm_8000.pid); sleep 15
  else log "$m serve FAILED: $(grep -iE 'error' $O/vllm_$m.log | tail -2)"; fi
  prev=$m
done
wait
log "DONE"
