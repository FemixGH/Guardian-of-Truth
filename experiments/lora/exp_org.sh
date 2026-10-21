#!/bin/bash
# Organisers'-definition lever: s11o1 (organisers' prompt, thinking, as the only judge prompt + rules + numbers + re-check)
# on GPU0, s11o2 (v3m + organisers' prompt instead of v2) on GPU1; t431 then APB. Qwen FP8, A100 numerics (Marlin).
G=/workspace/g; O=$G/lora/eval; PY=/workspace/venv/bin/python; mkdir -p $O
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -qs DLDONE $G/dl_qwen36_fp8.log && $PY -c "import vllm" 2>/dev/null; do sleep 15; done
log "ORG START (vllm $($PY -c 'import vllm;print(vllm.__version__)' 2>/dev/null))"
srv() { CUDA_VISIBLE_DEVICES=$1 setsid $PY -m vllm.entrypoints.openai.api_server --model $G/models/qwen36_fp8 --served-model-name judge \
  --port $2 --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 96 --enable-prefix-caching --language-model-only \
  --reasoning-parser qwen3 --moe-backend marlin > $O/srv_$1.log 2>&1 < /dev/null &
  for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q judge && { log "qwen GPU$1 ready"; return 0; }; sleep 10; done; log "qwen GPU$1 FAILED"; return 1; }
run() {  # port variant tag data
  local t0=$(date +%s)
  (cd $G/guardian/dist/$2 && $PY scripts/predict.py --input $4 --output $O/$2_$3.csv --base-url http://127.0.0.1:$1/v1 \
     --served-name judge --dump $O/$2_$3.jsonl > $O/$2_$3.log 2>&1)
  log "$2_$3 done $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$2_$3.log | tail -1)"
}
T431=$G/guardian/runs/timing_431_labeled.parquet; APB=$G/guardian/runs/lora/apb_eval_ru.parquet
( srv 0 8010 && { run 8010 s11o1 t431 $T431; run 8010 s11o1 apb $APB; } ) &
( srv 1 8011 && { run 8011 s11o2 t431 $T431; run 8011 s11o2 apb $APB; } ) &
wait; log "ORG DONE"
