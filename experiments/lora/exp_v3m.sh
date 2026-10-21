#!/bin/bash
# Recall prompt v3m (s11v = s11q with v3m instead of v3f): after the K2 experiments, Qwen FP8 (A100 numerics) on both
# GPUs; GPU0: t431 twice (noise), GPU1: APB. Outputs /workspace/g/lora/eval/s11v_{t431_r1,t431_r2,apb}.*
G=/workspace/g; O=$G/lora/eval; PY=/workspace/venv/bin/python
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
for i in $(seq 1 60); do [ $(grep -c -E "k2x_(t06|rep105) done" $O/exp.log) -ge 2 ] && break; sleep 20; done
for p in $(pgrep -f "[r]un_judge.py|[s]cripts/predict.py|[a]pi_server"); do kill $p; done; sleep 30
log "v3m: K2 servers stopped; starting Qwen FP8 on both GPUs"
srv() { CUDA_VISIBLE_DEVICES=$1 setsid $PY -m vllm.entrypoints.openai.api_server --model $G/models/qwen36_fp8 --served-model-name judge \
  --port $2 --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 96 --enable-prefix-caching --language-model-only \
  --reasoning-parser qwen3 --moe-backend marlin > $O/srv_qwen_$1.log 2>&1 < /dev/null &
  for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q judge && return 0; sleep 10; done; return 1; }
run() {  # port tag data
  local t0=$(date +%s)
  (cd $G/guardian/dist/s11v && $PY scripts/predict.py --input $3 --output $O/s11v_$2.csv --base-url http://127.0.0.1:$1/v1 \
     --served-name judge --dump $O/s11v_$2.jsonl > $O/s11v_$2.log 2>&1)
  log "s11v_$2 done $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/s11v_$2.log | tail -1)"
}
srv 0 8010 & srv 1 8011 & wait
T431=$G/guardian/runs/timing_431_labeled.parquet; APB=$G/guardian/runs/lora/apb_eval_ru.parquet
( run 8010 t431_r1 $T431; run 8010 t431_r2 $T431 ) &
( run 8011 apb $APB ) &
wait; log "V3M DONE"
