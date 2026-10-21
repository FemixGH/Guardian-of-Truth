#!/bin/bash
# s11t6 = s11v with the main pass (v3m) sampled at temperature 0.6 (Qwen's thinking-mode setting) instead of greedy.
# One H100: external Qwen FP8 server with A100 numerics (Marlin), t431 x2, then APB.
G=/workspace/g; O=$G/lora/eval; PY=/workspace/venv/bin/python; mkdir -p $O
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -qs DLDONE $G/dl_qwen36_fp8.log && $PY -c "import vllm" 2>/dev/null; do sleep 15; done
log "T6 START"
T431=$G/guardian/runs/timing_431_labeled.parquet; APB=$G/guardian/runs/lora/apb_eval_ru.parquet
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
setsid $PY -m vllm.entrypoints.openai.api_server --model $G/models/qwen36_fp8 --served-model-name judge \
  --port 8011 --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 96 --enable-prefix-caching --language-model-only \
  --reasoning-parser qwen3 --moe-backend marlin > $O/srv.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:8011/v1/models | grep -q judge && break; sleep 10; done
log "server ready"
run() { local t0=$(date +%s); (cd $G/guardian/dist/$1 && $PY scripts/predict.py --input $3 --output $O/$1_$2.csv \
    --base-url http://127.0.0.1:8011/v1 --served-name judge --budget-sec 3600 --dump $O/$1_$2.jsonl > $O/$1_$2.log 2>&1)
  log "$1_$2 done $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$1_$2.log | tail -1)"; }
run s11t6 t431_r1 $T431; run s11t6 t431_r2 $T431; run s11t6 apb $APB
log "T6 DONE"
