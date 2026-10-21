#!/bin/bash
# K2 speed test: after the Qwen recall-prompt runs on GPU $GPU, start 8-bit K2 under vLLM 0.19.1 with an FP8 KV cache
# ($KV) [+ $BACKEND], run s11k2 (T0.6, longest-first) on t431.   GPU=0 PORT=8006 KV=fp8 WAIT=s11v_t431_r2 TAG=fp8 bash k2fast.sh
O=/workspace/g/lora/eval; G=/workspace/g
until grep -q "$WAIT done" $O/exp.log; do sleep 15; done
QP=$([ $GPU = 0 ] && echo 8010 || echo 8011)
for p in $(pgrep -f "[p]ort $QP"); do kill $p; done; sleep 25
echo "[$(date +%H:%M:%S)] k2fast $TAG: Qwen on GPU$GPU stopped, starting K2 kv=$KV backend=${BACKEND:-auto}" >> $O/exp.log
cd /workspace/k2port && env ${BACKEND:+VLLM_ATTENTION_BACKEND=$BACKEND} GUARDIAN_K2_PORT=1 PYTHONPATH=/workspace/k2port CUDA_VISIBLE_DEVICES=$GPU \
  PATH=/workspace/v19/bin:/usr/local/cuda/bin:$PATH setsid /workspace/v19/bin/python -m vllm.entrypoints.openai.api_server \
  --model $G/models/k2int8 --served-model-name m --port $PORT --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 --enable-prefix-caching --trust-remote-code --chat-template-content-format string --kv-cache-dtype $KV \
  > /workspace/k2port/srv_fast_$TAG.log 2>&1 < /dev/null &
for i in $(seq 1 80); do curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"m"' && break; sleep 10; done
echo "[$(date +%H:%M:%S)] k2fast $TAG: server $(curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"m"' && echo READY || echo FAILED) $(grep -o 'GPU KV cache size: [0-9,]* tokens' /workspace/k2port/srv_fast_$TAG.log)" >> $O/exp.log
t0=$(date +%s)
(cd $G/guardian/dist/s11k2 && /workspace/v19/bin/python scripts/predict.py --input $G/guardian/runs/timing_431_labeled.parquet \
   --output $O/k2fast_$TAG.parquet --base-url http://127.0.0.1:$PORT/v1 --served-name m --concurrency 128 \
   --calibration /workspace/k2exp/cal_t06_lpt.json --dump $O/k2fast_$TAG.jsonl > $O/k2fast_$TAG.log 2>&1)
echo "[$(date +%H:%M:%S)] k2fast_$TAG done $(( $(date +%s) - t0 ))s" >> $O/exp.log
