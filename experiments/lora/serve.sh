#!/bin/bash
# Start an OpenAI-compatible vLLM server for a Qwen3.5/3.6 checkpoint and wait until it is ready.
#   bash serve.sh <model_dir> <served_name> [port] [max_model_len] [extra vllm args...]
# Writes the PID to /tmp/vllm_<port>.pid; stop with: kill $(cat /tmp/vllm_<port>.pid)
# --moe-backend triton: bf16 MoE without JIT-compiled FlashInfer kernels; FlashInfer sampler off for the same reason.
set -u
M=$1; NAME=$2; PORT=${3:-8000}; LEN=${4:-131072}; shift 4 2>/dev/null || shift $#
V=${VENV:-/workspace/venv}
LOG=${LOG:-/tmp/vllm_${PORT}.log}
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$V/bin:$PATH
MOE=(); MM=(); RP=(--reasoning-parser "${REASONING_PARSER:-qwen3}")
grep -q '"num_experts"' "$M/config.json" && MOE=(--moe-backend triton)
grep -q '"vision_config"' "$M/config.json" && MM=(--language-model-only)
[ "${REASONING_PARSER:-qwen3}" = none ] && RP=()
setsid "$V/bin/python" -m vllm.entrypoints.openai.api_server --model "$M" --served-model-name "$NAME" --port "$PORT" \
  --max-model-len "$LEN" --gpu-memory-utilization "${GPU_MEM:-0.90}" --max-num-seqs "${MAX_SEQS:-96}" \
  --enable-prefix-caching "${MM[@]}" "${RP[@]}" "${MOE[@]}" "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > /tmp/vllm_${PORT}.pid
for _ in $(seq 1 240); do
  sleep 5
  if curl -s "127.0.0.1:$PORT/v1/models" | grep -q '"object"'; then echo "READY $NAME on :$PORT"; exit 0; fi
  if ! kill -0 "$(cat /tmp/vllm_${PORT}.pid)" 2>/dev/null; then echo "DIED, see $LOG"; tail -20 "$LOG"; exit 1; fi
done
echo "TIMEOUT, see $LOG"; exit 1
