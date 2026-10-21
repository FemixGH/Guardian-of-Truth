#!/usr/bin/env bash
# Local dev server: Qwen3.6-35B-A3B GGUF on a 16 GB GPU with MoE experts partially on CPU.
# usage: experiments/serve_local.sh [N_CPU_MOE] [CTX] [UBATCH]
NCPU=${1:-24}
CTX=${2:-98304}
UB=${3:-1024}
MODEL=${MODEL:-$HOME/ai/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
LOG=$HOME/ai/models/server.log
pkill -x llama-server 2>/dev/null
sleep 2
cd "$HOME/ai/llama.cpp" || exit 1
LD_LIBRARY_PATH=$HOME/ai/cuda13/lib64 nohup build/bin/llama-server -m "$MODEL" -ngl 99 --n-cpu-moe "$NCPU" \
  -c "$CTX" -np 1 --jinja --reasoning-format deepseek -fa on --host 127.0.0.1 --port 8090 -t "${THREADS:-6}" \
  -b 2048 -ub "$UB" > "$LOG" 2>&1 &
for _ in $(seq 1 200); do
  sleep 3
  if curl -s 127.0.0.1:8090/health | grep -q '"ok"'; then echo READY; break; fi
  if ! pgrep -x llama-server > /dev/null; then echo DIED; break; fi
done
grep -E "model buffer size|KV buffer|compute buffer|cudaMalloc failed" "$LOG" | tail -8
nvidia-smi --query-gpu=memory.used --format=csv,noheader
