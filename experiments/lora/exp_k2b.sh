#!/bin/bash
# Re-prioritised queue (replaces exp_k2glm.sh after K2 BF16 run 1): the deciding question is whether the deployable
# K2 AWQ build keeps BF16 quality with the OFFICIAL settings. GPU0 = K2 AWQ (t431, APB, timing), GPU1 = GLM-4.7-Flash.
G=/workspace/g; R=$G/guardian; M=$G/models; O=$G/lora/eval; PY=/workspace/venv/bin/python
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet
# let K2 BF16 run 1 finish its stragglers (max 12 min), then free both GPUs
for i in $(seq 1 72); do pgrep -f "[r]un_judge.py.*k2b_v3f_t431_r1" > /dev/null || break; sleep 10; done
for p in $(pgrep -f "[r]un_judge.py"); do kill $p; done
for p in $(pgrep -f "[v]llm.entrypoints.openai.api_server"); do kill $p; done; sleep 30
log "k2b_v3f_t431_r1 closed with $(wc -l < $O/k2b_v3f_t431_r1.jsonl) rows; K2 BF16 server stopped"

serve() {  # name port gpus extra-args...
  local name=$1 port=$2 gpus=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpus setsid $PY -m vllm.entrypoints.openai.api_server --served-model-name m --port $port \
    --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching "$@" \
    > $O/srv_$name.log 2>&1 < /dev/null &
  echo $! > /tmp/srv_$port.pid
  for i in $(seq 1 90); do
    curl -s -m 5 http://127.0.0.1:$port/v1/models | grep -q '"m"' && { log "$name READY on :$port"; return 0; }
    kill -0 $(cat /tmp/srv_$port.pid) 2>/dev/null || { log "$name DIED (see srv_$name.log)"; return 1; }
    sleep 10
  done
  log "$name serve TIMEOUT"; return 1
}
stop() { kill $(cat /tmp/srv_$1.pid) 2>/dev/null; sleep 20; }
judge() {  # port out data prompt extra-args...
  local port=$1 out=$2 data=$3 prompt=$4; shift 4; local t0=$(date +%s)
  (cd $R && $PY experiments/run_judge.py --data $data --out $O/$out.jsonl --base-url http://127.0.0.1:$port/v1 --model m \
     --prompt src/guardian/$prompt --max-ctx-chars 76000 --layout head --concurrency 64 "$@" > $O/$out.log 2>&1)
  log "$out done $(( $(date +%s) - t0 ))s rows=$(wc -l < $O/$out.jsonl)"
}
K2REQ=(--temperature 1.0 --max-tokens 32768 --force-mode prefill --extra '{"chat_template_kwargs":{"reasoning_effort":"high"},"top_p":0.95}')
GLMREQ=(--temperature 1.0 --max-tokens 16384 --extra '{"top_p":0.95}')
K2SRV=(--trust-remote-code --chat-template-content-format string --reasoning-parser k2_horizon)
(
  [ -f $M/k2awq/.fixed ] || { $PY $R/experiments/k2h_fix_vexperts.py $M/k2awq > $O/k2awq_fix.log 2>&1 && touch $M/k2awq/.fixed; }
  log "k2awq fix: $(tail -1 $O/k2awq_fix.log 2>/dev/null)"
  if serve k2awq 8001 0 --model $M/k2awq "${K2SRV[@]}"; then
    judge 8001 k2a_v3f_t431 $T431 judge_prompt_v3f.md "${K2REQ[@]}"
    judge 8001 k2a_v3f_apb $APB judge_prompt_v3f.md "${K2REQ[@]}"
    judge 8001 k2a_v3f_t431_stop $T431 judge_prompt_v3f.md "${K2REQ[@]}" --stop-verdict
  fi
  stop 8001; log "GPU0 DONE"
) &
(
  if serve glm47f 8002 1 --model $M/glm47f --reasoning-parser glm45; then
    judge 8002 glm_v3f_t431 $T431 judge_prompt_v3f.md "${GLMREQ[@]}"
    judge 8002 glm_org_t431 $T431 judge_prompt_org.md --no-notes "${GLMREQ[@]}"
    judge 8002 glm_v3f_apb $APB judge_prompt_v3f.md "${GLMREQ[@]}"
  fi
  stop 8002; log "GPU1 DONE"
) &
wait
log "EXP DONE"
