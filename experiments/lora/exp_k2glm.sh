#!/bin/bash
# Fair second-model test on 2x RTX PRO 6000 Max-Q: K2-Horizon (official BF16 setup, TP2) and GLM-4.7-Flash vs Qwen3.6
# FP8, each with two prompts: v3f (s11q's prompt + domain rules + checker notes, tuned on Qwen's errors) and the
# organisers' neutral baseline prompt (judge_prompt_org.md, no notes). Each model gets its own recommended sampling.
# Outputs: /workspace/g/lora/eval/<model>_<prompt>_<set>[_rN].{jsonl,log}; progress in /workspace/g/lora/eval/exp.log
G=/workspace/g; R=$G/guardian; M=$G/models; O=$G/lora/eval; PY=/workspace/venv/bin/python
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet
until grep -q BOOTSTRAP_DONE $G/bootstrap.log 2>/dev/null; do sleep 15; done
log "EXP START"

serve() {  # name port gpus extra-args...  -> waits until ready (max 15 min)
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
  log "$out done $(( $(date +%s) - t0 ))s $(tail -2 $O/$out.log | tr '\n' ' ' | cut -c1-200)"
}
K2REQ=(--temperature 1.0 --max-tokens 32768 --force-mode prefill --extra '{"chat_template_kwargs":{"reasoning_effort":"high"},"top_p":0.95}')
GLMREQ=(--temperature 1.0 --max-tokens 16384 --extra '{"top_p":0.95}')
QWREQ=(--temperature 0 --max-tokens 4096 --stop-verdict)
K2SRV=(--trust-remote-code --dtype bfloat16 --chat-template-content-format string --reasoning-parser k2_horizon)

# ---- Phase A: K2-Horizon BF16, official recipe (TP2 + expert parallel) on both GPUs
if serve k2bf16 8000 0,1 --model $M/k2bf16 --tensor-parallel-size 2 --enable-expert-parallel "${K2SRV[@]}"; then
  judge 8000 k2b_v3f_t431_r1 $T431 judge_prompt_v3f.md "${K2REQ[@]}"
  judge 8000 k2b_org_t431 $T431 judge_prompt_org.md --no-notes "${K2REQ[@]}"
  judge 8000 k2b_v3f_apb $APB judge_prompt_v3f.md "${K2REQ[@]}"
  judge 8000 k2b_v3f_t431_r2 $T431 judge_prompt_v3f.md "${K2REQ[@]}"
fi
stop 8000
log "PHASE A DONE"

# ---- Phase B (parallel): GPU1 = GLM-4.7-Flash; GPU0 = K2 AWQ (deployable form) then Qwen3.6 FP8 (A100 numerics)
(
  if serve glm47f 8002 1 --model $M/glm47f --reasoning-parser glm45; then
    judge 8002 glm_v3f_t431 $T431 judge_prompt_v3f.md "${GLMREQ[@]}"
    judge 8002 glm_org_t431 $T431 judge_prompt_org.md --no-notes "${GLMREQ[@]}"
    judge 8002 glm_v3f_apb $APB judge_prompt_v3f.md "${GLMREQ[@]}"
  fi
  stop 8002; log "GLM DONE"
) &
(
  [ -f $M/k2awq/.fixed ] || { $PY $R/experiments/k2h_fix_vexperts.py $M/k2awq > $O/k2awq_fix.log 2>&1 && touch $M/k2awq/.fixed; }
  if serve k2awq 8001 0 --model $M/k2awq "${K2SRV[@]}"; then
    judge 8001 k2a_v3f_t431 $T431 judge_prompt_v3f.md "${K2REQ[@]}"
  fi
  stop 8001
  export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
  if serve qwen 8001 0 --model $M/qwen36_fp8 --reasoning-parser qwen3 --language-model-only --moe-backend marlin; then
    judge 8001 qwen_v3f_t431 $T431 judge_prompt_v3f.md "${QWREQ[@]}"
    judge 8001 qwen_org_t431 $T431 judge_prompt_org.md --no-notes "${QWREQ[@]}"
  fi
  stop 8001; log "GPU0 DONE"
) &
wait
log "EXP DONE"
