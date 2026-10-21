#!/bin/bash
# v3h (offers/plans count like actions) vs v3g on the same server, A100 numerics (Marlin). Waits for bootstrap.
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
export GUARDIAN_KB_FOCUS=1
W=/workspace/g/lora; PY=/workspace/venv/bin/python; L=/workspace/g/guardian/experiments/lora; F=/workspace/g/models/qwen36_fp8
mkdir -p $W/eval; cd /workspace/g/guardian
log() { echo "[$(date +%H:%M:%S)] $*" >> $W/pipeline.log; }
until grep -q BOOTSTRAP_DONE /workspace/g/bootstrap.log 2>/dev/null; do sleep 30; done
APB=runs/lora/apb_eval_ru.parquet; T431=runs/timing_431_labeled.parquet
run() {  # name prompt data
  local n=$1 p=$2 data=$3; local t0=$(date +%s)
  $PY experiments/run_judge.py --data $data --out $W/eval/${n}.jsonl --base-url http://127.0.0.1:8000/v1 --model qwen \
    --prompt src/guardian/$p --max-ctx-chars 76000 --layout head --concurrency 96 --max-tokens 4096 --stop-verdict > $W/eval/${n}.log 2>&1
  log "$n took $(( $(date +%s) - t0 ))s ($(wc -l < $W/eval/${n}.jsonl) rows)"
}
log "START v3h"
VENV=/workspace/venv LOG=$W/vllm.log bash $L/serve.sh $F qwen 8000 40960 --moe-backend marlin >> $W/pipeline.log || { log "serve FAILED"; exit 1; }
grep -E "Selected .*Fp8LinearMethod|MoE backend" $W/vllm.log | head -2 >> $W/pipeline.log
run H_v3h_t431 judge_prompt_v3h.md $T431
run H_v3h_apb judge_prompt_v3h.md $APB
run G_v3g_t431 judge_prompt_v3g.md $T431
run G_v3g_apb judge_prompt_v3g.md $APB
kill $(cat /tmp/vllm_8000.pid)
log "DONE v3h"
