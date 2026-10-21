#!/bin/bash
# Levers on top of candidate 12 (v3g+stop+KB focus), Marlin numerics. Waits for full_cmp.sh to finish.
#  L1: second v3g sample at T=0.7 (OR with E2)   L2: long rows (>76k chars) with a 160k-char context cap
#  L3: specialists spec_args / spec_escalation_v2 on routed rows
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
export GUARDIAN_KB_FOCUS=1
W=/workspace/g/lora; PY=/workspace/venv/bin/python; L=/workspace/g/guardian/experiments/lora; F=/workspace/g/models/qwen36_fp8
cd /workspace/g/guardian
until grep -q DONE $W/full/summary.txt 2>/dev/null; do sleep 30; done
log() { echo "[$(date +%H:%M:%S)] $*" >> $W/levers.log; }
APB=runs/lora/apb_eval_ru.parquet; T431=runs/timing_431_labeled.parquet
run() {  # name prompt data extra...
  local n=$1 p=$2 data=$3; shift 3; local t0=$(date +%s)
  $PY experiments/run_judge.py --data $data --out $W/eval/${n}.jsonl --base-url http://127.0.0.1:8000/v1 --model qwen \
    --prompt src/guardian/$p --max-ctx-chars 76000 --layout head --concurrency 96 --max-tokens 4096 --stop-verdict "$@" > $W/eval/${n}.log 2>&1
  log "$n took $(( $(date +%s) - t0 ))s ($(wc -l < $W/eval/${n}.jsonl) rows)"
}
$PY - <<'P'
import sys, pandas as pd; sys.path.insert(0, "src")
from guardian.specialists import route
for k, p in {"apb": "runs/lora/apb_eval_ru.parquet", "t431": "runs/timing_431_labeled.parquet"}.items():
    d = pd.read_parquet(p); r = [route(a, b) for a, b in zip(d.prompt, d.response)]
    for s in ("args", "escalation"):
        d[[s in x for x in r]].to_parquet(f"/workspace/g/lora/eval/route_{s}_{k}.parquet", index=False)
    d[(d.prompt.str.len() + d.response.str.len()) > 76000].to_parquet(f"/workspace/g/lora/eval/long_{k}.parquet", index=False)
P
log "START levers"
VENV=/workspace/venv LOG=$W/vllm_lev.log bash $L/serve.sh $F qwen 8000 65536 --moe-backend marlin >> $W/levers.log || exit 1
for D in apb t431; do
  data=$APB; [ $D = t431 ] && data=$T431
  run L3_args_$D spec_args.md $W/eval/route_args_$D.parquet
  run L3_esc_$D spec_escalation_v2.md $W/eval/route_escalation_$D.parquet
  run L1_t07_$D judge_prompt_v3g.md $data --temperature 0.7
done
run L2_long_t431 judge_prompt_v3g.md $W/eval/long_t431.parquet --max-ctx-chars 160000
run L2_long76_t431 judge_prompt_v3g.md $W/eval/long_t431.parquet
kill $(cat /tmp/vllm_8000.pid)
log "DONE levers"
