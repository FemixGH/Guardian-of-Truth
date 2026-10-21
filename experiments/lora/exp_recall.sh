#!/bin/bash
# Recall-oriented candidate vs submission 11, in A100 numerics (Marlin W8A16) on the FP8 checkpoint.
# Outputs: /workspace/g/lora/eval/<name>_{apb,t431}.jsonl + times in /workspace/g/lora/pipeline.log
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
W=/workspace/g/lora; PY=/workspace/venv/bin/python; L=/workspace/g/guardian/experiments/lora; F=/workspace/g/models/qwen36_fp8
mkdir -p $W/eval; cd /workspace/g/guardian
log() { echo "[$(date +%H:%M:%S)] $*" >> $W/pipeline.log; }
APB=runs/lora/apb_eval_ru.parquet; T431=runs/timing_431_labeled.parquet
run() {  # name prompt data_key extra_args...  (data_key: apb | t431 | path)
  local n=$1 p=$2 d=$3; shift 3
  local data=$d; [ "$d" = apb ] && data=$APB; [ "$d" = t431 ] && data=$T431
  local t0=$(date +%s)
  $PY experiments/run_judge.py --data $data --out $W/eval/${n}.jsonl --base-url http://127.0.0.1:8000/v1 --model qwen \
    --prompt src/guardian/$p --max-ctx-chars 76000 --layout head --concurrency 96 --max-tokens 4096 "$@" > $W/eval/${n}.log 2>&1
  log "$n took $(( $(date +%s) - t0 ))s ($(wc -l < $W/eval/${n}.jsonl) rows) $(tail -1 $W/eval/${n}.log | cut -c1-120)"
}
negs() {  # rows of $2 (parquet) judged 0 in $1 (jsonl) -> $3 (parquet)
  $PY - "$1" "$2" "$3" <<'P'
import json, sys, pandas as pd
r = {json.loads(l)["id"]: (json.loads(l)["p"] or 0) for l in open(sys.argv[1])}
d = pd.read_parquet(sys.argv[2]); d = d[d.id.map(lambda i: r.get(i, 0) < 0.5)]
d.to_parquet(sys.argv[3], index=False); print(len(d))
P
}
log "START recall experiments (Marlin numerics)"
VENV=/workspace/venv LOG=$W/vllm_exp.log bash $L/serve.sh $F qwen 8000 40960 --moe-backend marlin >> $W/pipeline.log || exit 1
grep -E "Selected .*Fp8LinearMethod|MoE backend" $W/vllm_exp.log | head -2 >> $W/pipeline.log
for D in apb t431; do run E0_v3f_${D} judge_prompt_v3f.md $D; done
for D in apb t431; do run E1_v3f_stop_${D} judge_prompt_v3f.md $D --stop-verdict; done
for D in apb t431; do GUARDIAN_KB_FOCUS=1 run E2_v3g_stop_kb_${D} judge_prompt_v3g.md $D --stop-verdict; done
for D in apb t431; do
  data=$APB; [ $D = t431 ] && data=$T431
  negs $W/eval/E2_v3g_stop_kb_${D}.jsonl $data $W/eval/neg_E2_${D}.parquet >> $W/pipeline.log
  GUARDIAN_KB_FOCUS=1 run E3_recall_stop_${D} judge_prompt_recall.md $W/eval/neg_E2_${D}.parquet --stop-verdict
  negs $W/eval/E1_v3f_stop_${D}.jsonl $data $W/eval/neg_E1_${D}.parquet >> $W/pipeline.log
  run E4_v2_stop_${D} judge_prompt_v2.md $W/eval/neg_E1_${D}.parquet --stop-verdict
done
kill $(cat /tmp/vllm_8000.pid)
log "DONE recall experiments"
