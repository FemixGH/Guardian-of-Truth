#!/bin/bash
# End-to-end judge fine-tuning pipeline (resumable: every stage leaves $W/done/<stage>).
#   bash run_pipeline.sh <stage|all>        stages: translate synth pool base_eval base_rollouts star sft gspo final
# Configuration via environment (defaults = rented H200 layout):
#   CODE   repo dir (guardian)            W      work dir              VENV  python venv
#   BASE   Qwen3.6-35B-A3B bf16 dir       FP8    Qwen3.6-35B-A3B-FP8   TPRO  translator (T-pro-it-2.1-FP8) dir
#   sizes: N_BASE (samples per row, base pass), N_GSPO, GSPO_ROWS (rows per iteration), GSPO_ITERS, EVAL_LIMIT (0=all)
set -euo pipefail
CODE=${CODE:-/workspace/g/guardian}; W=${W:-/workspace/g/lora}; VENV=${VENV:-/workspace/venv}
BASE=${BASE:-/workspace/g/models/qwen36_bf16}; FP8=${FP8:-/workspace/g/models/qwen36_fp8}; TPRO=${TPRO:-/workspace/g/models/tpro}
N_BASE=${N_BASE:-4}; N_GSPO=${N_GSPO:-8}; GSPO_ROWS=${GSPO_ROWS:-64}; GSPO_ITERS=${GSPO_ITERS:-4}
MAXTOK=${MAXTOK:-4096}; EVAL_LIMIT=${EVAL_LIMIT:-0}; SYN_MAXTOK=${SYN_MAXTOK:-6000}
LEN=${LEN:-40960}; PORT=${PORT:-8000}; CONC=${CONC:-96}; TP=${TP:-1}
APB_EN=${APB_EN:-runs/lora/apb_train_en.parquet}; SEEDS_EN=${SEEDS_EN:-runs/lora/tau2_seeds_en.parquet}
PY=$VENV/bin/python; L=$CODE/experiments/lora
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$VENV/bin:$PATH
mkdir -p "$W/done" "$W/data" "$W/eval" "$W/runs"
cd "$CODE"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$W/pipeline.log"; }
done_() { touch "$W/done/$1"; log "DONE $1"; }
is_done() { [ -f "$W/done/$1" ]; }
stop_server() { [ -f /tmp/vllm_${PORT}.pid ] && kill "$(cat /tmp/vllm_${PORT}.pid)" 2>/dev/null || true; sleep 10
  for _ in $(seq 1 30); do curl -s "127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break; sleep 3; done; }
serve() { stop_server; VENV=$VENV LOG=$W/vllm_$2.log bash "$L/serve.sh" "$1" qwen "$PORT" "${3:-$LEN}" --tensor-parallel-size "$TP"; }

# evaluate a model dir on timing_431 (valid + early turns) and the APB-ru eval half -> $W/eval/report.jsonl
evaluate() {  # <model_dir> <name>
  local M=$1 N=$2 LIM=()
  [ "$EVAL_LIMIT" != 0 ] && LIM=(--limit "$EVAL_LIMIT")
  serve "$M" "eval_$N"
  local t0=$(date +%s)
  for set in t431:runs/timing_431_labeled.parquet apb:runs/lora/apb_eval2_ru.parquet; do
    $PY experiments/run_judge.py --data "${set#*:}" --out "$W/eval/${N}_${set%%:*}.jsonl" --base-url "http://127.0.0.1:$PORT/v1" \
      --model qwen --prompt src/guardian/judge_prompt_v3f.md --max-ctx-chars 76000 --layout head --concurrency "$CONC" \
      --max-tokens "$MAXTOK" "${LIM[@]}" > "$W/eval/${N}_${set%%:*}.log" 2>&1
  done
  log "eval $N took $(( $(date +%s) - t0 ))s"
  stop_server
  $PY "$L/eval_report.py" --name "$N" --t431 "$W/eval/${N}_t431.jsonl" --apb "$W/eval/${N}_apb.jsonl" --out "$W/eval/report.jsonl" | tee -a "$W/pipeline.log"
}

st_translate() {
  REASONING_PARSER=none serve "$TPRO" translate 32768  # as in the earlier working T-pro run (max 40960)
  $PY experiments/translate_apb.py --base-url "http://127.0.0.1:$PORT/v1" --model qwen --src "$APB_EN" --out "$W/data/apb_train_ru.parquet" --concurrency 128
  $PY experiments/translate_apb.py --base-url "http://127.0.0.1:$PORT/v1" --model qwen --src "$SEEDS_EN" --out "$W/data/tau2_seeds_ru.parquet" --concurrency 128
  stop_server
}
st_synth() {
  serve "$BASE" synth
  $PY experiments/synth_llm.py --base-url "http://127.0.0.1:$PORT/v1" --model qwen --src "$W/data/tau2_seeds_ru.parquet" \
    --out "$W/data/tau2_synth_ru.parquet" --per-seed 1 --inject-frac 0.6 --para-frac 1.0 --concurrency "$CONC" --max-tokens "$SYN_MAXTOK"
  stop_server
}
st_pool() {
  $PY "$L/build_pool.py" --apb "$W/data/apb_train_ru.parquet" --seeds "$W/data/tau2_seeds_ru.parquet" --synth "$W/data/tau2_synth_ru.parquet" \
    --out "$W/data/pool.parquet" ${POOL_ARGS:-} | tee -a "$W/pipeline.log"
}
st_base_eval() { evaluate "$BASE" base; }
st_base_rollouts() {
  stop_server
  $PY "$L/rollouts.py" --tp "$TP" --model "$BASE" --data "$W/data/pool.parquet" --n "$N_BASE" --max-tokens "$MAXTOK" --max-model-len "$LEN" \
    --out "$W/runs/ro_base.pkl" 2>&1 | grep -v -E "^\(EngineCore|it/s\]" | tee -a "$W/pipeline.log"
  $PY "$L/select_rows.py" --pool "$W/data/pool.parquet" --summary "$W/runs/ro_base_summary.csv" --outdir "$W/sel" | tee -a "$W/pipeline.log"
}
st_star() {
  if [ "$($PY -c "import json;print(len(json.load(open('$W/sel/ids_star.json'))))")" = 0 ]; then
    $PY -c "import pickle;pickle.dump([],open('$W/runs/ro_star.pkl','wb'))"; return; fi
  $PY "$L/rollouts.py" --tp "$TP" --model "$BASE" --data "$W/data/pool.parquet" --ids "$W/sel/ids_star.json" --hint --n "$N_BASE" \
    --max-tokens "$MAXTOK" --max-model-len "$LEN" --out "$W/runs/ro_star.pkl" 2>&1 | grep -v -E "^\(EngineCore|it/s\]" | tee -a "$W/pipeline.log"
}
st_sft() {
  $PY "$L/train.py" --mode sft --base "$BASE" --rollouts "$W/runs/ro_base.pkl" "$W/runs/ro_star.pkl" --data "$W/data/pool.parquet" \
    --ids "$W/sel/ids_sft_rows.json" --out "$W/runs/sft" ${SFT_ARGS:-} 2>&1 | tee "$W/runs/sft.log" | grep -E "SFT:|saved|Error" | tee -a "$W/pipeline.log"
  rm -rf "$W/models/sft"; mkdir -p "$W/models"
  $PY "$L/merge.py" bf16 --base "$BASE" --adapter "$W/runs/sft/adapter" --out "$W/models/sft" 2>&1 | tail -1 | tee -a "$W/pipeline.log"
  evaluate "$W/models/sft" sft
}
st_gspo() {
  # starting point: SFT if it is not worse than base on the APB F1 (decided by hand: START=sft|base), default sft
  local START=${START:-sft} PREV_AD="" PREV_M=$BASE
  [ "$START" = sft ] && PREV_AD="$W/runs/sft/adapter" && PREV_M="$W/models/sft"
  local NG=$($PY -c "import json;print(len(json.load(open('$W/sel/ids_gspo.json'))))")
  log "informative GSPO rows: $NG"
  [ "$NG" -ge "${MIN_GSPO_ROWS:-50}" ] || { log "too few informative rows for GSPO ($NG) - stop"; return 1; }
  for it in $(seq 1 "$GSPO_ITERS"); do
    if is_done "gspo_$it"; then PREV_AD="$W/runs/gspo_$it/adapter"; PREV_M="$W/models/gspo_$it"; continue; fi
    $PY - "$W/sel/ids_gspo.json" "$it" "$GSPO_ROWS" "$W/sel/ids_gspo_$it.json" <<'E'
import json, sys
ids = json.load(open(sys.argv[1])); it, n = int(sys.argv[2]), int(sys.argv[3])
start = ((it - 1) * n) % max(1, len(ids)); part = (ids + ids)[start:start + n]
json.dump(part, open(sys.argv[4], "w")); print(f"iteration {it}: {len(part)} rows")
E
    $PY "$L/rollouts.py" --tp "$TP" --model "$PREV_M" --tokenizer "$BASE" --data "$W/data/pool.parquet" --ids "$W/sel/ids_gspo_$it.json" --n "$N_GSPO" \
      --max-tokens "$MAXTOK" --max-model-len "$LEN" --seed "$it" --out "$W/runs/ro_gspo_$it.pkl" 2>&1 | grep -E "prompts|done in|Error" | tee -a "$W/pipeline.log"
    $PY "$L/train.py" --mode gspo --base "$BASE" ${PREV_AD:+--adapter "$PREV_AD"} --rollouts "$W/runs/ro_gspo_$it.pkl" \
      --data "$W/data/pool.parquet" --out "$W/runs/gspo_$it" ${GSPO_ARGS:-} 2>&1 | tee "$W/runs/gspo_$it.log" | grep -E "GSPO:|saved|Error" | tee -a "$W/pipeline.log"
    tail -3 "$W/runs/gspo_$it/train_log.jsonl" | tee -a "$W/pipeline.log"
    rm -rf "$W/models/gspo_$it"
    $PY "$L/merge.py" bf16 --base "$BASE" --adapter "$W/runs/gspo_$it/adapter" --out "$W/models/gspo_$it" 2>&1 | tail -1 | tee -a "$W/pipeline.log"
    evaluate "$W/models/gspo_$it" "gspo_$it"
    # keep disk bounded: previous merged model is no longer needed (adapters are kept)
    [ "$it" -gt 1 ] && [ "$PREV_M" != "$BASE" ] && [ "$PREV_M" != "$W/models/sft" ] && rm -rf "$PREV_M"
    PREV_AD="$W/runs/gspo_$it/adapter"; PREV_M="$W/models/gspo_$it"
    done_ "gspo_$it"
  done
}
st_final() {  # FINAL=<merged bf16 dir to package>
  rm -rf "$W/models/final_fp8"
  $PY "$L/merge.py" fp8 --base "$BASE" --merged "${FINAL:?set FINAL=<merged bf16 dir>}" --fp8 "$FP8" --out "$W/models/final_fp8" 2>&1 | tail -2 | tee -a "$W/pipeline.log"
  evaluate "$W/models/final_fp8" final_fp8
}

STAGES=(translate synth pool base_eval base_rollouts star sft gspo)
run() { if is_done "$1"; then log "skip $1 (done)"; else log "START $1"; "st_$1"; done_ "$1"; fi; }
case "${1:-all}" in
  all) for s in "${STAGES[@]}"; do run "$s"; done ;;
  *) run "$1" ;;
esac
