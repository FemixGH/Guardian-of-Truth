#!/bin/bash
# Gated re-ask lever (s11gr = s11v + baseline one-token gate + pass-0 re-ask on clean rows with P_base >= 0.5).
# GPU0: platform-style (vLLM 0.19.1 venv, own server on 8012, default 27.5-min budget) on the 543-row private-like
# set, then an external-server t431 run. GPU1: external Qwen FP8 server (Marlin = A100 numerics): t431 x2 and APB.
G=/workspace/g; O=$G/lora/eval; PY=/workspace/venv/bin/python; mkdir -p $O
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -qs DLDONE $G/dl_qwen36_fp8.log && $PY -c "import vllm" 2>/dev/null && grep -qs V19DONE /workspace/v19_pip.log; do sleep 15; done
log "GATE START"
T431=$G/guardian/runs/timing_431_labeled.parquet; APB=$G/guardian/runs/lora/apb_eval_ru.parquet; T543=$G/guardian/runs/timing_543.parquet
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
serve() {  # gpu port
  CUDA_VISIBLE_DEVICES=$1 setsid env PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0 \
    $PY -m vllm.entrypoints.openai.api_server --model $G/models/qwen36_fp8 --served-model-name judge \
    --port $2 --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 96 --enable-prefix-caching --language-model-only \
    --reasoning-parser qwen3 --moe-backend marlin > $O/srv_$2.log 2>&1 < /dev/null &
  for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q judge && break; sleep 10; done; }
run() { local t0=$(date +%s); (cd $G/guardian/dist/$1 && $PY scripts/predict.py --input $3 --output $O/$1_$2.csv \
    --base-url http://127.0.0.1:$4/v1 --served-name judge --budget-sec 3600 --dump $O/$1_$2.jsonl > $O/$1_$2.log 2>&1)
  log "$1_$2 done $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$1_$2.log | tail -1)"; }
(
  S=$G/sub_s11gr; rm -rf $S; mkdir -p $S/model
  cp -r $G/guardian/dist/s11gr/src $G/guardian/dist/s11gr/scripts $G/guardian/dist/s11gr/pyproject.toml $S/
  cp $G/guardian/dist/s11gr/model/calibration.json $S/model/; for f in $G/models/qwen36_fp8/*; do ln -s $f $S/model/; done
  t0=$(date +%s)
  (cd $S && PATH=/workspace/v19/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 /workspace/v19/bin/python scripts/predict.py \
     --input $T543 --output $O/s11gr19_t543.parquet --port 8012 --dump $O/s11gr19_t543.jsonl > $O/s11gr19_t543.log 2>&1)
  log "s11gr19_t543 (GPU0, port 8012) exit=$? wall=$(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/s11gr19_t543.log | tail -1)"
  sleep 20; serve 0 8013; run s11gr t431_r3 $T431 8013
) &
(
  serve 1 8011
  run s11gr t431_r1 $T431 8011; run s11gr t431_r2 $T431 8011; run s11gr apb $APB 8011
) &
wait; log "GATE DONE"
