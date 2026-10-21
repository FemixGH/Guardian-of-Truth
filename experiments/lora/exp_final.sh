#!/bin/bash
# Final checks. GPU0: s11v platform-style (vLLM 0.19.1, own server, 26-min budget) on the 543-row private-like set.
# GPU1: Qwen FP8 (A100 numerics) server -> s11v t431 run 3, s11x (trimmed v3m) t431 and APB.
G=/workspace/g; O=$G/lora/eval; PY=/workspace/venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
T431=$G/guardian/runs/timing_431_labeled.parquet; APB=$G/guardian/runs/lora/apb_eval_ru.parquet; T543=$G/guardian/experiments/lora/timing_543.parquet
(
  S=$G/sub_s11v; rm -rf $S; mkdir -p $S/model
  cp -r $G/guardian/dist/s11v/src $G/guardian/dist/s11v/scripts $G/guardian/dist/s11v/pyproject.toml $S/
  cp $G/guardian/dist/s11v/model/calibration.json $S/model/; for f in $G/models/qwen36_fp8/*; do ln -s $f $S/model/; done
  t0=$(date +%s)
  (cd $S && PATH=/workspace/v19/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 /workspace/v19/bin/python scripts/predict.py \
     --input $T543 --output $O/s11v19_t543.parquet --dump $O/s11v19_t543.jsonl > $O/s11v19_t543.log 2>&1)
  log "s11v19_t543 exit=$? wall=$(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/s11v19_t543.log | tail -1)"
) &
(
  export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
  CUDA_VISIBLE_DEVICES=1 setsid $PY -m vllm.entrypoints.openai.api_server --model $G/models/qwen36_fp8 --served-model-name judge \
    --port 8011 --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 96 --enable-prefix-caching --language-model-only \
    --reasoning-parser qwen3 --moe-backend marlin > $O/srv_qwen_final.log 2>&1 < /dev/null &
  for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:8011/v1/models | grep -q judge && break; sleep 10; done
  run() { local t0=$(date +%s); (cd $G/guardian/dist/$1 && $PY scripts/predict.py --input $3 --output $O/$1_$2.csv \
      --base-url http://127.0.0.1:8011/v1 --served-name judge --dump $O/$1_$2.jsonl > $O/$1_$2.log 2>&1)
    log "$1_$2 done $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$1_$2.log | tail -1)"; }
  run s11v t431_r3 $T431; run s11x t431 $T431; run s11x apb $APB
) &
wait; log "FINAL CHECKS DONE"
