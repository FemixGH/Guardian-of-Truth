#!/bin/bash
# Third-pass lever. GPU0: platform-style (vLLM 0.19.1 venv, own server, default 27.5-min budget) on the 543-row
# private-like set for s11y then s11z. GPU1: Qwen FP8 (A100 numerics) external server: t431 x2 and APB for both.
G=/workspace/g; O=$G/lora/eval; PY=/workspace/venv/bin/python; mkdir -p $O
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -qs DLDONE $G/dl_qwen36_fp8.log && $PY -c "import vllm" 2>/dev/null && grep -qs V19DONE $G/../v19_pip.log /workspace/v19_pip.log; do sleep 15; done
log "3PASS START"
T431=$G/guardian/runs/timing_431_labeled.parquet; APB=$G/guardian/runs/lora/apb_eval_ru.parquet; T543=$G/guardian/runs/timing_543.parquet
(
  for v in s11y s11z; do
    S=$G/sub_$v; rm -rf $S; mkdir -p $S/model
    cp -r $G/guardian/dist/$v/src $G/guardian/dist/$v/scripts $G/guardian/dist/$v/pyproject.toml $S/
    cp $G/guardian/dist/$v/model/calibration.json $S/model/; for f in $G/models/qwen36_fp8/*; do ln -s $f $S/model/; done
    t0=$(date +%s)
    (cd $S && PATH=/workspace/v19/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 /workspace/v19/bin/python scripts/predict.py \
       --input $T543 --output $O/${v}19_t543.parquet --dump $O/${v}19_t543.jsonl > $O/${v}19_t543.log 2>&1)
    log "${v}19_t543 exit=$? wall=$(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/${v}19_t543.log | tail -1)"
  done
) &
(
  export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
  CUDA_VISIBLE_DEVICES=1 setsid $PY -m vllm.entrypoints.openai.api_server --model $G/models/qwen36_fp8 --served-model-name judge \
    --port 8011 --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 96 --enable-prefix-caching --language-model-only \
    --reasoning-parser qwen3 --moe-backend marlin > $O/srv_qwen1.log 2>&1 < /dev/null &
  for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:8011/v1/models | grep -q judge && break; sleep 10; done
  run() { local t0=$(date +%s); (cd $G/guardian/dist/$1 && $PY scripts/predict.py --input $3 --output $O/$1_$2.csv \
      --base-url http://127.0.0.1:8011/v1 --served-name judge --budget-sec 3600 --dump $O/$1_$2.jsonl > $O/$1_$2.log 2>&1)
    log "$1_$2 done $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$1_$2.log | tail -1)"; }
  run s11y t431_r1 $T431; run s11z t431_r1 $T431; run s11y t431_r2 $T431; run s11z t431_r2 $T431; run s11y apb $APB; run s11z apb $APB
) &
wait; log "3PASS DONE"
