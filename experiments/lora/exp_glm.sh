#!/bin/bash
# s11v and s11q on the external GLM-5.3-labelled set (664 rows), one external Qwen FP8 server (Marlin = A100 numerics) per GPU.
G=/workspace/g; O=$G/lora/eval; PY=/workspace/venv/bin/python; mkdir -p $O
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -qs DLDONE $G/dl_qwen36_fp8.log && $PY -c "import vllm" 2>/dev/null; do sleep 15; done
log "GLM START"
D=$G/guardian/runs/karimov_664.parquet
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
go() {  # gpu port variant
  CUDA_VISIBLE_DEVICES=$1 setsid $PY -m vllm.entrypoints.openai.api_server --model $G/models/qwen36_fp8 --served-model-name judge \
    --port $2 --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 96 --enable-prefix-caching --language-model-only \
    --reasoning-parser qwen3 --moe-backend marlin > $O/srv_$2.log 2>&1 < /dev/null &
  for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q judge && break; sleep 10; done
  local t0=$(date +%s); (cd $G/guardian/dist/$3 && $PY scripts/predict.py --input $D --output $O/$3_glm.csv \
    --base-url http://127.0.0.1:$2/v1 --served-name judge --budget-sec 3600 --dump $O/$3_glm.jsonl > $O/$3_glm.log 2>&1)
  log "$3_glm done $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$3_glm.log | tail -1)"; }
go 0 8011 s11v & go 1 8012 s11q & wait
log "GLM DONE"
