#!/bin/bash
# Calibration of the offline proxy against real test scores: every past submission (exact code from git) is run
# end-to-end (its own predict.py) against ONE shared vLLM server in A100 numerics (Marlin W8A16), on t431 (twice)
# and on APB-ru (once). Outputs: /workspace/g/calib/<sub>_<set>_<run>.{csv,jsonl,log}
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/calib; S=$G/guardian/runs/calib/subs; PY=/workspace/venv/bin/python
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/calib.log; }
until grep -q BOOTSTRAP_DONE $G/bootstrap.log 2>/dev/null; do sleep 30; done
log "START"
VENV=/workspace/venv LOG=$O/vllm.log bash $G/guardian/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/calib.log || { log "serve FAILED"; exit 1; }
grep -E "Selected .*Fp8LinearMethod|MoE backend" $O/vllm.log | head -2 >> $O/calib.log
T431=$G/guardian/runs/timing_431_labeled.parquet; APB=$G/guardian/runs/lora/apb_eval_ru.parquet
one() {  # sub set run
  local s=$1 set=$2 r=$3 data=$T431; [ $set = apb ] && data=$APB
  local t0=$(date +%s)
  (cd $S/$s && $PY scripts/predict.py --input $data --output $O/${s}_${set}_${r}.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/${s}_${set}_${r}.jsonl > $O/${s}_${set}_${r}.log 2>&1)
  log "$s $set run$r exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/${s}_${set}_${r}.log | tail -1)"
}
for r in 1 2; do for s in sub11 cand12 s11r sub8 sub5 sub7 sub4 sub1; do one $s t431 $r; done; done
for s in sub11 cand12 s11r sub8 sub5 sub7 sub4 sub1; do one $s apb 1; done
kill $(cat /tmp/vllm_8000.pid)
log "DONE"
