#!/bin/bash
# s11g = s11f + read-through-wrapper repeat fix, retail return+exchange offers, airline basic-economy cabin change and
# all-segment cabin quotes, banking user-chosen paid options, evidence windows inside long KB lines. Runs after
# exp_s11f.sh on a fresh plain FP8 server. Outputs: /workspace/g/lora/eval/s11g_{t431,apb}.{csv,jsonl,log}
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; PY=/workspace/venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q "S11F DONE" $O/exp.log 2>/dev/null; do sleep 15; done
sleep 10
log "S11G START"
VENV=/workspace/venv LOG=$O/vllm4.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin >> $O/exp.log \
  || { log "s11g serve FAILED"; exit 1; }
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet
run() {  # code_dir name set
  local data=$T431; [ $3 = apb ] && data=$APB; local t0=$(date +%s)
  (cd $1 && $PY scripts/predict.py --input $data --output $O/$2_$3.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/$2_$3.jsonl > $O/$2_$3.log 2>&1)
  log "$2 $3 exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$2_$3.log | tail -1)"
}
run $R/dist/s11g s11g t431
run $R/dist/s11g s11g apb
kill $(cat /tmp/vllm_8000.pid)
log "S11G DONE"
