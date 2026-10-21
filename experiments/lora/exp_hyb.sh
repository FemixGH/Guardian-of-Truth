#!/bin/bash
# Hybrid judge (dist/hyb: lordx64 Claude-4.7-Opus adapter first, base passes on capped rows) in the deployable form:
# organisers' FP8 + LoRA adapter in one vLLM server, A100 numerics (Marlin W8A16). s11r runs on the same server as
# the timing / noise reference. Outputs: /workspace/g/lora/eval/{hyb,s11r}_{t431,apb}.{csv,jsonl,log}
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; PY=/workspace/venv/bin/python
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q BOOTSTRAP_DONE $G/bootstrap.log 2>/dev/null; do sleep 20; done
log "START"
VENV=/workspace/venv LOG=$O/vllm.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin \
  --enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules opus47=$R/dist/hyb/model/lora_opus47 >> $O/exp.log \
  || { log "serve FAILED: $(grep -iE 'error' $O/vllm.log | tail -3)"; exit 1; }
curl -s 127.0.0.1:8000/v1/models | grep -o '"id":"[^"]*"' >> $O/exp.log
T431=$R/runs/timing_431_labeled.parquet; APB=$R/runs/lora/apb_eval_ru.parquet
run() {  # code_dir name set
  local data=$T431; [ $3 = apb ] && data=$APB; local t0=$(date +%s)
  (cd $1 && $PY scripts/predict.py --input $data --output $O/$2_$3.csv --base-url http://127.0.0.1:8000/v1 \
     --served-name judge --dump $O/$2_$3.jsonl > $O/$2_$3.log 2>&1)
  log "$2 $3 exit=$? $(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/$2_$3.log | tail -1) $(grep -o 'hybrid done.*' $O/$2_$3.log | tail -1)"
}
run $R/dist/hyb hyb t431
run $R/runs/calib/subs/s11r s11r t431
run $R/dist/hyb hyb apb
kill $(cat /tmp/vllm_8000.pid)
log "DONE"
