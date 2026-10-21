#!/bin/bash
# Platform-style end-to-end test of the K2-Horizon submission s11k2 on ONE GPU: build the submission dir (code +
# fixed K2 AWQ weights), `pip install .` into a fresh venv on top of the system vLLM env, run predict.py WITHOUT
# --base-url (it starts its own vLLM) with the 26-min budget. GPU chosen by $GPU (default 1).
#   GPU=1 bash exp_e2e_k2.sh <data.parquet> <tag>
G=/workspace/g; R=$G/guardian; O=$G/lora/eval; PY=/workspace/venv/bin/python; GPU=${GPU:-1}
DATA=$1; TAG=$2; S=$G/sub_k2
export PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until [ -f $G/models/k2awq/.fixed ]; do sleep 15; done
if [ ! -f $S/.built ]; then
  rm -rf $S; mkdir -p $S/model
  cp -r $R/dist/s11k2/src $R/dist/s11k2/scripts $R/dist/s11k2/pyproject.toml $R/dist/s11k2/README.md $S/
  cp -rL $G/models/k2awq/. $S/model/; rm -rf $S/model/.cache $S/model/.fixed
  cp $R/dist/s11k2/model/calibration.json $S/model/
  touch $S/.built; log "e2e: submission dir built, $(du -sh $S | cut -f1)"
fi
t0=$(date +%s)
(cd $S && CUDA_VISIBLE_DEVICES=$GPU $PY scripts/predict.py --input $DATA --output $O/e2e_$TAG.parquet \
   --dump $O/e2e_$TAG.jsonl > $O/e2e_$TAG.log 2>&1)
rc=$?
log "e2e $TAG exit=$rc wall=$(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/e2e_$TAG.log | tail -1) $(grep -o 'vLLM ready[^;]*' $O/e2e_$TAG.log | head -1)"
