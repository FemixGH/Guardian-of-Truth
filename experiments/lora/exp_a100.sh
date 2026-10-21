#!/bin/bash
# Platform-like timing on a real A100 80GB: for each candidate code dir, build a submission dir (code + FP8 weights
# linked into model/), `pip install .`, then `python scripts/predict.py --input timing_543 --output pred.parquet` with
# NO external server (predict.py starts vLLM from ./model itself, as on the platform). Wall time includes vLLM start.
#   CANDS="s11f s11r" bash exp_a100.sh      Outputs: /workspace/g/a100/<cand>.{log,parquet,jsonl,time}
G=/workspace/g; R=$G/guardian; O=$G/a100; V=/workspace/venv
mkdir -p $O; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/a100.log; }
until grep -q BOOTSTRAP_DONE $G/bootstrap.log 2>/dev/null; do sleep 20; done
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader >> $O/a100.log
for c in ${CANDS:-s11f s11r}; do
  S=$G/sub_$c; rm -rf $S; mkdir -p $S/model
  src=$R/dist/$c; [ -d $src ] || src=$R/runs/calib/subs/$c
  cp -r $src/src $src/scripts $src/pyproject.toml $S/; cp $src/model/calibration.json $S/model/
  for f in $G/models/qwen36_fp8/*; do ln -s $f $S/model/; done
  $V/bin/pip install -q $S > $O/$c.pip.log 2>&1 || { log "$c pip FAILED"; continue; }
  t0=$(date +%s)
  (cd $S && PATH=$V/bin:$PATH $V/bin/python scripts/predict.py --input $R/runs/timing_543.parquet --output $O/$c.parquet \
     --dump $O/$c.jsonl > $O/$c.log 2>&1)
  rc=$?; dt=$(( $(date +%s) - t0 )); echo $dt > $O/$c.time
  log "$c exit=$rc wall=${dt}s $(grep -o 'vLLM ready.*' $O/$c.log | head -1 | cut -c1-40) $(grep -o 'judge calls finished.*' $O/$c.log | tail -1) $(grep -o 'positives=[0-9]*' $O/$c.log | tail -1)"
  pkill -f "vllm.entrypoints" ; sleep 20
done
log "A100 DONE"
