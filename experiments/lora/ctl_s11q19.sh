#!/bin/bash
# Platform check of the REAL final candidate: s11q end-to-end under vLLM 0.19.1 (the platform's version per the
# organisers' forum), own server, 26-min budget, on GPU0 after the K2 AWQ t431 run. GPU1 keeps GLM.
G=/workspace/g; O=$G/lora/eval; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q -E "k2a_v3f_t431 done|k2awq DIED|k2awq serve TIMEOUT" $O/exp.log 2>/dev/null; do sleep 20; done
SP=$(cat /tmp/srv_8001.pid 2>/dev/null); SUB=$(ps -o ppid= -p $SP 2>/dev/null | tr -d ' ')
[ -n "$SUB" ] && kill $SUB 2>/dev/null; for p in $(pgrep -f "[r]un_judge.py.*k2a_"); do kill $p; done
kill $SP 2>/dev/null; sleep 30; log "ctl19: K2 AWQ branch stopped; GPU0 -> s11q under vLLM 0.19.1"
/workspace/v19/bin/pip install -q pandas pyarrow openai >> $G/v19_pip.log 2>&1
S=$G/sub_s11q; rm -rf $S; mkdir -p $S/model
cp -r $G/guardian/dist/s11q/src $G/guardian/dist/s11q/scripts $G/guardian/dist/s11q/pyproject.toml $S/
cp $G/guardian/dist/s11q/model/calibration.json $S/model/
for f in $G/models/qwen36_fp8/*; do ln -s $f $S/model/; done
t0=$(date +%s)
(cd $S && PATH=/workspace/v19/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 /workspace/v19/bin/python scripts/predict.py \
   --input $G/guardian/runs/timing_431_labeled.parquet --output $O/s11q19_t431.parquet --dump $O/s11q19_t431.jsonl > $O/s11q19_t431.log 2>&1)
log "s11q19 t431 exit=$? wall=$(( $(date +%s) - t0 ))s $(grep -o 'positives=[0-9]*' $O/s11q19_t431.log | tail -1)"
log "ctl19: DONE"
