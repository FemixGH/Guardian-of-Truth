#!/bin/bash
# After GLM's main t431 run, free GPU1 (stop the rest of the GLM branch) and run the s11k2 end-to-end tests there.
O=/workspace/g/lora/eval; log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q -E "glm_v3f_t431 done|glm47f DIED|glm47f serve TIMEOUT" $O/exp.log 2>/dev/null; do sleep 20; done
SP=$(cat /tmp/srv_8002.pid 2>/dev/null); SUB=$(ps -o ppid= -p $SP 2>/dev/null | tr -d ' ')
[ -n "$SUB" ] && kill $SUB 2>/dev/null; for p in $(pgrep -f "[r]un_judge.py.*glm_"); do kill $p; done
kill $SP 2>/dev/null; sleep 30; log "ctl: GLM branch stopped after its t431 run; GPU1 -> s11k2 e2e"
GPU=1 bash /workspace/g/guardian/experiments/lora/exp_e2e_k2.sh /workspace/g/guardian/runs/timing_431_labeled.parquet t431
GPU=1 bash /workspace/g/guardian/experiments/lora/exp_e2e_k2.sh /workspace/g/guardian/experiments/lora/timing_543.parquet t543
log "ctl: E2E DONE"
