#!/bin/bash
# K2 8-bit settings experiments: s11k2 predict.py against an external K2 server under vLLM 0.19.1.
#   bash run_exp.sh <port> <variant> [<variant> ...]
PORT=$1; shift; O=/workspace/g/lora/eval; C=/workspace/k2exp
until curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"m"'; do sleep 10; done
for v in "$@"; do
  t0=$(date +%s)
  (cd /workspace/g/guardian/dist/s11k2 && /workspace/v19/bin/python scripts/predict.py --input /workspace/g/guardian/runs/timing_431_labeled.parquet \
     --output $O/k2x_$v.parquet --base-url http://127.0.0.1:$PORT/v1 --served-name m --calibration $C/cal_$v.json \
     --dump $O/k2x_$v.jsonl > $O/k2x_$v.log 2>&1)
  echo "[$(date +%H:%M:%S)] k2x_$v done $(( $(date +%s) - t0 ))s" >> $O/exp.log
done
