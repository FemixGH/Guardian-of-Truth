#!/bin/bash
# Step 2 of the hybrid check: rows where the Opus-4.7 adapter hit the 4096-token cap in exp_hyb.sh are re-judged by
# the same adapter with an 8192-token budget (same prompt/layout/notes as predict.py). Compared offline with what
# the base did on these rows. Outputs: /workspace/g/lora/eval/a8k_{t431,apb}.jsonl
export CUDA_HOME=/usr/local/cuda PATH=/workspace/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
G=/workspace/g; O=$G/lora/eval; R=$G/guardian; PY=/workspace/venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*" >> $O/exp.log; }
until grep -q "\] DONE$" $O/exp.log 2>/dev/null; do sleep 20; done
sleep 10
log "STEP2 START"
VENV=/workspace/venv LOG=$O/vllm2.log bash $R/experiments/lora/serve.sh $G/models/qwen36_fp8 judge 8000 131072 --moe-backend marlin \
  --enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules opus47=$R/dist/hyb/model/lora_opus47 >> $O/exp.log \
  || { log "step2 serve FAILED"; exit 1; }
for set in t431 apb; do
  data=$R/runs/timing_431_labeled.parquet; [ $set = apb ] && data=$R/runs/lora/apb_eval_ru.parquet
  $PY - "$O/hyb_$set.jsonl" "$data" "$O/capped_$set.parquet" <<'E'
import json, sys
import pandas as pd
dump, data, out = sys.argv[1:]
capped = {r["id"] for r in map(json.loads, open(dump)) if r["pass"] == "A" and r.get("forced")}
df = pd.read_parquet(data)
df[df.id.isin(capped)].to_parquet(out, index=False)
print(len(capped), "capped rows")
E
  t0=$(date +%s)
  (cd $R/dist/hyb && $PY - "$O/capped_$set.parquet" "$O/a8k_$set.jsonl" > $O/a8k_$set.log 2>&1 <<'E2'
import json, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, "src")
from guardian.features import flags
from guardian.judge import Judge
from guardian.parse import parse_example
data, out = sys.argv[1:]
df = pd.read_parquet(data)
jg = Judge("http://127.0.0.1:8000/v1", "opus47", mode="think", max_tokens=8192,
           prompt_path="src/guardian/judge_prompt_v3f.md", layout="head")
def work(row):
    notes = flags(parse_example(row.prompt, row.response))[1]
    r = jg.score(row.prompt, row.response, notes, 76000, 2)
    return {"id": row.id, **{k: r.get(k) for k in ("p", "verdict", "forced", "finish_reason", "completion_tokens", "prompt_tokens")}}
with ThreadPoolExecutor(96) as ex, open(out, "w") as fh:
    for r in ex.map(work, df.itertuples(index=False)):
        fh.write(json.dumps(r) + "\n")
print("done", len(df))
E2
)
  log "a8k $set exit=$? $(( $(date +%s) - t0 ))s rows=$(wc -l < $O/a8k_$set.jsonl)"
done
kill $(cat /tmp/vllm_8000.pid)
log "STEP2 DONE"
