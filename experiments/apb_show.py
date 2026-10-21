"""Print APB candidate-error steps compactly for manual review: python experiments/apb_show.py START END"""
import json, re, sys
import pandas as pd
d = pd.read_parquet("runs/apb_tau2.parquet")
apb = {f"apb_{json.loads(l)['data_source'].replace('tau2_','')}_{json.loads(l)['query_index']}_{json.loads(l)['sample_index']}": json.loads(l)
       for l in open("../data_ext/agentprocessbench/tau2/test.jsonl")}
pos = d[d.label == 1].reset_index(drop=True)
a, b = int(sys.argv[1]), int(sys.argv[2])
for k in range(a, min(b, len(pos))):
    r = pos.iloc[k]
    t = apb[r.traj]
    blocks = re.split(r"(?=⟦(?:USER|ASSISTANT)[^⟧]*⟧)", r.prompt)
    tail = blocks[-3:] if len(blocks) > 3 else blocks[1:]
    print(f"\n######## [{k}] {r.id}  (task purpose: {((t.get('task_description') or {}).get('purpose') or '')[:200]})")
    for blk in tail:
        s = re.sub(r"(← TOOL_RESPONSE [^:]+:)(.{400}).*", r"\1\2 …", blk.strip(), flags=re.S)
        print("  CTX>", s[:900].replace("\n", "\n       "))
    print("  STEP>", r.response.strip()[:1100].replace("\n", "\n        "))
