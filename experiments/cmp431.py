"""Compare judge runs on timing_431: valid confusion (gold>=0), flag rate on early turns (gold -1), agreement.
python experiments/cmp431.py a.jsonl b.jsonl ..."""
import json, sys
import pandas as pd
lab = dict(pd.read_parquet("runs/timing_431_labeled.parquet")[["id", "label"]].values)
runs = {}
for f in sys.argv[1:]:
    runs[f] = {json.loads(l)["id"]: (json.loads(l)["p"] or 0) >= .5 for l in open(f) if json.loads(l).get("p") is not None}
for f, r in runs.items():
    v = [(lab[i], r[i]) for i in r if lab.get(i, -9) >= 0]
    tp = sum(a == 1 and b for a, b in v); fp = sum(a == 0 and b for a, b in v); fn = sum(a == 1 and not b for a, b in v)
    early = [r[i] for i in r if lab.get(i) == -1]
    print(f"{f.split('/')[-1]:28s} n={len(r)} valid TP/FP/FN {tp}/{fp}/{fn}  early flag rate {sum(early)/max(1,len(early)):.3f} ({sum(early)}/{len(early)})  all-rows flag {sum(r.values())/len(r):.3f}")
fs = list(runs)
for a in range(len(fs)):
    for b in range(a + 1, len(fs)):
        common = set(runs[fs[a]]) & set(runs[fs[b]])
        ag = sum(runs[fs[a]][i] == runs[fs[b]][i] for i in common) / max(1, len(common))
        print("agree", fs[a].split('/')[-1], fs[b].split('/')[-1], round(ag, 3))
