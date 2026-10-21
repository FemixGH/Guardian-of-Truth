"""Effect of the fact-verifier: which flags it removes, and their gold labels.
python experiments/verify_eval.py flags.jsonl ver.jsonl labels.parquet"""
import json, sys
import pandas as pd
fl = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
ve = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2])}
lab = dict(pd.read_parquet(sys.argv[3])[["id", "label"]].values)
flagged = [i for i, r in fl.items() if (r["p"] or 0) >= .5]
from collections import Counter
print("flagged", len(flagged), "verified", len(ve), "types", Counter(v["type"] for v in ve.values()))
for thr in (0.5, 0.2, 0.05):
    rem = [i for i in flagged if i in ve and ve[i]["type"] == "FACT" and ve[i]["p"] is not None and ve[i]["p"] < thr]
    c = Counter(lab.get(i) for i in rem)
    print(f"removed p<{thr}: {len(rem)} gold {dict(c)}  -> precision of removed (1=true pos) {c[1]/max(1,c[1]+c[0]):.2f}")
fact = [i for i in flagged if i in ve and ve[i]["type"] == "FACT"]
print("FACT-typed flags gold", dict(Counter(lab.get(i) for i in fact)))
