"""Compact view of sampled label-0 APB steps for the negative spot-check: python experiments/apb_show_neg.py START END"""
import re, sys
import pandas as pd
d = pd.read_parquet("runs/apb_tau2.parquet")
neg = pd.concat([g.sample(20, random_state=7) for _, g in d[d.label == 0].groupby("domain")]).reset_index(drop=True)
neg[["id"]].to_csv("runs/apb_neg_sample_ids.csv", index=False)
a, b = int(sys.argv[1]), int(sys.argv[2])
for k in range(a, min(b, len(neg))):
    r = neg.iloc[k]
    blocks = re.split(r"(?=⟦(?:USER|ASSISTANT)[^⟧]*⟧)", r.prompt)
    print(f"\n######## [{k}] {r.id}")
    for blk in blocks[-1:]:
        s = re.sub(r"(← TOOL_RESPONSE [^:]+:)(.{450}).*", r"\1\2 …", blk.strip(), flags=re.S)
        print("  CTX>", s[:600].replace("\n", "\n       "))
    print("  STEP>", r.response.strip()[:700].replace("\n", "\n        "))
