"""Show the full last tool outputs + step for given APB ids (for checking claims that my compact view truncated)."""
import re, sys
import pandas as pd
d = pd.read_parquet("runs/apb_tau2.parquet").set_index("id")
for i in sys.argv[1:]:
    r = d.loc[i]
    blocks = re.split(r"(?=⟦(?:USER|ASSISTANT)[^⟧]*⟧)", r.prompt)
    print(f"\n######## {i}")
    for blk in blocks[-4:]:
        print("  CTX>", blk.strip()[:4000].replace("\n", "\n       "))
    print("  STEP>", r.response.strip()[:1500])
