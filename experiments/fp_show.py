"""Show judge false positives on the RU-translated APB eval with the key justification from the reasoning.
python experiments/fp_show.py runs/cc/ru_qwen_v3f.jsonl START END"""
import json, re, sys
import pandas as pd
d = pd.read_parquet("runs/apb_eval.parquet").set_index("id")
ru = pd.read_parquet("runs/cc/apb_eval_ru.parquet").set_index("id") if __import__("os").path.exists("runs/cc/apb_eval_ru.parquet") else d
r = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
fp = sorted(i for i, x in r.items() if (x["p"] or 0) >= .5 and d.loc[i].label == 0)
a, b = int(sys.argv[2]), int(sys.argv[3])
KEY = re.compile(r"[^.\n]*(hallucinat|unsupported|not supported|contradict|violat|fabricat|not in the context|no evidence|false)[^.\n]*[.\n]", re.I)
for k, i in enumerate(fp[a:b], a):
    x = r[i]; t = (x.get("reasoning") or "") + "\n" + (x.get("text") or "")
    keys = [m.group(0).strip() for m in KEY.finditer(t)]
    keys = [s for s in keys if "Only these are NOT" not in s and "Output format" not in s and "VERDICT: 1` (at least" not in s][-3:]
    print(f"\n[{k}] {i} | STEP: {ru.loc[i].response.strip()[:260].replace(chr(10), ' ')}")
    for s in keys:
        print("   WHY:", s[:260])
print("\nTOTAL FP", len(fp))
