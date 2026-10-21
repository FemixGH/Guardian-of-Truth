"""Split the re-labelled AgentProcessBench τ² rows BY TASK into a training half and an evaluation half.

The eval half reuses rows of runs/apb_eval.parquet (already translated to Russian: apb_eval_ru.parquet), so only the
training half needs translation. Labels: 1 = manually confirmed hallucination, 0 = clean (B/C/negative-issue codes are
dropped). Split is stratified by domain and balanced on positives (greedy).

python experiments/lora/split_apb.py
"""
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
a = pd.read_parquet(ROOT / "runs/apb_tau2_labeled.parquet")
ev = pd.read_parquet(ROOT / "runs/apb_eval.parquet")
a = a[a.label.isin([0, 1])]
random.seed(7)
train_tasks, eval_tasks = set(), set()
for dom, g in a.groupby("domain"):
    st = g.groupby("task").label.agg(pos="sum", n="size").reset_index().sample(frac=1, random_state=7)
    st = st.sort_values("pos", ascending=False, kind="stable")
    tp = ep = tn = en = 0
    for t in st.itertuples():  # greedy: give the task to the half with fewer positives (then fewer rows)
        if (tp, tn) <= (ep, en):
            train_tasks.add(t.task); tp += t.pos; tn += t.n
        else:
            eval_tasks.add(t.task); ep += t.pos; en += t.n
tr = a[a.task.isin(train_tasks)]
ev2 = ev[ev.task.isin(eval_tasks) & ev.label.isin([0, 1])]
out = ROOT / "runs/lora"
tr.to_parquet(out / "apb_train_en.parquet", index=False)
json.dump({"train_tasks": sorted(train_tasks), "eval_tasks": sorted(eval_tasks), "eval_ids": sorted(ev2.id)},
          open(out / "apb_split.json", "w"), indent=1)
print("train rows", len(tr), tr.label.value_counts().to_dict(), tr.groupby("domain").label.sum().to_dict())
print("eval2 rows (from apb_eval)", len(ev2), ev2.label.value_counts().to_dict(), ev2.groupby("domain").label.sum().to_dict())
print("train rows that are also in apb_eval (must be 0 after use of eval2 only):", len(set(tr.id) & set(ev2.id)))
