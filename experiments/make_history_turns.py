"""Slice earlier assistant turns out of the valid contexts -> extra (mostly clean, unlabelled) examples.

Each output row: prompt = transcript up to the turn header, response = that assistant turn without tool
responses. label = -1 (unknown, presumed mostly 0). Used to estimate false-positive rates.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_parquet(ROOT.parent / "valid.parquet")
rows = []
seen = set()
for _, r in d.iterrows():
    task = r.id.split("::")[0]
    heads = [m for m in re.finditer(r"⟦ASSISTANT · ход (\d+)⟧", r.prompt)]
    for k, m in enumerate(heads):
        turn = int(m.group(1))
        key = (task, turn)
        if key in seen:
            continue
        seen.add(key)
        end = heads[k + 1].start() if k + 1 < len(heads) else len(r.prompt)
        body = r.prompt[m.start():end]
        # cut at next USER message
        u = body.find("⟦USER⟧")
        if u > 0:
            body = body[:u]
        cut = body.find("\t← TOOL_RESPONSE")
        if cut > 0:
            body = body[:cut]
        resp = body.rstrip() + "\n"
        rows.append(dict(id=f"{task}::h{turn}", prompt=r.prompt[:m.start()], response=resp, label=-1,
                         tool_only=int("TOOL_CALL" in resp and len(re.sub(r"⟦[^⟧]*⟧|\s*→ TOOL_CALL.*", "", resp).strip()) == 0)))
out = pd.DataFrame(rows)
print(len(out), "turns;", out.tool_only.sum(), "tool-only")
out.to_parquet(ROOT / "experiments" / "history_turns.parquet", index=False)
