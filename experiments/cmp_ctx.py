"""Compare judge verdicts on the same rows across context modes (full / truncate / compress)."""
import json
import sys

man = json.load(open("experiments/manual_history_labels.json"))


def load(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("p") is not None:
            out[r["id"]] = r
    return out


runs = {name: load(path) for name, path in (a.split("=", 1) for a in sys.argv[1:])}
ids = sorted(set.intersection(*(set(r) for r in runs.values())))
print("rows", len(ids))
for name, r in runs.items():
    fl = [i for i in ids if r[i]["p"] >= 0.5]
    tp = sum(1 for i in fl if r[i].get("label") == 1)
    fp = sum(1 for i in fl if r[i].get("label") == 0)
    fn = sum(1 for i in ids if r[i].get("label") == 1 and r[i]["p"] < 0.5)
    hist = [i for i in ids if r[i].get("label") in (-1, None)]
    hf = sum(1 for i in hist if r[i]["p"] >= 0.5)
    mr = sum(1 for i in fl if man.get(i) == 1); mn = sum(1 for i in fl if man.get(i) == 0)
    ct = sum(r[i].get("completion_tokens") or 0 for i in ids) / len(ids)
    print(f"{name:12s} flagged {len(fl):3d} | valid tp/fp/fn {tp}/{fp}/{fn} | history {hf}/{len(hist)} | manual real {mr} nitpick {mn} | ctok {ct:.0f}")
names = list(runs)
for i in ids:
    v = [int(runs[n][i]["p"] >= 0.5) for n in names]
    if len(set(v)) > 1:
        print("  diff", i[:55], dict(zip(names, v)), "label", runs[names[0]][i].get("label"), "manual", man.get(i))
