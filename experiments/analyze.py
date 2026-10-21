"""Compare combination strategies on judged valid rows (judge jsonl + deterministic features).

python experiments/analyze.py runs/think_v1.jsonl [runs/other.jsonl ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian import combine  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian.parse import parse_example  # noqa: E402


def f1(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = ((y == 1) & (yhat == 1)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn), tp, fp, fn


def boot_ci(y, yhat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, yhat = np.asarray(y), np.asarray(yhat)
    s = [f1(y[idx], yhat[idx])[0] for idx in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5]).round(3)


def main(paths):
    d = pd.read_parquet(ROOT.parent / "valid.parquet")
    feats = {r.id: flags(parse_example(r.prompt, r.response))[0] for r in d.itertuples()}
    cfg = combine.load(ROOT / "model" / "calibration.json")
    for p in paths:
        rows = {}
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            rows[r["id"]] = r
        sub = d[d.id.isin(rows)].copy()
        def _p(r):  # older runs: trust the textual verdict when logprob extraction disagreed
            p_, v = r.get("p"), r.get("verdict")
            if v is not None and (p_ is None or int(p_ >= 0.5) != v):
                return 0.97 if v == 1 else 0.03
            return p_
        sub["p"] = [_p(rows[i]) for i in sub.id]
        sub["ctok"] = [rows[i].get("completion_tokens") for i in sub.id]
        y = sub.label.values
        pj = sub.p.fillna(0.5).values
        print(f"\n=== {p}  n={len(sub)}  missing p={sub.p.isna().sum()}  mean ctok={np.nanmean(sub.ctok.astype(float)):.0f}"
              f"  max ctok={np.nanmax(sub.ctok.astype(float)):.0f}")
        for name, yhat in [
            ("judge@0.5", (pj >= 0.5).astype(int)),
            ("judge+hard", np.array([combine.final_label(feats[i], p_, cfg) for i, p_ in zip(sub.id, sub.p)])),
            ("rules only", np.array([combine.final_label(feats[i], None, cfg) for i in sub.id])),
            ("judge OR tool_only", ((pj >= 0.5) | np.array([feats[i]["tool_only"] > 0 for i in sub.id])).astype(int)),
        ]:
            sc, tp, fp, fn = f1(y, yhat)
            print(f"  {name:22s} F1={sc:.3f} tp={tp} fp={fp} fn={fn} CI95={boot_ci(y, yhat)}")
        for t in (0.2, 0.3, 0.4, 0.6, 0.7, 0.8):
            sc, tp, fp, fn = f1(y, (pj >= t).astype(int))
            print(f"    judge@{t}: F1={sc:.3f} fp={fp} fn={fn}")
        try:
            from sklearn.metrics import roc_auc_score
            print("  judge AUC", round(roc_auc_score(y, pj), 3))
        except Exception:
            pass
        wrong = sub[(pj >= 0.5).astype(int) != y]
        for r in wrong.itertuples():
            print(f"  WRONG {r.id} y={r.label} p={r.p:.3f}" if r.p is not None else f"  WRONG {r.id} y={r.label} p=None")


if __name__ == "__main__":
    main(sys.argv[1:])
