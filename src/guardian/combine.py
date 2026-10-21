"""Combine the LLM judge probability with deterministic flags into the final label.

Parameters live in model/calibration.json so they can be refit without code changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT = {
    "threshold": 0.5,
    # deterministic flags that are near-certain hallucinations on their own
    "hard_flags": ["n_unavailable", "n_repeat_failed", "n_placeholder", "n_schema_missing",
                   "discoverable_not_unlocked", "n_identity_swap", "policy_hard"],
    # fallback logistic model used when the LLM did not answer (time budget / error)
    "fallback": {"bias": -2.0, "w": {"tool_only": 3.0, "n_writes": 0.5, "action_claim_no_call": -0.5}},
    "llm_weight": 1.0,
    # how judge passes (different prompts) are merged: "max" = OR of verdicts, "mean" = average
    "agg": "max",
    # treat turns that consist only of tool calls as hallucinations (valid: 20/22 of them are)
    "tool_only_positive": False,
    # judge prompts = passes, in priority order (later passes only use leftover time)
    "prompts": ["judge_prompt_v2.md", "judge_prompt_v1.md"],
    # fast single-token pass over all rows first (graded score for quota ranking); None = off
    "single_prompt": "single_prompt.md",
    # label at least this fraction of rows as 1 (top by score); None = plain thresholding
    "quota": None,
    "score_w": {"hard": 10.0, "think": 3.0, "single": 1.0, "tool_only": 0.5},
}


def load(path: str | Path | None) -> dict:
    cfg = dict(DEFAULT)
    if path and Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


def aggregate(votes: list[float], cfg: dict) -> float:
    agg = cfg.get("agg", "max")
    if agg == "mean":
        return sum(votes) / len(votes)
    if agg == "vote":  # at least `min_votes` passes say 1 (capped by the number of passes that finished)
        k = min(int(cfg.get("min_votes", 2)), len(votes))
        return 1.0 if sum(v >= 0.5 for v in votes) >= k else 0.0
    return max(votes)


def _sig(x):
    return 1 / (1 + math.exp(-x))


def fallback_p(f: dict, cfg: dict) -> float:
    fb = cfg["fallback"]
    return _sig(fb["bias"] + sum(w * float(f.get(k, 0.0)) for k, w in fb["w"].items()))


def final_p(f: dict, p_llm: float | None, cfg: dict) -> float:
    if any(float(f.get(k, 0)) > 0 for k in cfg["hard_flags"]):
        return 1.0
    if cfg.get("tool_only_positive") and float(f.get("tool_only", 0)) > 0:
        return 1.0
    if p_llm is None:
        return fallback_p(f, cfg)
    return p_llm


def final_label(f: dict, p_llm: float | None, cfg: dict) -> int:
    return int(final_p(f, p_llm, cfg) >= cfg["threshold"])


def score(f: dict, think_votes: list[float], p_single: float | None, cfg: dict) -> float:
    """Continuous suspicion score used for quota labelling (higher = more likely hallucination)."""
    w = cfg.get("score_w", {"hard": 10.0, "think": 3.0, "single": 1.0, "tool_only": 0.5})
    s = 0.0
    if any(float(f.get(k, 0)) > 0 for k in cfg["hard_flags"]):
        s += w["hard"]
    if think_votes:
        s += w["think"] * aggregate(think_votes, cfg)
    s += w["single"] * (0.5 if p_single is None else p_single)
    s += w["tool_only"] * float(f.get("tool_only", 0))
    return s


def quota_labels(scores: list[float], confident: list[bool], quota: float | None) -> list[int]:
    """Label confident rows 1; if a quota (fraction of rows) is set, add the highest-scoring rows until
    at least round(quota * n) rows are positive."""
    n = len(scores)
    lab = [int(c) for c in confident]
    if quota:
        k = int(round(quota * n))
        for i in sorted(range(n), key=lambda j: -scores[j]):
            if sum(lab) >= k:
                break
            lab[i] = 1
    return lab
