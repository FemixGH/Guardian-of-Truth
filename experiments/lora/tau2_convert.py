"""Public τ²-bench result trajectories (sierra-research/tau2-bench data/tau2/results/final) -> Guardian rows.

Every assistant turn (after the greeting) of a SUCCESSFUL simulation (reward == 1) becomes a seed row with label 0
("orig"); hallucinations are injected later on the GPU (experiments/synth_llm.py). Policy text and tool schemas are
taken from the AgentProcessBench trajectories of the same domain (same τ² environments). Tasks that appear in
AgentProcessBench are excluded completely, so nothing overlaps with the APB evaluation rows; valid.parquet dialogues
are a different (translated) source and are not used.

python experiments/lora/tau2_convert.py --n-seeds 1500
"""
import argparse
import ast
import glob
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from apb_convert import INSTRUCTIONS, _args, tools_block  # noqa: E402

APB = ROOT.parent / "data_ext/agentprocessbench/tau2/test.jsonl"
RES = ROOT.parent / "data_ext/tau2_results"


def _lit(x):
    if isinstance(x, str):
        if x == "None":
            return None
        try:
            return ast.literal_eval(x)
        except Exception:  # noqa: BLE001
            return x
    return x


def domain_env():
    env, apb_tasks = {}, {}
    for line in open(APB):
        t = json.loads(line)
        dom = t["data_source"].replace("tau2_", "")
        sysm = next(m for m in t["messages"] if m["role"] == "system")["content"].strip()
        env.setdefault(dom, (sysm, t["tools"]))
        gt = _lit(t["ground_truth"]) or {}
        for a in (gt.get("actions") or []) if isinstance(gt, dict) else []:
            apb_tasks.setdefault(dom, set()).add(str(a.get("action_id", "")).split("_")[0])
        apb_tasks.setdefault(dom + "_text", set()).add(" ".join(str(t["question"]).split()))
        apb_tasks.setdefault(dom + "_gt", set()).add(gt_key(gt))
    return env, apb_tasks


def gt_key(gt) -> str:
    """Identity of a task by its reference actions + environment assertions."""
    if not isinstance(gt, dict):
        return ""
    acts = [(a.get("requestor"), a.get("name"), json.dumps(a.get("arguments"), sort_keys=True)) for a in gt.get("actions") or []]
    env = [(e.get("func_name"), json.dumps(e.get("arguments"), sort_keys=True)) for e in gt.get("env_assertions") or []]
    return json.dumps([acts, env])


def in_apb(task, dom, apb_tasks):
    if dom != "telecom" and str(task["id"]) in apb_tasks.get(dom, set()):
        return True
    k = gt_key(_lit(task.get("evaluation_criteria")))
    if k and k != json.dumps([[], []]) and k in apb_tasks.get(dom + "_gt", set()):
        return True
    ins = (task.get("user_scenario") or {}).get("instructions") or {}
    reason = " ".join(str(ins.get("reason_for_call") or "").split()) if isinstance(ins, dict) else ""
    return bool(reason) and any(reason[:300] in q for q in apb_tasks.get(dom + "_text", ()))


def tool_lines(calls, results, prefix="\t→ TOOL_CALL"):
    lines, res = [], []
    for tc in calls or []:
        name, args = tc.get("name") or tc.get("function", {}).get("name"), tc.get("arguments")
        lines.append(f"{prefix} {name}: {_args(args)}")
        r = results.get(tc.get("id"))
        if r is not None:
            text = str(r.get("content") or "")
            err = " [ERROR]" if _lit(r.get("error")) in (True, "True") or text.lower().startswith(("error", '{"error"')) else ""
            body = text.split("\n")
            res.append(f"\t← TOOL_RESPONSE {name}{err}: {body[0]}")
            res += ["\t  " + b for b in body[1:]]
    return lines, res


def convert(sim, dom, sysm, tools, src):
    msgs = sim["messages"]
    results = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
    blocks = [f"⟦SYSTEM⟧\n{INSTRUCTIONS}\n<policy>\n{sysm}\n</policy>\n\n[AVAILABLE TOOLS]\n{tools_block(tools)}\n\n"]
    rows, turn = [], 0
    tid = f"t2_{dom}_{sim['task_id']}_{src}_{sim.get('trial', 0)}"
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = (_lit(m.get("content")) or "")
        content = content.strip() if isinstance(content, str) else str(content)
        calls = _lit(m.get("tool_calls"))
        if role == "user":
            if content.startswith("###"):
                continue
            cl, rl = tool_lines(calls, results)
            blocks.append("\n".join(["⟦USER⟧"] + ([content] if content else []) + cl + rl) + "\n\n")
        elif role == "assistant":
            if turn == 0 and i <= 1 and not calls:
                blocks.append(f"⟦ASSISTANT⟧\n{content}\n\n")
                turn = 0.5
                continue
            turn = int(turn) + 1
            head = f"⟦ASSISTANT · ход {turn}⟧"
            cl, rl = tool_lines(calls, results)
            response = "\n".join([head] + ([content] if content else []) + cl) + "\n"
            rows.append(dict(id=f"{tid}::t{turn}", traj=tid, task=f"{dom}_{sim['task_id']}", domain=dom,
                             prompt="".join(blocks), response=response, label=0, mtype="orig",
                             src=f"{tid}::t{turn}", n_calls=len(calls or []), has_text=bool(content)))
            blocks.append("\n".join([head] + ([content] if content else []) + cl + rl) + "\n\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=1500)
    ap.add_argument("--max-prompt-chars", type=int, default=90000)
    ap.add_argument("--out", default=str(ROOT / "runs/lora/tau2_seeds_en.parquet"))
    a = ap.parse_args()
    env, apb_tasks = domain_env()
    rows, stats = [], {}
    for f in sorted(glob.glob(str(RES / "*.json"))):
        name = Path(f).name
        if "_default_" not in name and "_base_" not in name:
            continue  # telecom-workflow / no-user / op variants use different policies or agent capabilities
        dom = next((d for d in ("airline", "retail", "telecom") if f"_{d}_" in name), None)
        if dom is None or dom not in env:
            continue
        d = json.load(open(f))
        excluded = {str(t["id"]) for t in d["tasks"] if in_apb(t, dom, apb_tasks)}
        stats[f"apb_tasks_{dom}"] = len(excluded)
        src = name.split("_")[0].replace("-2025", "").replace("-", "")[:14]
        for sim in d["simulations"]:
            rw = _lit(sim.get("reward_info")) or {}
            if not isinstance(rw, dict) or float(rw.get("reward", 0)) < 1.0:
                continue
            if str(sim["task_id"]) in excluded:
                stats["skip_apb_task"] = stats.get("skip_apb_task", 0) + 1
                continue
            sysm, tools = env[dom]
            rows += convert(sim, dom, sysm, tools, src)
    df = pd.DataFrame(rows)
    df = df[df.prompt.str.len() <= a.max_prompt_chars]
    print("all candidate turns", len(df), df.domain.value_counts().to_dict(), stats)
    # sample seeds: at most 2 turns per trajectory, balanced over domains, prefer turns with text (more error types)
    random.seed(11)
    parts = []
    for dom, g in df.groupby("domain"):
        g = g.sample(frac=1, random_state=11).groupby("traj").head(2)
        parts.append(g.head(a.n_seeds // 3))
    out = pd.concat(parts).reset_index(drop=True)
    out.to_parquet(a.out, index=False)
    print("seeds", len(out), out.domain.value_counts().to_dict(), "tasks", out.task.nunique(),
          "with text", int(out.has_text.sum()), "with calls", int((out.n_calls > 0).sum()),
          "prompt chars median", int(out.prompt.str.len().median()))


if __name__ == "__main__":
    main()
