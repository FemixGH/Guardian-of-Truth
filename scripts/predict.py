#!/usr/bin/env python
"""Guardian of Truth — entry point.

    pip install .
    python scripts/predict.py --input <test.csv> --output <predictions.csv>

Pipeline: deterministic checks -> LLM judge with reasoning (vLLM, local weights in ./model)
-> combination (model/calibration.json). A global deadline guarantees the 30-minute limit:
whatever the judge has not finished in time is labelled by the deterministic fallback.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

T_START = time.time()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

import pandas as pd  # noqa: E402

from guardian import combine  # noqa: E402
from guardian.features import flags  # noqa: E402
from guardian.parse import parse_example  # noqa: E402
from guardian.rules2 import EX_TRANSFER  # noqa: E402


def log(*a):
    print(f"[guardian {time.time() - T_START:7.1f}s]", *a, flush=True)


def read_table(p):
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)


def write_table(df, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False) if str(p).endswith(".parquet") else df.to_csv(p, index=False)


def finalize(feats, votes, p_single, cfg):
    """Final labels: confident positives (hard flags / judge) + optional quota of top-scoring rows."""
    confident, scores = [], []
    for f, v, ps in zip(feats, votes, p_single):
        p_think = combine.aggregate(v, cfg) if v else None
        # no reasoning verdict (deadline / error) -> deterministic fallback decides the confident part
        conf = combine.final_label(f, p_think, cfg) == 1
        confident.append(conf)
        scores.append(combine.score(f, v, ps, cfg))
    return combine.quota_labels(scores, confident, cfg.get("quota"))


# server launch attempts: (gpu_util delta, reasoning parser, extra hybrid-cache flags).
# NB: with --enable-prefix-caching vLLM already selects mamba "align" mode for Qwen3.5/3.6 by itself; the extra
# flags (--max-num-batched-tokens 16384) caused an OOM in the GDN layers, so they stay off.
LAUNCH_PLAN = [(0.0, True, False), (-0.12, False, False)]
# with --mtp N the plan is tried first with MTP speculative decoding, then exactly as above without it


def launch_server(args, deadline, start_at=0):
    """Start vLLM following LAUNCH_PLAN from `start_at`; returns (proc, plan_index)."""
    from guardian.server import start_vllm, wait_ready, stop_vllm
    plan = [(du, rp, hyb, args.mtp) for du, rp, hyb in LAUNCH_PLAN[:1]] if args.mtp else []
    plan += [(du, rp, hyb, 0) for du, rp, hyb in LAUNCH_PLAN]
    for k in range(min(start_at, len(plan) - 1), len(plan)):
        du, rp, hyb, mtp = plan[k]
        util = round(args.gpu_util + du, 2)
        proc = start_vllm(args.model_path, args.served_name, args.port, max_model_len=args.max_model_len,
                          gpu_util=util, reasoning_parser=rp, hybrid_prefix_cache=hyb,
                          max_num_seqs=max(16, args.concurrency), mtp_tokens=mtp)
        try:
            wait_ready(args.port, timeout=max(60, deadline - time.time() - 120), proc=proc)
            log(f"vLLM ready (gpu_util={util}, reasoning_parser={rp}, hybrid_prefix_cache={hyb}, mtp={mtp})")
            return proc, k
        except Exception as e:  # noqa: BLE001
            log(f"vLLM start failed (gpu_util={util}, rp={rp}, hybrid={hyb}, mtp={mtp}): {e!r}")
            stop_vllm(proc)
    raise RuntimeError("vLLM could not start")


# transfer offers / calls (case-insensitive, RU + EN), same pattern as the rules2 transfer checks
TRANSFER_RE = EX_TRANSFER


def proc_alive(proc) -> bool:
    return proc is None or proc.poll() is None


def confirm_stage(args, cfg, df, notes, votes, judges, flagger, todo, deadline, dump):
    """Majority of 3 with early stop, at a sampling temperature, in two shared rounds:
    - confirm: rows an LLM pass flagged are re-asked with that pass; the flag is kept if the first re-sample agrees,
      else a second re-sample decides. Rows not finished in time keep their flag.
    - recall (cfg confirm.recall): rows the first pass left clean are re-asked with pass 0; if the re-sample flags, a
      second one is asked and the row is flagged only if both flag. Replaces the v2 pass (same cost, cached prefix).
    Run-to-run flips are ~13-16% of rows; majority of 3 gave +0.005..+0.014 F1 offline (6 runs)."""
    import copy
    c = cfg["confirm"]
    thr, temp = cfg["threshold"], float(c.get("temperature", 0.6))
    stop_at = deadline - float(c.get("margin_sec", 90))
    order = {i: n for n, i in enumerate(todo)}  # shortest first, as in the main stage
    # transfer turns keep their flag: re-samples call premature transfers fine (dropped 3 real / 2 clean, other rows 4 / 51)
    conf = sorted((i for i in flagger if not TRANSFER_RE.search(df["response"].iat[i])), key=lambda i: order.get(i, 0))
    rec = [i for i in todo if i not in flagger and votes[i]] if c.get("recall") else []
    jgs = {}
    for k in set(flagger.values()) | ({0} if rec else set()):
        jgs[k] = copy.copy(judges[k])
        jgs[k].temperature = temp
    who = {**{i: flagger[i] for i in conf}, **{i: 0 for i in rec}}
    samples = {i: [] for i in conf + rec}
    for rnd in (1, 2):
        need = [i for i in conf if len(samples[i]) == rnd - 1 and all(p < thr for p in samples[i])]
        need += [i for i in rec if len(samples[i]) == rnd - 1 and all(p >= thr for p in samples[i])]
        if not need or time.time() >= stop_at:
            break
        ex = ThreadPoolExecutor(args.concurrency)  # confirm rows are submitted first
        futs = {ex.submit(jgs[who[i]].score, df["prompt"].iat[i], df["response"].iat[i], notes[i],
                          args.max_ctx_chars, 1): i for i in need}
        pending = set(futs)
        while pending and time.time() < stop_at:
            done, pending = wait(pending, timeout=min(10, max(1, stop_at - time.time())), return_when=FIRST_COMPLETED)
            for fu in done:
                i = futs[fu]
                try:
                    r = fu.result()
                except Exception:  # noqa: BLE001
                    continue
                if r.get("p") is None:
                    continue
                samples[i].append(r["p"])
                if dump:
                    tag = 100 * rnd + who[i] if i in flagger else 300 + rnd  # 1xx/2xx confirm, 301/302 recall
                    dump.write(json.dumps({"id": str(df["id"].iat[i]), "pass": tag, **r}, ensure_ascii=False) + "\n")
        ex.shutdown(wait=False, cancel_futures=True)
        log(f"vote round {rnd}: {sum(1 for i in need if len(samples[i]) == rnd)}/{len(need)} re-samples done")
    dropped = [i for i in conf if len(samples[i]) == 2 and all(p < thr for p in samples[i])]
    for i in dropped:
        votes[i] = [max(samples[i])]  # 1 of 3 -> not a hallucination
    added = [i for i in rec if len(samples[i]) == 2 and all(p >= thr for p in samples[i])]
    for i in added:
        votes[i].append(min(samples[i]))  # 2 of 3 -> hallucination
    log(f"confirm: {len(conf)} LLM-flagged rows, {len(dropped)} flags dropped; recall: {len(rec)} clean rows, "
        f"{sum(1 for i in rec if samples[i])} re-sampled, {len(added)} flags added (temperature {temp})")


def run_llm_stage(args, cfg, df, feats, notes, votes, p_single, deadline):
    """Judge passes with a global deadline. Rows are processed shortest-first (max rows judged if time runs out);
    a watchdog restarts vLLM with safer flags if it dies and re-queues the unfinished rows."""
    from guardian.judge import Judge
    from guardian.server import stop_vllm
    n = len(df)
    proc, plan_k = (None, None)
    base = args.base_url
    if base is None:
        proc, plan_k = launch_server(args, deadline)
        base = f"http://127.0.0.1:{args.port}/v1"
    names = args.prompts.split(",") if args.prompts else cfg["prompts"]
    layout = cfg.get("layout", "head")
    judges = []
    for x in names:  # "prompt.md" or "prompt.md@0.7" (sampling temperature for that pass)
        fname, _, temp = x.partition("@")
        judges.append(Judge(base, args.served_name, mode="think", max_tokens=args.max_tokens,
                            timeout=max(30, deadline - time.time()), prompt_path=str(ROOT / "src" / "guardian" / fname),
                            layout=layout, temperature=float(temp) if temp else 0.0,
                            digest=bool(cfg.get("digest", False)),
                            stop_at_verdict=bool(cfg.get("stop_at_verdict", False))))
    single = None
    if cfg.get("single_prompt"):
        single = Judge(base, args.served_name, mode="single", timeout=max(30, deadline - time.time()),
                       prompt_path=str(ROOT / "src" / "guardian" / cfg["single_prompt"]), layout=layout)
    # hard-flagged rows are already 1 -> skip them; cheapest (shortest) rows first
    todo = [i for i in range(n) if combine.final_p(feats[i], 0.0, cfg) < 1.0]
    todo.sort(key=lambda i: len(df["prompt"].iat[i]) + len(df["response"].iat[i]))
    tasks = []  # (pass k, row i); -1 = single-token pass
    for i in todo:
        if single is not None:
            tasks.append((-1, i))
        tasks.append((0, i))
    for k in range(1, len(judges)):
        tasks += [(k, i) for i in todo]
    # narrow specialist checks (lowest priority): only rows routed to them, appended after the general passes
    specs = cfg.get("specialists") or {}
    if specs:
        from guardian.specialists import route
        for sname, sprompt in specs.items():
            judges.append(Judge(base, args.served_name, mode="think", max_tokens=args.max_tokens,
                                timeout=max(30, deadline - time.time()),
                                prompt_path=str(ROOT / "src" / "guardian" / sprompt), layout=layout,
                                stop_at_verdict=bool(cfg.get("stop_at_verdict", False))))
            k = len(judges) - 1
            routed = [i for i in todo if sname in route(df["prompt"].iat[i], df["response"].iat[i])]
            tasks += [(k, i) for i in routed]
            log(f"specialist '{sname}': {len(routed)} rows")
    log(f"{len(todo)} rows to judge x {len(judges)} passes (+single={single is not None}); "
        f"{n - len(todo)} decided by hard flags; layout={layout}")
    dump = open(args.dump, "w") if args.dump else None
    flagger = {}  # row -> first judge pass that flagged it (for the confirmation stage)
    finished = set()
    n_skipped = 0
    attempts = {}
    last_write = time.time()
    while time.time() < deadline:
        remaining = [t for t in tasks if t not in finished and attempts.get(t, 0) < 2]
        if not remaining:
            break
        ex = ThreadPoolExecutor(args.concurrency)
        futs = {}
        skip_flagged = cfg.get("skip_flagged", True) and cfg.get("agg", "max") == "max"

        def run(k, i, jg):
            # with OR-aggregation a row that some pass already flagged cannot change -> skip the later passes
            if k >= 1 and skip_flagged and any(v >= cfg["threshold"] for v in votes[i]):
                return {"skipped": True}
            return jg.score(df["prompt"].iat[i], df["response"].iat[i], notes[i], args.max_ctx_chars, 2)

        for k, i in remaining:
            attempts[(k, i)] = attempts.get((k, i), 0) + 1
            jg = single if k < 0 else judges[k]
            futs[ex.submit(run, k, i, jg)] = (k, i)
        pending = set(futs)
        server_died = False
        while pending and time.time() < deadline:
            done, pending = wait(pending, timeout=min(10, max(1, deadline - time.time())), return_when=FIRST_COMPLETED)
            for fu in done:
                k, i = futs[fu]
                try:
                    r = fu.result()
                except Exception as e:  # noqa: BLE001
                    log("judge error", e)
                    continue
                if r.get("skipped"):
                    finished.add((k, i))
                    n_skipped += 1
                    continue
                if r.get("p") is None:
                    continue  # failed -> stays in `remaining` for a retry round
                finished.add((k, i))
                if k < 0:
                    p_single[i] = r["p"]
                else:
                    votes[i].append(r["p"])
                    if r["p"] >= cfg["threshold"]:
                        flagger.setdefault(i, k)
                if dump:
                    dump.write(json.dumps({"id": str(df["id"].iat[i]), "pass": k, **r}, ensure_ascii=False) + "\n")
            if done:
                log(f"finished {len(finished)}/{len(tasks)} judge calls")
            if time.time() - last_write > 60:  # checkpoint
                write_table(pd.DataFrame({"id": df["id"].values, "label": finalize(feats, votes, p_single, cfg)}),
                            args.output)
                last_write = time.time()
            if proc is not None and proc.poll() is not None:
                server_died = True
                break
        ex.shutdown(wait=False, cancel_futures=True)
        if server_died:
            log(f"vLLM died (code {proc.returncode}); restarting with safer flags")
            stop_vllm(proc)
            proc, plan_k = launch_server(args, deadline, start_at=plan_k + 1)
            continue
        if not pending:
            # all submitted calls returned; loop again only to retry failed ones
            if all(t in finished or attempts.get(t, 0) >= 2 for t in tasks):
                break
            continue
        break  # deadline
    log(f"judge calls finished {len(finished)}/{len(tasks)} ({n_skipped} skipped: row already flagged); rows with >=1 verdict: "
        f"{sum(1 for v in votes if v)}/{len(todo)}")
    if cfg.get("confirm") and proc_alive(proc):
        try:
            confirm_stage(args, cfg, df, notes, votes, judges, flagger, todo, deadline, dump)
        except Exception as e:  # noqa: BLE001
            log("confirm stage failed, flags kept:", repr(e))
    if dump:
        dump.close()
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-path", default=str(ROOT / "model"))
    ap.add_argument("--calibration", default=str(ROOT / "model" / "calibration.json"))
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--base-url", default=None, help="use an already running OpenAI-compatible server")
    ap.add_argument("--served-name", default="judge")
    ap.add_argument("--budget-sec", type=float, default=float(os.environ.get("GUARDIAN_BUDGET", 26 * 60)))
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-ctx-chars", type=int, default=76000)  # ~24K tokens
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--prompts", default=None,
                    help="comma-separated judge prompts (passes); later passes only use leftover time")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--mtp", type=int, default=int(os.environ.get("GUARDIAN_MTP", 0)),
                    help="MTP speculative tokens (0 = off); falls back to plain decoding if vLLM rejects it")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--dump", default=None, help="optional jsonl with judge outputs")
    args = ap.parse_args()
    deadline = T_START + args.budget_sec

    df = read_table(args.input)
    for c in ("prompt", "response"):
        if c not in df.columns:
            df[c] = ""
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    n = len(df)
    cfg = combine.load(args.calibration)
    log(f"{n} rows; calibration={cfg}")

    feats, notes = [], []
    for p, r in zip(df["prompt"], df["response"]):
        try:
            f, nt = flags(parse_example(p, r))
        except Exception as e:  # noqa: BLE001
            log("feature error", e)
            f, nt = {}, []
        feats.append(f)
        notes.append(nt)
    # write a safe fallback immediately, so that an output exists whatever happens later
    labels = finalize(feats, [[] for _ in range(n)], [None] * n, cfg)
    write_table(pd.DataFrame({"id": df["id"].values, "label": labels}), args.output)
    log("fallback predictions written")

    votes = [[] for _ in range(n)]  # think-pass probabilities per row
    p_single = [None] * n  # single-token probability per row
    proc = None
    if not args.no_llm and n:
        try:
            proc = run_llm_stage(args, cfg, df, feats, notes, votes, p_single, deadline)
        except Exception as e:  # noqa: BLE001
            log("LLM stage failed, using fallback:", repr(e))

    labels = finalize(feats, votes, p_single, cfg)
    write_table(pd.DataFrame({"id": df["id"].values, "label": labels}), args.output)
    log(f"wrote {n} predictions -> {args.output}; positives={sum(labels)}")
    if proc is not None:
        from guardian.server import stop_vllm
        stop_vllm(proc)
    os._exit(0)  # do not wait for abandoned HTTP threads


if __name__ == "__main__":
    main()
