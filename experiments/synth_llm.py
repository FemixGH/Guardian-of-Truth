"""LLM-generated semantic perturbations of clean agent turns (the error types rules cannot catch).

For every consensus-clean turn and every applicable error type, the generator model rewrites the agent turn so that
it contains exactly ONE hallucination of that type (label 1), plus one meaning-preserving paraphrase (label 0).
The full dialogue context is given, so the injected error is plausible for this conversation.

python experiments/synth_llm.py --base-url http://127.0.0.1:8000/v1 --out runs/synth_llm.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian.judge import fit_context  # noqa: E402
from guardian.parse import parse_example  # noqa: E402

TYPES = {
    "false_refusal": ("text", "The agent claims it cannot do something (or that the policy/system does not allow it) "
                      "although an available tool or policy path clearly allows it in this situation."),
    "premature_transfer": ("any", "Replace the turn by a transfer to a human agent (call transfer_to_human_agents with "
                           "a plausible summary in the same style) although obvious available steps were not tried "
                           "and the policy conditions for a transfer are not met."),
    "fabricated_fact": ("text", "Add ONE concrete but unsupported detail: a fee, price, deadline, condition, policy "
                        "rule, option or timeline that appears nowhere in the context."),
    "claimed_action": ("text", "The agent states that it has already performed an action (cancelled, refunded, "
                       "changed, sent, unblocked …) although no such tool call was made."),
    "asks_known_info": ("text", "The agent asks the user for a piece of information that is already present in the "
                        "context (in the user's messages or in a tool result)."),
    "contradiction": ("text", "Change ONE stated fact so that it contradicts the tool results or user messages "
                      "(a status, name, class, date, amount, count, or which item/reservation it refers to)."),
    "policy_violation": ("call", "Change the tool call so that it violates a policy precondition visible in the "
                         "context (e.g. acting on the wrong/ineligible object, skipping required user confirmation, "
                         "wrong option or amount), while keeping a plausible format."),
}
GEN_SYSTEM = """You create training data for a hallucination detector of customer-service LLM agents.
You receive the dialogue context the agent saw and the agent's ORIGINAL turn (which is correct).
Rewrite the turn so that it contains EXACTLY ONE error of the requested type and is otherwise as close as possible
to the original: same language (Russian text, English tool names and JSON), same tone and length, same transcript
format (first line `⟦ASSISTANT · ход N⟧`, tool calls as lines `\t→ TOOL_CALL name: {json}`).
The error must be realistic (something an LLM agent would plausibly do) and detectable by a careful reader of the
context. Do not add any comments or marks about the error.
Reply with the rewritten turn only, between <turn> and </turn>, then one line `ERROR: <short description>`."""
PARA_SYSTEM = """You create training data for a hallucination detector of customer-service LLM agents.
You receive the dialogue context and the agent's ORIGINAL turn (which is correct). Rewrite it so that the meaning,
every fact, number, id and every tool call stay EXACTLY the same, but the wording of the text is paraphrased
(different phrasing, maybe different order of sentences). Tool call lines must be copied unchanged.
Keep the transcript format (first line `⟦ASSISTANT · ход N⟧`). If the turn has no text, return it unchanged.
Reply with the rewritten turn only, between <turn> and </turn>."""


def parse_turn(txt: str, original: str) -> str | None:
    """Rewritten turn from the generator reply: text between the LAST <turn> ... </turn> pair, nested tags and stray
    `ERROR:` lines removed, transcript header restored if missing."""
    m = re.findall(r"<turn>(.*?)</turn>", txt, re.S)
    if m:
        body = m[-1]
    else:  # the model sometimes answers with the bare turn (header line ... then `ERROR:`), without the tags
        i = txt.find("⟦ASSISTANT")
        if i < 0:
            return None
        body = txt[i:]
    new = re.sub(r"</?turn>", "", body)
    new = "\n".join(l for l in new.split("\n") if not l.strip().startswith("ERROR:")).strip("\n")
    if not new.strip():
        return None
    if not new.lstrip().startswith("⟦"):
        new = original.split("\n")[0] + "\n" + new
    return new.rstrip("\n") + "\n"


def applicable(resp_text: str, n_calls: int, kind: str) -> bool:
    return {"text": bool(resp_text.strip()), "call": n_calls > 0, "any": True}[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--src", default=str(ROOT / "runs" / "synth_det.parquet"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "synth_llm.parquet"))
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--max-ctx-chars", type=int, default=60000)
    ap.add_argument("--per-seed", type=int, default=0, help="random error types per seed (0 = all applicable)")
    ap.add_argument("--para-frac", type=float, default=1.0, help="share of seeds that also get a paraphrase negative")
    ap.add_argument("--inject-frac", type=float, default=1.0, help="share of seeds that get injected errors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    args = ap.parse_args()
    import random
    rnd = random.Random(args.seed)
    from openai import OpenAI
    cli = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1200)
    src = pd.read_parquet(args.src)
    orig = src[src.mtype == "orig"]
    jobs = []
    for r in orig.itertuples(index=False):
        ex = parse_example(r.prompt, r.response)
        types = [(t, desc) for t, (kind, desc) in TYPES.items()
                 if applicable(ex.resp_text or "", len(ex.resp_calls), kind)]
        if args.per_seed:
            types = rnd.sample(types, min(args.per_seed, len(types))) if rnd.random() < args.inject_frac else []
        jobs += [(r, t, desc) for t, desc in types]
        if (ex.resp_text or "").strip() and rnd.random() < args.para_frac:
            jobs.append((r, "neg_paraphrase", None))

    from collections import Counter
    why = Counter()

    def run(job):
        r, t, desc = job
        ctx = fit_context(r.prompt, args.max_ctx_chars) if len(r.prompt) > args.max_ctx_chars else r.prompt
        sysm = PARA_SYSTEM if desc is None else GEN_SYSTEM
        user = f"<context>\n{ctx}\n</context>\n\n<original_turn>\n{r.response}\n</original_turn>\n"
        if desc:
            user += f"\nError type to inject: {t} — {desc}\n"
        try:
            out = cli.chat.completions.create(
                model=args.model, messages=[{"role": "system", "content": sysm}, {"role": "user", "content": user}],
                temperature=0.7, max_tokens=args.max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
            txt = out.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            why["error:" + type(e).__name__] += 1
            if why["error:" + type(e).__name__] == 1:
                print("first error:", repr(e)[:300], flush=True)
            return None
        new = parse_turn(txt, r.response)
        if new is None:
            why["no_turn_tag"] += 1
            if why["no_turn_tag"] == 1:
                print("first reply without <turn>:", repr(txt[:300]), "finish:", out.choices[0].finish_reason, flush=True)
            return None
        err = re.search(r"ERROR:\s*(.*)", txt.split("</turn>")[-1]) or re.search(r"ERROR:\s*(.*)", txt)
        if new.strip() == r.response.strip():
            why["unchanged"] += 1
            return None
        why["ok"] += 1
        extra = {k: getattr(r, k) for k in ("domain", "task") if hasattr(r, k)}
        return dict(**extra, id=f"{r.src}#llm_{t}", src=r.src, traj=r.traj, prompt=r.prompt, response=new,
                    label=0 if desc is None else 1, mtype=("neg_paraphrase" if desc is None else t),
                    note=err.group(1).strip() if err else "")

    rows = []
    with ThreadPoolExecutor(args.concurrency) as ex:
        for i, res in enumerate(ex.map(run, jobs)):
            if res:
                rows.append(res)
            if i % 100 == 0:
                print(i, "/", len(jobs), flush=True)
    print("outcomes:", dict(why), flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no rows generated from {len(jobs)} jobs")
    df.to_parquet(args.out, index=False)
    print(len(jobs), "jobs ->", len(df), "rows")
    print(df.groupby(["label", "mtype"]).size().to_string())
    json.dump({"jobs": len(jobs), "rows": len(df)}, open(args.out + ".json", "w"))


if __name__ == "__main__":
    main()
