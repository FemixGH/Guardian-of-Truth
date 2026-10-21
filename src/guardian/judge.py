"""LLM judge over an OpenAI-compatible endpoint (vLLM on the A100, llama-server locally)."""
from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path

PROMPT_PATH = Path(__file__).with_name("judge_prompt.md")
VERDICT_RE = re.compile(r"VERDICT\s*[:：]\s*\**\s*([01])", re.I)  # stop strings include "Verdict: X"


SHARED_SYSTEM = ("You are an expert auditor of customer-service LLM agents. You check single agent turns for "
                 "contextual hallucinations. Follow the audit instructions given at the end of the user message.")


def build_messages(prompt: str, response: str, notes: list[str] | None = None, mode: str = "think",
                   max_ctx_chars: int | None = None, prompt_path: Path | None = None,
                   layout: str = "tail", digest: bool = False) -> list[dict]:
    """layout="tail": shared short system prompt, context first, pass-specific instructions LAST, so that all passes
    over the same row share one long KV-cache prefix (vLLM prefix caching). layout="head": instructions in system."""
    instructions = Path(prompt_path or PROMPT_PATH).read_text(encoding="utf-8")
    if "{DOMAIN_RULES}" in instructions:
        instructions = instructions.replace("{DOMAIN_RULES}", domain_rules(prompt or ""))
    ctx = prompt or ""
    if max_ctx_chars and len(ctx) > max_ctx_chars:
        ctx = fit_context(ctx, max_ctx_chars, response)
    user = f"<context>\n{ctx}\n</context>\n\n<turn_under_review>\n{response or ''}\n</turn_under_review>\n"
    if digest:
        from .digest import turn_digest
        d = turn_digest(prompt or "", response or "")
        if d:
            user += ("\n<turn_digest>\n(deterministic summary of the turn under review and its immediate preconditions; "
                     "the full context above remains the ground truth)\n" + d + "\n</turn_digest>\n")
    if notes:
        user += ("\nAutomatic checker notes (deterministic, may contain false alarms — verify each one, they are "
                 "hints, not verdicts):\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    if mode == "think":
        ask = "\nAudit the turn under review now. End with `VERDICT: 0` or `VERDICT: 1`."
    else:
        ask = "\nAnswer with a single character: 1 if the turn under review contains a hallucination, otherwise 0."
    if layout == "tail":
        user += "\n=== AUDIT INSTRUCTIONS ===\n" + instructions + "\n" + ask
        return [{"role": "system", "content": SHARED_SYSTEM}, {"role": "user", "content": user}]
    return [{"role": "system", "content": instructions}, {"role": "user", "content": user + ask}]


DOMAIN_MARKERS = [("airline", "# Airline Agent Policy"), ("retail", "# Retail agent policy"),
                  ("telecom", "# Telecom Agent Policy"), ("banking", "Rho-Bank")]


def domain_rules(prompt: str) -> str:
    """Domain-specific checklist (judge_domain_<domain>.md) for the agent policy found in the context."""
    head = prompt[:30000]
    for name, marker in DOMAIN_MARKERS:
        if marker.lower() in head.lower():
            f = Path(__file__).with_name(f"judge_domain_{name}.md")
            return f.read_text(encoding="utf-8").strip() if f.exists() else ""
    return ""


CRITIC_PATH = Path(__file__).with_name("critic_prompt.md")
CATEGORY_RE = re.compile(r"CATEGORY\s*[:：]\s*\**\s*([A-Z_]+)")
REAL_CATEGORIES = {"TOOL_NOT_AVAILABLE", "WRONG_ARGUMENT", "POLICY_VIOLATION", "UNCONFIRMED_WRITE", "WRONG_COMPUTATION",
                   "FABRICATED_ACTION", "INVENTED_CONTENT", "FALSE_REFUSAL", "ASKS_KNOWN_INFO"}
NITPICK_CATEGORIES = {"PARALLEL_CALLS", "ORDINARY_LOOKUP", "PROCEDURAL_QUESTION", "GENERIC_EXPLANATION",
                      "USER_INSISTED_TRANSFER", "CARRIED_OVER", "COSMETIC", "FINDING_WRONG"}
_FILLER = re.compile(r"(?i)^\s*(\[?(output|final|done|proceed|ready|all good|matches|structure|wait, the prompt|i will|self-correction|"
                     r"note:|check format|let'?s produce|\(note|\(self|verdict))")


def finding_of(reasoning: str, text: str = "", max_chars: int = 2500) -> str:
    """Condense a flagging judge's reasoning into its finding: drop the formatting/meta loops, keep the last
    substantive lines."""
    src = (reasoning or "") + "\n" + (text or "")
    lines = [l.strip() for l in src.split("\n") if l.strip() and not _FILLER.match(l.strip()) and len(l.strip()) > 25]
    out, total = [], 0
    for l in reversed(lines):
        if total + len(l) > max_chars:
            break
        out.append(l)
        total += len(l) + 1
    return "\n".join(reversed(out))


def critic_messages(prompt: str, response: str, finding: str, max_ctx_chars: int | None = None,
                    critic_path: Path | None = None) -> list[dict]:
    ctx = prompt or ""
    if max_ctx_chars and len(ctx) > max_ctx_chars:
        ctx = fit_context(ctx, max_ctx_chars)
    user = (f"<context>\n{ctx}\n</context>\n\n<turn_under_review>\n{response or ''}\n</turn_under_review>\n\n"
            f"<auditor_finding>\n{finding}\n</auditor_finding>\n\n"
            + ("Classify the finding now. End with `CATEGORY: <NAME>`." if critic_path and "CATEGORY" in
               Path(critic_path).read_text(encoding="utf-8") else
               "Verify the finding now. End with `VERDICT: 1` (confirmed) or `VERDICT: 0` (rejected)."))
    return [{"role": "system", "content": Path(critic_path or CRITIC_PATH).read_text(encoding="utf-8")},
            {"role": "user", "content": user}]


STOP_VERDICT = ["VERDICT: 0", "VERDICT: 1", "Verdict: 0", "Verdict: 1"]  # the API accepts at most 4
CTX_MODE = os.environ.get("GUARDIAN_CTX_MODE", "compress")  # "compress" | "truncate"


ANCHOR_RE = re.compile(r"\$\s?\d[\d,.]*|\b\d{3,}(?:[.,]\d+)?\b|\b\d+[.,]\d+\b|\b[A-Z][A-Z_]{3,}\b|"
                       r"\b\w*\d\w*_\w+\b|\b\w+_\w*\d\w*\b")
ANCHOR_STOP = {"ASSISTANT", "USER", "SYSTEM", "TOOL_CALL", "TOOL_RESPONSE", "TOOL_RESULT", "ERROR", "VERDICT"}
EVIDENCE_CHARS = int(os.environ.get("GUARDIAN_EVIDENCE_CHARS", "1500"))  # per shortened tool output


def anchors_of(text: str | None) -> set[str]:
    """Values the turn under review states (amounts, long numbers, CAPS enums, ids): the evidence for them must
    survive context compression, otherwise the judge flags correct values as unsupported."""
    return {a.replace(" ", "") for a in ANCHOR_RE.findall(text or "")} - ANCHOR_STOP


def _evidence(cut: list[str], anchors: set[str] | None) -> list[str]:
    """Lines removed by shortening that contain values of the turn under review; lines with more distinct values
    first, bounded by EVIDENCE_CHARS, original order kept."""
    if not anchors:
        return []
    pats = [re.compile(r"(?<![\w$])" + re.escape(a) + r"(?![\d\w])") for a in anchors]
    hits = []
    for k, ln in enumerate(cut):
        flat = ln.replace(" ", "")
        m = sum(1 for pt in pats if pt.search(flat) or pt.search(ln))
        if m:
            hits.append((-m, k))
    keep, n = {}, 0
    for _, k in sorted(hits):
        ln = cut[k]
        if len(ln) > 400:  # very long line (a whole KB paragraph): keep a window around the first value it contains
            pos = min((m.start() for pt in pats for m in [pt.search(ln)] if m), default=0)
            ln = "\t  …" + ln[max(0, pos - 200):pos + 200] + "…"
        if n + len(ln) <= EVIDENCE_CHARS:
            keep[k] = ln
            n += len(ln) + 1
    if not keep:
        return []
    return ["\t  …[lines below kept: they contain values stated in the turn under review]…"] + [keep[k] for k in sorted(keep)]


def _shrink_block(lines: list[str], cap: int, anchors: set[str] | None = None) -> list[str]:
    """Shorten one TOOL_RESPONSE block to ~cap chars. Knowledge-base hits keep every document's header lines
    (title, ID) and the start of its content; anything else keeps its head. Cut lines that contain values stated
    in the turn under review are kept as evidence."""
    text = "\n".join(lines)
    if len(text) <= cap:
        return lines
    docs = [k for k, ln in enumerate(lines[:-1]) if re.search(r"\d+\. ", ln) and lines[k + 1].strip().startswith("ID:")]
    out, used = [], 0
    if len(docs) >= 2:
        per = max(120, cap // len(docs))
        bounds = docs + [len(lines)]
        out = lines[:docs[0]]
        cut = []
        for a, b in zip(bounds, bounds[1:]):
            chunk, n = [], 0
            for k, ln in enumerate(lines[a:b]):
                if n + len(ln) > per and chunk:
                    chunk.append("\t  …[document shortened]…")
                    cut += lines[a + k:b]
                    break
                chunk.append(ln)
                n += len(ln) + 1
            out += chunk
        return out + _evidence(cut, anchors)
    for k, ln in enumerate(lines):
        if used + len(ln) > cap:
            out.append(f"\t  …[tool output shortened: {len(text) - used} more chars]…")
            return out + _evidence(lines[k:], anchors)
        out.append(ln)
        used += len(ln) + 1
    return out


def compress_context(ctx: str, max_chars: int, focus_text: str | None = None) -> str:
    """Fit the context by shortening long tool outputs (oldest first, progressively harder) while keeping every
    user / assistant message and every tool call intact. The tool outputs of the latest assistant turn are
    shortened last."""
    lines = ctx.split("\n")
    blocks = []  # (start, end) of TOOL_RESPONSE blocks
    k = 0
    while k < len(lines):
        if lines[k].startswith("\t← TOOL_RESPONSE"):  # valid.parquet format: continuation lines are "\t  ..."
            e = k + 1
            while e < len(lines) and lines[e].startswith("\t  "):
                e += 1
            blocks.append((k, e))
            k = e
        elif lines[k].startswith("⟦TOOL_RESULT"):  # task-page format: the result runs until the next ⟦…⟧ marker
            e = k + 1
            while e < len(lines) and not lines[e].startswith("⟦"):
                e += 1
            blocks.append((k, e))
            k = e
        else:
            k += 1
    if not blocks:
        return ctx
    anchors = anchors_of(focus_text)
    last_turn = max((k for k, ln in enumerate(lines) if ln.startswith("⟦ASSISTANT")), default=0)
    old = [b for b in blocks if b[0] < last_turn]
    recent = [b for b in blocks if b[0] >= last_turn]
    cur = {b: lines[b[0]:b[1]] for b in blocks}

    def total():
        return len(ctx) - sum(len("\n".join(lines[a:e])) - len("\n".join(cur[(a, e)])) for a, e in blocks)

    for group in (old, recent):
        for cap in (6000, 3000, 1500, 700, 300, 120):
            if total() <= max_chars:
                break
            for b in group:  # oldest first, stop as soon as it fits
                if total() <= max_chars:
                    break
                cur[b] = _shrink_block(cur[b], cap, anchors)
    out, k, bi = [], 0, 0
    starts = {a: (a, e) for a, e in blocks}
    while k < len(lines):
        if k in starts:
            b = starts[k]
            out += cur[b]
            k = b[1]
        else:
            out.append(lines[k])
            k += 1
    res = "\n".join(out)
    return res if len(res) <= max_chars else truncate_context(res, max_chars)


def fit_context(ctx: str, max_chars: int, focus_text: str | None = None) -> str:
    return compress_context(ctx, max_chars, focus_text) if CTX_MODE == "compress" else truncate_context(ctx, max_chars)


def truncate_context(ctx: str, max_chars: int) -> str:
    """Keep the system prompt head (policy + tools) and the most recent part of the dialogue."""
    i = ctx.find("⟦USER⟧")
    head = ctx[:i] if i > 0 else ctx[: max_chars // 2]
    head_budget = min(len(head), int(max_chars * 0.55))
    tail_budget = max_chars - head_budget
    return head[:head_budget] + "\n…[truncated]…\n" + ctx[-tail_budget:]


def _p_from_top(top: list[tuple[str, float]]) -> float | None:
    l1 = max((lp for t, lp in top if t.strip() == "1"), default=None)
    l0 = max((lp for t, lp in top if t.strip() == "0"), default=None)
    if l1 is None and l0 is None:
        return None
    if l0 is None:
        return 1.0
    if l1 is None:
        return 0.0
    m = max(l1, l0)
    return math.exp(l1 - m) / (math.exp(l1 - m) + math.exp(l0 - m))


class Judge:
    def __init__(self, base_url: str, model: str, mode: str = "think", max_tokens: int = 4096,
                 enable_thinking: bool = True, temperature: float = 0.0, timeout: float = 1800,
                 prompt_path: str | None = None, layout: str = "tail", digest: bool = False,
                 stop_at_verdict: bool = False):
        # stop generating as soon as a verdict is written: the reasoning otherwise re-checks the output format
        # for ~40% more tokens and changes the verdict in <1% of rows
        self.stop = STOP_VERDICT if stop_at_verdict else None
        self.prompt_path = prompt_path
        self.layout = layout
        self.digest = digest
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        self.mode = mode
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.temperature = temperature

    @staticmethod
    def structured_decision(r: dict) -> dict:
        """Prompts with an `ISSUES: [...]` list (v5): the label is computed from the issues (real type, new,
        HIGH/MEDIUM impact) instead of trusting the model's own VERDICT line; falls back to VERDICT."""
        text = (r.get("text") or "")
        i = text.rfind("ISSUES:")
        if i < 0:
            return r
        blob = text[i + 7:]
        j = blob.find("VERDICT")
        blob = (blob[:j] if j >= 0 else blob).strip()
        try:
            import json as _json
            issues = _json.loads(blob)
        except Exception:  # noqa: BLE001
            r["issues_parse_error"] = True
            return r
        if not isinstance(issues, list):
            return r
        real = [x for x in issues if isinstance(x, dict) and str(x.get("type", "")).upper() in REAL_CATEGORIES
                and x.get("new", True) not in (False, "false", "False")
                and str(x.get("impact", "HIGH")).upper() in ("HIGH", "MEDIUM")]
        r["issues"] = issues
        r["verdict_model"] = r.get("verdict")
        r["verdict"] = int(bool(real))
        r["p"] = 0.97 if real else 0.03
        return r

    def score(self, prompt: str, response: str, notes=None, max_ctx_chars=None, retries: int = 3) -> dict:
        msgs = build_messages(prompt, response, notes, self.mode, max_ctx_chars, self.prompt_path, self.layout,
                              self.digest)
        last = None
        for a in range(retries):
            try:
                return self.structured_decision(self._score(msgs))
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** a)
        return {"p": None, "verdict": None, "text": f"ERROR {last}", "reasoning": ""}

    def critic(self, prompt: str, response: str, finding: str, max_ctx_chars=None, retries: int = 2,
               critic_path: str | None = None) -> dict:
        """Second opinion on a flagged turn: p = probability that the flag is confirmed. A taxonomy critic prompt
        (`CATEGORY: X`) is mapped to confirmed / rejected by code."""
        msgs = critic_messages(prompt, response, finding, max_ctx_chars, Path(critic_path) if critic_path else None)
        last = None
        for a in range(retries):
            try:
                r = self._score(msgs)
                m = CATEGORY_RE.findall((r.get("text") or "") + "\n" + (r.get("reasoning") or "")[-400:])
                if m:
                    cat = m[-1]
                    r["category"] = cat
                    r["p"] = 0.0 if cat in NITPICK_CATEGORIES else 1.0
                    r["verdict"] = int(r["p"])
                return r
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** a)
        return {"p": None, "verdict": None, "text": f"ERROR {last}", "reasoning": ""}

    def _score(self, msgs) -> dict:
        extra = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if self.mode == "think":
            import openai
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens,
                    logprobs=True, top_logprobs=10, stop=self.stop,
                    extra_body={**extra, **({"include_stop_str_in_output": True} if self.stop else {})})
            except openai.BadRequestError as e:
                if not self.stop:
                    raise
                # retry this call without stop strings; switch early stop off for later calls only if the server
                # rejected the stop strings themselves (any other bad request must not cost every later call ~40%)
                if "stop" in str(e).lower():
                    self.stop = None
                r = self.client.chat.completions.create(
                    model=self.model, messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens,
                    logprobs=True, top_logprobs=10, extra_body=extra)
            ch = r.choices[0]
            text = ch.message.content or ""
            reasoning = getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None) or ""
            p = None
            # probability of the verdict token: find the digit right after the last "VERDICT"
            lp = ch.logprobs.content if ch.logprobs and ch.logprobs.content else []
            if lp:
                acc = ""
                idx_v = None
                for k, t in enumerate(lp):
                    acc += t.token
                    if acc.upper().rfind("VERDICT") >= 0 and t.token.strip() in ("0", "1") \
                            and acc[:-len(t.token)].rstrip().endswith((":", "*", "：")):
                        idx_v = k  # keep the LAST verdict digit (the model may echo the format earlier)
                if idx_v is not None:
                    p = _p_from_top([(x.token, x.logprob) for x in lp[idx_v].top_logprobs])
            m = VERDICT_RE.findall(text) or VERDICT_RE.findall(reasoning)
            verdict = int(m[-1]) if m else None
            if verdict is not None and (p is None or int(p >= 0.5) != verdict):
                p = 0.97 if verdict == 1 else 0.03
            forced = False
            if p is None:  # reasoning budget exhausted -> force a verdict from the partial analysis
                p = self._force_verdict(msgs, (reasoning + "\n" + text).strip())
                forced = True
            return {"p": p, "verdict": verdict, "text": text, "reasoning": reasoning, "forced": forced,
                    "finish_reason": ch.finish_reason,
                    "completion_tokens": r.usage.completion_tokens if r.usage else None,
                    "prompt_tokens": r.usage.prompt_tokens if r.usage else None}
        r = self.client.chat.completions.create(
            model=self.model, messages=msgs, temperature=0.0, max_tokens=2, logprobs=True, top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        lp = r.choices[0].logprobs.content
        p = _p_from_top([(x.token, x.logprob) for x in lp[0].top_logprobs]) if lp else None
        return {"p": p, "verdict": None if p is None else int(p >= 0.5), "text": r.choices[0].message.content,
                "reasoning": "", "prompt_tokens": r.usage.prompt_tokens if r.usage else None}

    def _force_verdict(self, msgs, partial: str) -> float | None:
        tail = partial[-6000:]
        m2 = msgs + [{"role": "assistant", "content": "My analysis so far:\n" + tail},
                     {"role": "user", "content": "Stop the analysis. Based on it, answer with a single character: "
                                                 "1 if the turn under review contains a hallucination, otherwise 0."}]
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=m2, temperature=0.0, max_tokens=2, logprobs=True, top_logprobs=20,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        except Exception:  # noqa: BLE001
            return None
        lp = r.choices[0].logprobs.content if r.choices[0].logprobs else None
        return _p_from_top([(x.token, x.logprob) for x in lp[0].top_logprobs]) if lp else None
