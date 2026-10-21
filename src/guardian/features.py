"""Deterministic, model-free signals of contextual hallucination.

Each check returns a numeric feature; `flags()` also returns human-readable findings
that are injected into the LLM judge prompt as hints ("checker notes").
"""
from __future__ import annotations

import json
import re

from .parse import Example, ToolCall

WRITE_PREFIXES = ("book_", "cancel_", "update_", "modify_", "return_", "exchange_", "resume_",
                  "suspend_", "close_", "send_", "make_", "apply_", "change_", "refuel_", "enable_",
                  "disable_", "approve_", "log_", "toggle_", "set_", "order_", "file_", "freeze_",
                  "unfreeze_", "dispute_", "transfer_funds", "pay_")
DB_WRITE_PREFIXES = ("book_", "cancel_", "update_", "modify_", "return_", "exchange_", "resume_", "suspend_",
                     "close_", "order_", "apply_", "change_", "approve_", "refuel_", "enable_roaming", "send_certificate",
                     "file_", "freeze_", "unfreeze_", "dispute_", "transfer_funds", "make_payment", "pay_")
CONFIRM_RE = re.compile(r"\b(да|yes|подтверждаю|согласен|согласна|давайте|confirm|proceed|go ahead|верно|хорошо|ок|okay)\b",
                        re.I)
PLACEHOLDER_RE = re.compile(r"placeholder|<[^>]{1,30}>|\bunknown\b|\bxxx+\b|\bTBD\b|example_id|your_|\.\.\.|\bN/A\b",
                            re.I)
ACTION_CLAIM_RE = re.compile(
    r"(отправ(ил|лен|лено)|выполн(ил|яю|ено)|оформ(ил|лен|лено)|отмен(ил|ено|ена)|измен(ил|ено|ена)|"
    r"сброс(ил)?|перезагруз|провер(ил|яю)|обнов(ил|лено)|возврат оформлен|заказ(ал|ан)|"
    r"успешно|готово|I have|I've|has been|was successfully)", re.I)
NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
ID_RE = re.compile(r"\b(?=[A-Za-z_#]*\d)[A-Za-z0-9_#\-]{5,}\b")


def _flat_values(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _flat_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flat_values(v, out)
    elif obj is not None:
        out.append(obj)
    return out


def _canon(c: ToolCall) -> str:
    if c.args is not None:
        try:
            return c.name + json.dumps(c.args, sort_keys=True, ensure_ascii=False)
        except Exception:
            pass
    return c.name + re.sub(r"\s+", "", c.raw_args)


def _norm_num(s: str) -> float | None:
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def schema_issues(ex: Example, c: ToolCall) -> list[str]:
    sch = ex.tools.get(c.name)
    if sch is None or not isinstance(c.args, dict):
        return []
    out = []
    for k in c.args:
        if sch.params and k not in sch.params:
            out.append(f"argument `{k}` is not in the schema of `{c.name}`")
    for k, req in sch.params.items():
        if req and k not in c.args:
            out.append(f"required argument `{k}` of `{c.name}` is missing")
    for parent, kids in sch.nested.items():
        v = c.args.get(parent)
        items = v if isinstance(v, list) else ([v] if isinstance(v, dict) else [])
        for it in items:
            if not isinstance(it, dict):
                continue
            for k in it:
                if k not in kids:
                    out.append(f"`{parent}` item field `{k}` is not in the schema of `{c.name}`")
            for k, req in kids.items():
                if req and k not in it:
                    out.append(f"`{parent}` item misses required field `{k}` in `{c.name}`")
    # dedupe
    return list(dict.fromkeys(out))


def flags(ex: Example) -> tuple[dict, list[str]]:
    f: dict[str, float] = {}
    notes: list[str] = []
    calls = ex.resp_calls
    text = ex.resp_text
    ctx = ex.prompt_text
    avail = set(ex.tools)

    f["n_calls"] = len(calls)
    f["has_text"] = float(len(text) > 0)
    f["tool_only"] = float(len(calls) > 0 and len(text) == 0)
    f["text_len"] = len(text)
    f["log_prompt_len"] = len(ctx) ** 0.5 / 100

    # 1. tools not in the schema
    hist_unlocked = set()
    for c in ex.hist_calls + calls:  # unlocks inside the same turn count too
        if c.name == "unlock_discoverable_agent_tool" and isinstance(c.args, dict):
            hist_unlocked.add(str(c.args.get("agent_tool_name", "")))
    unavailable = [c.name for c in calls if avail and c.name not in avail]
    f["n_unavailable"] = len(unavailable)
    for n in dict.fromkeys(unavailable):
        extra = " (it was unlocked earlier as a discoverable tool, but discoverable tools must be invoked via call_discoverable_agent_tool)" if n in hist_unlocked else ""
        notes.append(f"Tool `{n}` is called directly by the agent but is NOT listed in [AVAILABLE TOOLS]{extra}.")
    errored_names = {r.name for r in ex.hist_responses if r.error and "not found" in r.text.lower()}
    f["unavailable_prev_notfound"] = float(any(n in errored_names for n in unavailable))

    # 2. call to discoverable tool that was not unlocked
    for c in calls:
        if c.name == "call_discoverable_agent_tool" and isinstance(c.args, dict):
            tn = str(c.args.get("agent_tool_name", ""))
            given = tn in ctx and ("give_discoverable_user_tool" in ctx or "list_discoverable" in ctx)
            if tn and tn not in hist_unlocked and not given:
                notes.append(f"call_discoverable_agent_tool uses `{tn}` which was never unlocked in the history.")
                f["discoverable_not_unlocked"] = 1.0

    # 3. schema mismatches
    sch = [s for c in calls for s in schema_issues(ex, c)]
    missing = [s for s in sch if "missing" in s or "misses" in s]
    f["n_schema_missing"] = len(missing)
    f["n_schema_extra"] = len(sch) - len(missing)
    if missing:  # extra keys alone are usually tolerated by annotators
        notes += [f"Schema check: {s}." for s in sch[:6]]

    # 4. repeated identical call; repeated call that failed before
    prev = {}
    for m in ex.history:
        queues = {}
        for r in m.responses:
            queues.setdefault(r.name, []).append(r)
        for c in m.calls:  # pair the k-th call of a tool with the k-th response of that tool
            q = queues.get(c.name) or []
            err = q.pop(0).error if q else False
            prev[_canon(c)] = prev.get(_canon(c), False) or bool(err)
    rep = [c for c in calls if _canon(c) in prev]
    def _unlocked_now(c):
        return isinstance(c.args, dict) and str(c.args.get("agent_tool_name", "")) in hist_unlocked
    rep_err = [c for c in rep if prev[_canon(c)] and not (c.name == "call_discoverable_agent_tool" and _unlocked_now(c))]
    f["n_repeat_calls"] = len(rep)
    f["n_repeat_failed"] = len(rep_err)
    for c in rep_err:
        notes.append(f"`{c.name}` is re-issued with exactly the same arguments as an earlier call that returned an error.")
    for c in rep:
        inner = str(c.args.get("agent_tool_name", "")) if isinstance(c.args, dict) else ""
        if c not in rep_err and not (c.name.startswith(("get_", "search", "KB_", "grep", "list_", "check_", "find_"))
                                     or inner.startswith(("get_", "search", "list_", "check_", "find_"))):
            notes.append(f"`{c.name}` is re-issued with exactly the same arguments as an earlier call.")

    # 5. placeholders / fabricated argument values
    vals = []
    for c in calls:
        if c.args is not None:
            for v in _flat_values(c.args, []):
                vals.append((c.name, v))
            if isinstance(c.args, dict):
                for v in c.args.values():
                    if isinstance(v, str) and v.strip().startswith("{"):
                        try:
                            for vv in _flat_values(json.loads(v), []):
                                vals.append((c.name, vv))
                        except Exception:
                            pass
    ph = [(n, v) for n, v in vals if isinstance(v, str) and PLACEHOLDER_RE.search(v)]
    f["n_placeholder"] = len(ph)
    for n, v in ph[:3]:
        notes.append(f"Argument value `{v[:60]}` in `{n}` looks like a placeholder.")
    novel_ids = []
    ctx_low = ctx.lower()
    for n, v in vals:
        if isinstance(v, str) and ID_RE.fullmatch(v.strip()) and v.lower() not in ctx_low:
            novel_ids.append((n, v))
    f["n_novel_ids"] = len(novel_ids)
    for n, v in novel_ids[:4]:
        notes.append(f"Identifier `{v}` passed to `{n}` does not appear anywhere in the context.")
    # numbers in args not present in context (ignore small ints)
    ctx_nums = {x for x in (_norm_num(s) for s in NUM_RE.findall(ctx)) if x is not None}
    novel_nums = []
    for n, v in vals:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) > 10 and float(v) not in ctx_nums:
            novel_nums.append((n, v))
    f["n_novel_arg_nums"] = len(novel_nums)
    for n, v in novel_nums[:3]:
        notes.append(f"Numeric argument {v} in `{n}` does not appear in the context (may be computed - verify).")
    # numbers in text not present in context
    tnums = [x for x in (_norm_num(s) for s in NUM_RE.findall(text)) if x is not None and x > 31 and not (1900 <= x <= 2100 and x == int(x))]
    novel_tnums = [x for x in tnums if x not in ctx_nums]
    f["n_novel_text_nums"] = len(novel_tnums)
    f["frac_novel_text_nums"] = len(novel_tnums) / max(1, len(tnums))
    if novel_tnums:
        notes.append("Numbers in the reply text that do not appear verbatim in the context (check whether they are "
                     "correctly computed): " + ", ".join(str(int(x) if x == int(x) else x) for x in novel_tnums[:8]))

    # 6. writes and confirmation
    writes = [c for c in calls if c.name.startswith(WRITE_PREFIXES) and c.name != "log_verification"]
    f["n_writes"] = len(writes)
    f["write_no_confirm"] = float(bool(writes) and not CONFIRM_RE.search(ex.last_user or ""))
    f["n_transfer"] = float(any(c.name.startswith("transfer_to_human") for c in calls))
    f["multi_write_same_obj"] = 0.0
    objs = {}
    for c in writes:
        if isinstance(c.args, dict):
            for k in ("order_id", "reservation_id", "account_id", "line_id"):
                if k in c.args:
                    objs.setdefault(str(c.args[k]), []).append(c.name)
    for o, ns in objs.items():
        if len(ns) > 1:
            f["multi_write_same_obj"] = 1.0
            notes.append(f"Several state-changing calls ({', '.join(ns)}) target the same object {o} in one turn.")

    # 6a. trace rules (layer 1, soft): passed to the judge as hints, never decisive on their own
    db_writes = []
    for c in calls:
        name = c.name
        if name == "call_discoverable_agent_tool" and isinstance(c.args, dict):
            name = str(c.args.get("agent_tool_name", ""))
        if name.startswith(DB_WRITE_PREFIXES):
            db_writes.append((name, c))
    if db_writes and not CONFIRM_RE.search(ex.last_user or ""):
        f["trace_write_unconfirmed"] = 1.0
        notes.append("State-changing call(s) " + ", ".join(dict.fromkeys(n for n, _ in db_writes)) +
                     " issued, but the last user message contains no explicit confirmation — check that the user "
                     "explicitly confirmed exactly these action details before.")
    hist_text = " ".join(c.raw_args for c in ex.hist_calls if c.name.startswith(("get_", "find_", "search")))
    for name, c in db_writes:
        if name.startswith(("cancel_", "modify_", "return_", "exchange_", "update_")) and isinstance(c.args, dict):
            for k in ("order_id", "reservation_id"):
                oid = c.args.get(k)
                if oid and str(oid) not in hist_text:
                    f["trace_no_status_check"] = 1.0
                    notes.append(f"`{name}` acts on {oid}, but no details/status lookup of {oid} appears in the "
                                 "history — the policy requires checking its status first.")
    if calls:
        streak, last_name = 0, None
        for m in reversed(ex.history):
            if m.role != "ASSISTANT" or not m.calls:
                continue
            errs = [r for r in m.responses if r.error]
            if len(m.calls) == 1 and errs and (last_name in (None, m.calls[0].name)):
                streak += 1
                last_name = m.calls[0].name
            else:
                break
        if streak >= 2 and any(c.name == last_name for c in calls):
            f["trace_error_loop"] = 1.0
            notes.append(f"`{last_name}` already failed {streak} times in a row in the history and is called again.")

    # 6c. deterministic policy-precondition validators (hard when the policy is unambiguous, otherwise hints)
    from . import policy as _policy
    pol = _policy.check(ex)
    f["policy_hard"] = float(sum(1 for _, _, hard in pol if hard))
    f["policy_soft"] = float(sum(1 for _, _, hard in pol if not hard))
    for code, note, hard in pol:
        notes.append(("Policy check (certain): " if hard else "Policy check (verify): ") + note)

    # 6d. second-generation high-precision checks (validated on valid / APB / early turns, see experiments/rules2_eval.py)
    from . import rules2 as _rules2
    r2 = _rules2.check(ex)
    f["rules2_hard"] = float(sum(len(v) for v in r2.values()))
    for v in r2.values():
        for note in v[:2]:
            notes.append("Policy check (certain): " + note)

    # 6b. modify/exchange an item into itself
    f["n_identity_swap"] = 0.0
    for c in calls:
        if isinstance(c.args, dict) and isinstance(c.args.get("item_ids"), list) \
                and isinstance(c.args.get("new_item_ids"), list):
            same = [a for a, b in zip(c.args["item_ids"], c.args["new_item_ids"]) if str(a) == str(b)]
            if same:
                f["n_identity_swap"] += len(same)
                notes.append(f"`{c.name}` maps item(s) {', '.join(map(str, same[:3]))} to the identical new item id "
                             "(no actual change).")

    # 7. claimed actions in text without calls
    f["action_claim_no_call"] = float(bool(text) and not calls and bool(ACTION_CLAIM_RE.search(text)))
    f["text_and_calls"] = float(bool(text) and bool(calls))

    # 8. last history tool responses errored
    last_resps = [r for m in ex.history[-2:] for r in m.responses]
    f["last_resp_error"] = float(any(r.error for r in last_resps))
    f["domain_" + ex.domain] = 1.0
    return f, notes


FEATURE_ORDER = ["n_calls", "has_text", "tool_only", "text_len", "log_prompt_len", "n_unavailable",
                 "unavailable_prev_notfound", "discoverable_not_unlocked", "n_schema_missing", "n_schema_extra", "n_repeat_calls",
                 "n_repeat_failed", "n_placeholder", "n_novel_ids", "n_novel_arg_nums", "n_novel_text_nums",
                 "frac_novel_text_nums", "n_writes", "write_no_confirm", "n_transfer", "multi_write_same_obj", "n_identity_swap", "trace_write_unconfirmed", "trace_no_status_check", "trace_error_loop", "policy_hard", "policy_soft",
                 "action_claim_no_call", "text_and_calls", "last_resp_error", "domain_airline", "domain_retail",
                 "domain_telecom", "domain_banking"]


def feature_vector(f: dict) -> list[float]:
    return [float(f.get(k, 0.0)) for k in FEATURE_ORDER]
