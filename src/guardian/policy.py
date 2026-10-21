"""Deterministic policy-precondition validators (layer 1, "state transition" checks).

Each validator looks at a state-changing call in the turn under review, takes the facts it needs from the most recent
tool results in the history (and the current time stated in the policy), and reports a violation only when all the
facts it needs are present. Missing facts -> no finding (high precision by design).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from .parse import Example

TIME_RE = re.compile(r"current time is (\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2})?)", re.I)
FREE_BAGS = {"regular": {"basic_economy": 0, "economy": 1, "business": 2},
             "silver": {"basic_economy": 1, "economy": 2, "business": 3},
             "gold": {"basic_economy": 2, "economy": 3, "business": 4}}
CANCEL_REASONS = {"no longer needed", "ordered by mistake"}


def _now(ex: Example) -> datetime | None:
    m = TIME_RE.search(ex.system) or TIME_RE.search(" ".join(r.text[:200] for r in ex.hist_responses
                                                           if r.name == "get_current_time"))
    if not m:
        return None
    t = m.group(2) if len(m.group(2)) == 8 else m.group(2) + ":00"
    try:
        return datetime.fromisoformat(f"{m.group(1)} {t}")
    except ValueError:
        return None


def _date(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    s = s.strip().replace("T", " ")
    for fmt, n in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d %H:%M", 16), ("%Y-%m-%d", 10)):
        try:
            return datetime.strptime(s[:n], fmt)
        except ValueError:
            continue
    return None


def _json_results(ex: Example, name: str) -> list:
    """Parsed JSON results of a tool in the history, oldest first."""
    out = []
    for r in ex.hist_responses:
        if r.name != name or r.error:
            continue
        try:
            out.append(json.loads(r.text.strip()))
        except Exception:  # noqa: BLE001
            continue
    return out


def _latest(objs: list, key: str, value) -> dict | None:
    for o in reversed(objs):
        if isinstance(o, dict) and str(o.get(key)) == str(value):
            return o
    return None


def _user_text(ex: Example) -> str:
    return "\n".join(m.text for m in ex.history if m.role == "USER")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


# ----------------------------------------------------------------------------------------------- airline
def _airline(ex: Example, calls) -> list[tuple[str, str, bool]]:
    out = []
    now = _now(ex)
    reservations = _json_results(ex, "get_reservation_details")
    users = _json_results(ex, "get_user_details")
    statuses = [r.text.lower() for r in ex.hist_responses if r.name == "get_flight_status" and not r.error]
    for c in calls:
        a = c.args if isinstance(c.args, dict) else {}
        if c.name == "cancel_reservation" and now is not None:
            res = _latest(reservations, "reservation_id", a.get("reservation_id"))
            if not res:
                continue
            flights = res.get("flights") or []
            dates = [_date(f.get("date")) for f in flights if isinstance(f, dict)]
            flown = any(d is not None and d.date() < now.date() for d in dates)
            created = _date(res.get("created_at"))
            within_24h = created is not None and timedelta(0) <= now - created <= timedelta(hours=24)
            business = str(res.get("cabin")) == "business"
            # insurance only covers health / weather reasons stated by the user
            covered = bool(re.search(r"(?i)болен|болела|заболел|болезн|здоров|госпитал|больниц|травм|врач|sick|ill\b|"
                                     r"health|medical|hospital|погод|шторм|ураган|снегопад|weather|storm", _user_text(ex)))
            insured = str(res.get("insurance")).lower() == "yes" and covered
            airline_cancelled = any("cancelled" in s for s in statuses)
            if flown:
                out.append(("cancel_flown", f"cancel_reservation {a.get('reservation_id')}: part of the trip is dated "
                            f"before the current time ({now.date()}) — the policy requires a transfer, not a cancellation.",
                            False))
            elif created is not None and not (within_24h or business or insured or airline_cancelled):
                out.append(("cancel_not_eligible", f"cancel_reservation {a.get('reservation_id')}: booked "
                            f"{res.get('created_at')} (>24h before {now}), cabin {res.get('cabin')}, insurance "
                            f"{res.get('insurance')}, no airline cancellation seen — none of the policy's cancellation "
                            "conditions holds.", False))
        if c.name in ("update_reservation_baggages", "book_reservation"):
            total, nonfree = a.get("total_baggages"), a.get("nonfree_baggages")
            if not isinstance(total, int) or not isinstance(nonfree, int):
                continue
            if c.name == "book_reservation":
                cabin, npass, uid = a.get("cabin"), len(a.get("passengers") or []), a.get("user_id")
            else:
                res = _latest(reservations, "reservation_id", a.get("reservation_id"))
                if not res:
                    continue
                cabin, npass, uid = res.get("cabin"), len(res.get("passengers") or []), res.get("user_id")
            user = _latest(users, "user_id", uid)
            tier = str((user or {}).get("membership", "")).lower()
            if tier not in FREE_BAGS or cabin not in FREE_BAGS[tier] or npass <= 0:
                continue
            expected = max(0, total - FREE_BAGS[tier][cabin] * npass)
            if nonfree != expected:
                out.append(("baggage_mismatch", f"{c.name}: {tier} member, {cabin}, {npass} passenger(s) -> "
                            f"{FREE_BAGS[tier][cabin]} free bag(s) each, so {total} total bags means "
                            f"{expected} non-free, but nonfree_baggages={nonfree}.", True))
        if c.name == "update_reservation_flights":
            res = _latest(reservations, "reservation_id", a.get("reservation_id"))
            if res and res.get("cabin") == "basic_economy" and a.get("cabin") == "basic_economy":
                old = sorted(str(f.get("flight_number")) for f in res.get("flights") or [] if isinstance(f, dict))
                new = sorted(str(f.get("flight_number")) for f in a.get("flights") or [] if isinstance(f, dict))
                if old and new and old != new:
                    out.append(("basic_economy_modified", f"update_reservation_flights changes the flights of basic "
                                f"economy reservation {a.get('reservation_id')}, which the policy forbids.", True))
    return out


# ----------------------------------------------------------------------------------------------- telecom
def _telecom(ex: Example, calls) -> list[tuple[str, str, bool]]:
    out = []
    now = _now(ex)
    lines = _json_results(ex, "get_details_by_id")
    bills = []
    for b in _json_results(ex, "get_bills_for_customer"):
        bills += b if isinstance(b, list) else [b]
    bills += [b for b in _json_results(ex, "get_details_by_id") if isinstance(b, dict) and "bill_id" in b]
    for c in calls:
        a = c.args if isinstance(c.args, dict) else {}
        if c.name == "resume_line" and now is not None:
            line = _latest(lines, "line_id", a.get("line_id"))
            end = _date((line or {}).get("contract_end_date"))
            if end is not None and end.date() < now.date():
                out.append(("resume_expired_contract", f"resume_line {a.get('line_id')}: contract_end_date "
                            f"{line.get('contract_end_date')} is before the current time {now.date()} — the policy "
                            "forbids lifting the suspension.", True))
        if c.name == "send_payment_request":
            bill = _latest(bills, "bill_id", a.get("bill_id"))
            if bill and str(bill.get("status", "")).lower() not in ("overdue", ""):
                out.append(("payment_request_not_overdue", f"send_payment_request for bill {a.get('bill_id')} whose "
                            f"status is {bill.get('status')} (only overdue bills may get a payment request).", True))
    return out


# ----------------------------------------------------------------------------------------------- retail
def _retail(ex: Example, calls) -> list[tuple[str, str, bool]]:
    out = []
    orders = _json_results(ex, "get_order_details")
    need = {"cancel_pending_order": "pending", "modify_pending_order_items": "pending",
            "modify_pending_order_address": "pending", "modify_pending_order_payment": "pending",
            "return_delivered_order_items": "delivered", "exchange_delivered_order_items": "delivered"}
    for c in calls:
        a = c.args if isinstance(c.args, dict) else {}
        if c.name in need:
            order = _latest(orders, "order_id", a.get("order_id"))
            st = str((order or {}).get("status", "")).lower()
            if order and st and st != need[c.name]:
                out.append(("order_status", f"{c.name} on order {a.get('order_id')} whose status is '{st}' "
                            f"(must be '{need[c.name]}').", True))
        if c.name == "cancel_pending_order":
            reason = str(a.get("reason", "")).strip().lower()
            if reason and reason not in CANCEL_REASONS:
                out.append(("cancel_reason", f"cancel_pending_order with reason '{reason}' — only 'no longer needed' "
                            "or 'ordered by mistake' are allowed.", True))
    return out


# ----------------------------------------------------------------------------------------------- banking
def _banking(ex: Example, calls) -> list[tuple[str, str, bool]]:
    out = []
    utext = _user_text(ex)
    udig = _digits(utext)
    ulow = utext.lower()
    for c in calls:
        if c.name != "log_verification" or not isinstance(c.args, dict):
            continue
        a = c.args
        provided = []
        email = str(a.get("email", "")).lower()
        if email and email in ulow:
            provided.append("email")
        phone = _digits(str(a.get("phone_number", "")))
        if len(phone) >= 7 and phone in udig:
            provided.append("phone")
        addr = str(a.get("address", ""))
        num = re.match(r"\s*(\d+)\s+(\w+)", addr)
        if num and num.group(1) in utext and num.group(2).lower() in ulow:
            provided.append("address")
        dob = _digits(str(a.get("date_of_birth", "")))
        if dob and (dob in udig or (len(dob) == 8 and (dob[-4:] in utext and (dob[:2] in udig or dob[2:4] in udig)))):
            provided.append("dob")
        if len(provided) < 2:
            out.append(("verification_insufficient", "log_verification: in the user's own messages only "
                        f"{len(provided)} of date of birth / email / phone / address ({', '.join(provided) or 'none'}) "
                        "can be found — the policy requires the user to provide 2 of them.", True))
    return out


ASK_PATTERNS = {  # only patterns that were precise on valid + history (ids / e-mails were too noisy)
    "date of birth": (r"(?i)(дат[уыа] рождения|date of birth)", ["dob", "date_of_birth"]),
}
ACTION_CLAIM = re.compile(r"(?i)\b(отправил[аи]?|отправлен[аоы]?|сбросил[аи]?|сброшен[аоы]?|оформил[аи]?|оформлен[аоы]?|"
                          r"отменил[аи]?|изменил[аи]?|заказал[аи]?|заблокировал[аи]?|разблокировал[аи]?|перевёл|перевел|"
                          r"выполняю|выполнил[аи]?|запустил[аи]?|проверил[аи]? (?:статус|диагностик))")


def _hints(ex: Example) -> list[tuple[str, str, bool]]:
    """Generic soft hints (all domains)."""
    out = []
    text = ex.resp_text
    calls = ex.resp_calls
    ctx_tools = " ".join(r.text for r in ex.hist_responses)
    # 1. asking for information that is already in earlier tool results
    if text and re.search(r"(?i)уточните|укажите|предоставьте|сообщите|назовите|напишите|подскажите|provide|"
                          r"please (?:give|tell|share)|\?", text):
        for what, (pat, keys) in ASK_PATTERNS.items():
            if re.search(pat, text):
                for k in keys:
                    m = re.search(r'"%s"\s*:\s*"([^"]{4,})"' % k, ctx_tools) or re.search(r"\b%s:\s*(\S{4,})" % k, ctx_tools)
                    if m:
                        out.append(("asks_known_info", f"The reply asks the user for their {what}, but an earlier tool "
                                    f"result already contains {k} = {m.group(1)[:40]} — check whether it is the same "
                                    "person/object.", False))
                        break
    # 2. banking: transfer / "no information" while KB_search_bm25 never used
    if ex.domain == "banking":
        used = {c.name for c in ex.hist_calls}
        refusal = bool(re.search(r"(?i)(нет информаци|не наш[её]л|отсутству|не могу найти|not found|no information)", text))
        if ("KB_search_bm25" not in used) and (any(c.name.startswith("transfer_to_human") for c in calls) or refusal):
            out.append(("transfer_without_bm25", "The agent transfers / states the information is missing, but "
                        "KB_search_bm25 (an available search tool) was never used in this conversation.", False))
    return out


# soft hints (verify-type notes) gave no reliable gain on A100 checks (v3g/v3h): off by default; hard checks stay on
SOFT_HINTS = False


def check(ex: Example, soft: bool | None = None) -> list[tuple[str, str, bool]]:
    """Returns [(code, note, hard)] for the state-changing calls of the turn under review."""
    soft = SOFT_HINTS if soft is None else soft
    calls = list(ex.resp_calls)
    for c in list(calls):  # unwrap discoverable calls
        if c.name == "call_discoverable_agent_tool" and isinstance(c.args, dict):
            try:
                inner = json.loads(c.args.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                inner = {}
            calls.append(type(c)(str(c.args.get("agent_tool_name", "")), "", inner, c.turn, c.role))
    dom = ex.domain
    hints = []
    try:
        hints = _hints(ex) if soft else []
    except Exception:  # noqa: BLE001
        hints = []
    try:
        if dom == "airline":
            return [x for x in _airline(ex, calls) if soft or x[2]] + hints
        if dom == "telecom":
            return _telecom(ex, calls) + hints
        if dom == "retail":
            return _retail(ex, calls) + hints
        if dom == "banking":
            return [x for x in _banking(ex, calls) if soft or x[2]] + hints
    except Exception:  # noqa: BLE001 — a validator must never break the pipeline
        return hints
    return hints
