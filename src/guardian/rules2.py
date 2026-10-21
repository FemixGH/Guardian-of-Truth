"""High-precision deterministic checks for recall (second generation), each validated on all labelled rows before use.

tier_unverified      airline: the turn states the user's membership tier (gold/silver/regular) while no profile with
                     a membership field was fetched, or the profile says another tier.
payment_false_limit  telecom: the agent claims it cannot take the payment / lacks make_payment (the user pays the
                     payment request on the device) -> false limitation.
arg_provenance       an identifier-like argument (zip, email, ids, phone) of a tool call appears nowhere in the user's
                     messages or tool results as a standalone value (a zip inside an e-mail does not count).
return_and_exchange  retail: return AND exchange (or a second one) on the same delivered order.
payment_schema       airline: book_reservation payment_methods items without payment_id / amount.
exchange_false_refusal  retail: refuses + offers a human while an exchange to a named variant is available.
airline_transfer_individual  airline: offers a human for an individual review / exception (premature transfer).
asks_known_contact   after a failed user lookup asks for an e-mail / ID the user already gave.
carried_over         (not a flag, a veto candidate) the turn repeats an earlier assistant turn unchanged.
"""
from __future__ import annotations

import json
import re

from .parse import Example

TIERS = {"gold": r"gold|золот", "silver": r"silver|серебр", "regular": r"regular|обычн|стандартн"}
TIER_CTX = r"(?:member|membership|tier|status|участник|участни|член|статус|уровн|программ)"


def _user_text(ex: Example) -> str:
    return "\n".join(m.text for m in ex.history if m.role == "USER")


def _tool_text(ex: Example) -> str:
    return "\n".join(r.text for r in ex.hist_responses if not r.error)


def _profiles(ex: Example) -> list[dict]:
    out = []
    for r in ex.hist_responses:
        if r.name == "get_user_details" and not r.error:
            try:
                out.append(json.loads(r.text.strip()))
            except Exception:  # noqa: BLE001
                pass
    return out


def tier_unverified(ex: Example) -> list[str]:
    if ex.domain != "airline":
        return []
    t = (ex.resp_text or "").lower()
    stated = [k for k, pat in TIERS.items()
              if re.search(rf"(?:{pat})\w*\W{{0,3}}(?:\w+\W+){{0,2}}{TIER_CTX}|{TIER_CTX}\w*\W{{0,3}}(?:\w+\W+){{0,2}}(?:{pat})", t)]
    if not stated:
        return []
    # only when the tier is USED for an entitlement (free bags, compensation, certificate) with a number
    if not (re.search(r"(?i)bag|luggage|suitcase|багаж|сумк|чемодан|место|compensat|компенсац|certificate|сертификат", t)
            and re.search(r"\d", t)):
        return []
    tiers = {str(p.get("membership", "")).lower() for p in _profiles(ex) if isinstance(p, dict) and p.get("membership")}
    if not tiers:
        return [f"The turn states the user's membership tier ({', '.join(stated)}) but no user profile with a "
                "membership field was retrieved - the tier is taken from the user's words."]
    wrong = [s for s in stated if s not in tiers]
    if wrong and not (set(stated) & tiers):
        return [f"The turn states membership {', '.join(stated)} but the profile says {', '.join(sorted(tiers))}."]
    return []


PAY_LIMIT = re.compile(r"(?i)make_payment|(?:нет|не имею|отсутству\w*)\s+(?:\w+\s+){0,3}(?:доступ\w*\s+(?:к\s+)?)?"
                       r"(?:инструмент\w*|возможност\w*|функци\w*)\s+(?:\w+\s+){0,2}(?:для\s+)?(?:оплат|платеж|платёж)|"
                       r"не\s+мог\w*\s+(?:\w+\s+){0,2}(?:провести|выполнить|принять|обработать|осуществить)\s+(?:\w+\s+){0,2}"
                       r"(?:оплат|платеж|платёж)|(?:don'?t|do not|cannot|can'?t)\s+(?:\w+\s+){0,3}(?:make|process|take)\s+(?:\w+\s+){0,2}payment")


def payment_false_limit(ex: Example) -> list[str]:
    if ex.domain != "telecom":
        return []
    t = ex.resp_text or ""
    if PAY_LIMIT.search(t):
        return ["The agent claims it cannot take the payment / lacks a payment tool; in this policy the user pays the "
                "payment request on the device (false limitation)."]
    return []


ID_KEYS = {"zip", "email", "user_id", "reservation_id", "order_id", "payment_id", "payment_method_id", "customer_id",
           "line_id", "bill_id", "account_id", "card_id", "item_id", "item_ids", "new_item_ids", "product_id",
           "phone_number", "phone", "device_id", "flight_number", "transaction_id", "dispute_id", "id"}


def _standalone(value: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_@.])" + re.escape(value) + r"(?![A-Za-z0-9_])", text) is not None


def _values(k, v):
    if isinstance(v, list):
        for x in v:
            yield from _values(k, x)
    elif isinstance(v, (str, int)) and not isinstance(v, bool):
        yield k, str(v).strip()


def arg_provenance(ex: Example) -> list[str]:
    src = _user_text(ex) + "\n" + _tool_text(ex)
    prev_args = json.dumps([c.args for c in ex.hist_calls], ensure_ascii=False)
    sdig = re.sub(r"\D", "", src)
    out = []
    for c in ex.resp_calls:
        if not isinstance(c.args, dict):
            continue
        for k, v in c.args.items():
            if k not in ID_KEYS:
                continue
            for kk, val in _values(k, v):
                if len(val) < 3 or val.lower() in ("none", "null", "true", "false"):
                    continue
                if k in ("phone_number", "phone"):
                    d = re.sub(r"\D", "", val)
                    if len(d) >= 7 and d not in sdig:
                        out.append(f"{c.name}.{k} = {val}: not found in the user's messages or tool results.")
                    continue
                if _standalone(val, src) or val in prev_args:
                    continue
                if k == "email" and val.lower() in src.lower():
                    continue
                out.append(f"{c.name}.{k} = {val}: this value appears nowhere in the user's messages or tool results "
                           "as a standalone value.")
    return out


def return_and_exchange(ex: Example) -> list[str]:
    if ex.domain != "retail":
        return []
    names = ("return_delivered_order_items", "exchange_delivered_order_items")
    done = {}
    for c in ex.hist_calls:
        if c.name in names and isinstance(c.args, dict):
            done.setdefault(str(c.args.get("order_id")), c.name)
    seen = {}
    out = []
    for c in ex.resp_calls:
        if c.name in names and isinstance(c.args, dict):
            oid = str(c.args.get("order_id"))
            if oid in seen or oid in done:
                out.append(f"{c.name} on order {oid}, which already has a {seen.get(oid) or done.get(oid)} - a delivered "
                           "order can be returned OR exchanged only once.")
            seen[oid] = c.name
    return out


def payment_schema(ex: Example) -> list[str]:
    """airline book_reservation: every payment_methods item needs payment_id and amount (the tool schema); agents
    often write {"id": ..., "source": ...} copied from the user profile, which the API rejects / misbooks."""
    out = []
    for c in ex.resp_calls:
        if c.name != "book_reservation" or not isinstance(c.args, dict):
            continue
        for pm in c.args.get("payment_methods") or []:
            if isinstance(pm, dict) and not {"payment_id", "amount"} <= set(pm):
                out.append(f"book_reservation.payment_methods item {json.dumps(pm, ensure_ascii=False)[:120]} lacks the "
                           "required payment_id / amount fields of the tool schema.")
    return out


EX_REFUSE = re.compile(r"(?i)не\s+(?:смогу|могу|можем|получится|удастся)|невозможн|нельзя|cannot|can'?t|unable|"
                       r"not (?:able|possible)")
EX_TRANSFER = re.compile(r"(?i)перев(?:ести|еду|ела|ёл|ед)\w*\s+(?:\w+\s+){0,2}(?:на\s+)?(?:специалист|оператор|сотрудник|"
                         r"человек|агент)|transfer\w*\s+(?:you\s+)?to\s+(?:a\s+)?(?:human|live|specialist)|transfer_to_human")


def exchange_false_refusal(ex: Example) -> list[str]:
    """retail text reply that refuses and offers a human, while an item it names is a variant of the same product as an
    item of a delivered order (not yet returned/exchanged/cancelled): the exchange is an available path."""
    if ex.domain != "retail" or ex.resp_calls:
        return []
    t = ex.resp_text or ""
    if not (EX_REFUSE.search(t) and EX_TRANSFER.search(t)) or re.search(r"(?i)разниц\w* в цене|price difference", t):
        return []
    items, delivered = {}, {}
    for r in ex.hist_responses:
        if r.error or r.name not in ("get_order_details", "get_product_details"):
            continue
        try:
            d = json.loads(r.text.strip())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        if r.name == "get_order_details":
            for it in d.get("items") or []:
                items[str(it.get("item_id"))] = str(it.get("product_id"))
                if d.get("status") == "delivered":
                    delivered[str(it.get("item_id"))] = (str(it.get("product_id")), str(d.get("order_id")))
        else:
            for iid, v in (d.get("variants") or {}).items():
                if not isinstance(v, dict) or v.get("available", True):
                    items[str(iid)] = str(d.get("product_id"))
    used = {str(c.args.get("order_id")) for c in ex.hist_calls if isinstance(c.args, dict) and c.name in (
        "return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order")}
    out = []
    for y in sorted(set(re.findall(r"(?<!\d)\d{10}(?!\d)", t))):
        pid = items.get(y)
        for x, (px, oid) in delivered.items():
            if pid and px == pid and x != y and oid not in used:
                out.append(f"The reply refuses and offers a human, but delivered order {oid} has item {x}, a variant of the "
                           f"same product as {y}: exchange_delivered_order_items {x} -> {y} is an available path "
                           "(false refusal).")
    return out


EX_EXCEPTION = re.compile(r"(?i)индивидуальн|в порядке исключени|исключение|individual(?:ly)?|case[- ]by[- ]case|exception")


def airline_transfer_individual(ex: Example) -> list[str]:
    """airline text reply offering a human for an individual review / exception: the policy transfers only requests
    outside the agent's scope, a request the policy denies must be denied (2 hallucinations / 1 human-labelled clean)."""
    if ex.domain != "airline" or ex.resp_calls:
        return []
    t = ex.resp_text or ""
    if EX_TRANSFER.search(t) and EX_EXCEPTION.search(t):
        return ["The reply offers a transfer to a human for an individual review / exception; the policy transfers only "
                "requests outside the agent's scope (premature transfer)."]
    return []


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
ASK = re.compile(r"(?i)(?:предостав|укаж|сообщ|напиш|назов|уточн|provide|share|tell me|give me|confirm)\w*")
USER_LOOKUP = re.compile(r"find_user|get_user_information|get_user_details|lookup|search_user")
NOT_FOUND = re.compile(r"(?i)not found|no records|no (?:user|customer)|не найден")


def asks_known_contact(ex: Example) -> list[str]:
    """after a failed user lookup the reply asks for an e-mail / ID although the user already gave an e-mail that was
    never tried in any call (the obvious next lookup)."""
    if ex.resp_calls:
        return []
    t = ex.resp_text or ""
    if not ASK.search(t) or not re.search(r"(?i)e-?mail|электронн\w* почт|почт", t):
        return []
    if not any(USER_LOOKUP.search(r.name or "") and (r.error or NOT_FOUND.search(r.text or "")) for r in ex.hist_responses):
        return []
    used = json.dumps([c.args for c in ex.hist_calls], ensure_ascii=False).lower()
    users = "\n".join(m.text for m in ex.history if m.role == "USER")
    return [f"After a failed lookup the reply asks for an e-mail / ID, but the user already gave {e}, which was never "
            "used in a lookup." for e in sorted(set(EMAIL.findall(users))) if e.rstrip(".").lower() not in used]


# tier_unverified is kept for reference but disabled: precision on APB only 0.44 (11 pos / 14 neg)
RULES = {"payment_false_limit": payment_false_limit, "arg_provenance": arg_provenance,
         "return_and_exchange": return_and_exchange, "payment_schema": payment_schema,
         "exchange_false_refusal": exchange_false_refusal, "airline_transfer_individual": airline_transfer_individual,
         "asks_known_contact": asks_known_contact}


def check(ex: Example, rules=None) -> dict[str, list[str]]:
    out = {}
    for name, fn in RULES.items():
        if rules is not None and name not in rules:
            continue
        try:
            r = fn(ex)
        except Exception:  # noqa: BLE001
            r = []
        if r:
            out[name] = r
    return out
