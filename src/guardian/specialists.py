"""Routing of turns to narrow specialist checks (run only on matching rows, after the general judge)."""
from __future__ import annotations

import re

from .parse import parse_example

REFUSAL_RE = re.compile(
    r"не (?:могу|смогу|сможем|можем|имею возможности|в состоянии|получится|удастся|предусмотрен)|невозможно|нет возможности|"
    r"не позволя|не разрешает|не допускает|за пределами (?:моих|наших)|вне (?:моих|моей) компетенц|"
    r"(?:перевести|переведу|перевожу|передать|передам|соединить|соединю)\s+(?:вас\s+)?(?:на|с|к)\s+(?:оператор|специалист|сотрудник|живо|человек|агент)|"
    r"\b(?:cannot|can't|unable to|not able to|not possible|human agent|transfer you)\b",
    re.I)
NUM_TEXT_RE = re.compile(r"\$\s?\d|\d[\d\s,.]*\s?(?:долл|usd|руб|€|eur)|\d{4}-\d{2}-\d{2}|"
                         r"(?:итого|всего|сумм|общая стоимость|возврат|компенсац|доплат|разниц|баланс|total|refund)", re.I)


def route(prompt: str, response: str) -> list[str]:
    """Names of specialist checks that apply to this turn."""
    try:
        ex = parse_example(prompt or "", response or "")
    except Exception:  # noqa: BLE001
        return []
    out = []
    text = ex.resp_text or ""
    names = [c.name for c in ex.resp_calls]
    if any(n.startswith("transfer_to_human") for n in names) or REFUSAL_RE.search(text):
        out.append("escalation")
    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    if "calculate" in names or (NUM_TEXT_RE.search(text) and len(nums) >= 2):
        out.append("numbers")
    return out
