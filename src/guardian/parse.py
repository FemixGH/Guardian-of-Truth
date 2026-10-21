"""Parser for the ⟦…⟧ transcript format used in Guardian of Truth.

Handles both variants seen so far:
  * dataset variant:  ⟦ASSISTANT · ход N⟧ ... \t→ TOOL_CALL name: {json} / \t← TOOL_RESPONSE name [ERROR]: ...
  * task-page variant: ⟦ASSISTANT_TOOL_CALL name="X"⟧ {json} / ⟦TOOL_RESULT name="X" requestor="..."⟧ ...
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

HEADER_RE = re.compile(r"⟦([A-Z_]+)([^⟧]*)⟧")
CALL_RE = re.compile(r"^\s*→\s*TOOL_CALL\s+([\w\-.]+)\s*:\s*(.*)$")
RESP_RE = re.compile(r"^\s*←\s*TOOL_RESPONSE\s+([\w\-.]+)\s*(\[ERROR\])?\s*:\s*(.*)$")
TOOL_DEF_RE = re.compile(r"^- ([A-Za-z_][\w\-.]*)\s+—", re.M)
PARAM_RE = re.compile(r"^(\s+)(?:·\s*)?([A-Za-z_][\w]*)\s*:\s*([\w\[\]|]+)(!?)")
NAME_ATTR_RE = re.compile(r'name="([^"]+)"')


@dataclass
class ToolCall:
    name: str
    raw_args: str
    args: object = None  # parsed json or None
    turn: int = -1
    role: str = "assistant"


@dataclass
class ToolResponse:
    name: str
    text: str
    error: bool = False
    turn: int = -1


@dataclass
class Message:
    role: str  # SYSTEM / USER / ASSISTANT
    turn: int
    text: str  # free text (without tool lines)
    calls: list = field(default_factory=list)
    responses: list = field(default_factory=list)


@dataclass
class ToolSchema:
    name: str
    params: dict  # top-level name -> required(bool)
    nested: dict  # parent param -> {child: required}


def _try_json(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def _is_error(text: str) -> bool:
    t = text[:300].lower()
    return t.startswith("error") or "not found" in t[:120] or '"error"' in t[:60]


def split_messages(text: str) -> list[Message]:
    msgs: list[Message] = []
    if not isinstance(text, str):
        return msgs
    pos = [(m.start(), m.end(), m.group(1), m.group(2)) for m in HEADER_RE.finditer(text)]
    for i, (s, e, tag, attrs) in enumerate(pos):
        body = text[e: pos[i + 1][0] if i + 1 < len(pos) else len(text)]
        tm = re.search(r"(\d+)", attrs)
        turn = int(tm.group(1)) if tm else -1
        if tag in ("ASSISTANT_TOOL_CALL", "USER_TOOL_CALL", "TOOL_CALL"):  # TOOL_CALL: baseline-prompt variant
            nm = NAME_ATTR_RE.search(attrs)
            role = "user" if tag.startswith("USER") else "assistant"
            tc = ToolCall(nm.group(1) if nm else "?", body.strip(), _try_json(body), turn, role)
            if msgs and msgs[-1].role == ("ASSISTANT" if role == "assistant" else "USER"):
                msgs[-1].calls.append(tc)
            else:
                msgs.append(Message("ASSISTANT" if role == "assistant" else "USER", turn, "", [tc], []))
            continue
        if tag == "TOOL_RESULT":
            nm = NAME_ATTR_RE.search(attrs)
            tr = ToolResponse(nm.group(1) if nm else "?", body.strip(), _is_error(body.strip()), turn)
            if msgs:
                msgs[-1].responses.append(tr)
            continue
        role = tag if tag in ("SYSTEM", "USER", "ASSISTANT") else tag
        free, calls, resps = [], [], []
        in_resp = False
        for line in body.split("\n"):
            cm = CALL_RE.match(line)
            if cm:
                calls.append(ToolCall(cm.group(1), cm.group(2), _try_json(cm.group(2)), turn,
                                      "assistant" if role == "ASSISTANT" else "user"))
                in_resp = False
                continue
            rm = RESP_RE.match(line)
            if rm:
                resps.append(ToolResponse(rm.group(1), rm.group(3), bool(rm.group(2)) or _is_error(rm.group(3)), turn))
                in_resp = True
                continue
            if in_resp and (line.startswith(("\t ", "\t\t", "    ")) or not line.strip()):
                resps[-1].text += "\n" + line.strip()  # multi-line tool output continuation
                continue
            in_resp = False
            free.append(line)
        msgs.append(Message(role, turn, "\n".join(free).strip(), calls, resps))
    return msgs


def parse_tools(system_text: str) -> dict[str, ToolSchema]:
    """Parse the [AVAILABLE TOOLS] block of the system prompt."""
    i = system_text.find("[AVAILABLE TOOLS]")
    if i < 0:
        return {}
    block = system_text[i:]
    tools: dict[str, ToolSchema] = {}
    cur = None
    last_top = None
    for line in block.split("\n"):
        m = re.match(r"^- ([A-Za-z_][\w\-.]*)\s+—", line) or re.match(r"^- ([A-Za-z_][\w\-.]*)\s*$", line)
        if m:
            cur = ToolSchema(m.group(1), {}, {})
            tools[cur.name] = cur
            last_top = None
            continue
        if cur is None:
            continue
        pm = PARAM_RE.match(line)
        if pm and "—" in line:
            indent = len(pm.group(1).expandtabs(4))
            nested = "·" in line[: len(pm.group(1)) + 3]
            name, req = pm.group(2), pm.group(4) == "!"
            if nested and last_top:
                cur.nested.setdefault(last_top, {})[name] = req
            elif indent <= 4:
                cur.params[name] = req
                last_top = name
    return tools


@dataclass
class Example:
    system: str
    tools: dict
    history: list  # Messages of the prompt (excluding SYSTEM)
    response: list  # Messages of the response
    prompt_text: str
    response_text: str

    @property
    def resp_calls(self):
        return [c for m in self.response for c in m.calls]

    @property
    def resp_text(self):
        return "\n".join(m.text for m in self.response if m.role == "ASSISTANT").strip()

    @property
    def hist_calls(self):
        return [c for m in self.history for c in m.calls]

    @property
    def hist_responses(self):
        return [r for m in self.history for r in m.responses]

    @property
    def last_user(self) -> str:
        for m in reversed(self.history):
            if m.role == "USER":
                return m.text
        return ""

    @property
    def domain(self) -> str:
        s = self.system[:20000].lower()
        if "telecom" in s or "mms" in s or "roaming" in s:
            return "telecom"
        if "airline" in s or "reservation" in s:
            return "airline"
        if "rho-bank" in s or "bank" in s:
            return "banking"
        if "order" in s or "retail" in s:
            return "retail"
        return "other"


def parse_example(prompt: str, response: str) -> Example:
    pm = split_messages(prompt or "")
    system = "\n".join(m.text for m in pm if m.role == "SYSTEM")
    history = [m for m in pm if m.role != "SYSTEM"]
    response = response or ""
    if response.strip() and not HEADER_RE.search(response):
        response = "⟦ASSISTANT⟧\n" + response
    rm = split_messages(response)
    return Example(system, parse_tools(system), history, rm, prompt or "", response or "")
