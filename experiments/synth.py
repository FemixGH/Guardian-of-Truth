"""Synthetic training data for a small hallucination classifier (TrajAD-style perturbation of clean turns).

Source: early agent turns of the valid dialogues that NO judge run ever flagged (consensus-clean). Each clean turn
is kept as a negative (label 0) and perturbed with every applicable deterministic mutation (label 1, `mtype`).
Hard negatives (label 0) apply meaning-preserving edits that the task explicitly allows (reordered parallel calls,
number formatting) so that the model does not learn "any edit = hallucination".

python experiments/synth.py --out runs/synth_det.parquet
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian.parse import parse_example  # noqa: E402

CALL_LINE_RE = re.compile(r"^(\s*→\s*TOOL_CALL\s+)([\w\-.]+)(\s*:\s*)(.*)$", re.M)
# user-device actions in telecom (not agent tools) and plausible-but-absent agent tools per domain
FAKE_TOOLS = {
    "telecom": ["check_network_status", "check_sim_status", "run_speed_test", "check_apn_settings",
                "toggle_airplane_mode", "reset_network_settings", "check_status_bar", "reboot_device"],
    "airline": ["get_reservation_status", "update_reservation", "issue_refund", "get_user_reservations",
                "check_flight_availability", "apply_compensation", "get_baggage_policy"],
    "retail": ["get_order_status", "cancel_order", "update_order_address", "refund_order", "get_user_orders",
               "return_order_items", "check_item_availability"],
    "banking": ["get_account_balance", "freeze_card", "get_transactions", "update_user_info",
                "close_account", "get_card_details", "issue_replacement_card"],
}
ID_PATTERNS = [
    re.compile(r"^#W\d{7}$"),          # retail order
    re.compile(r"^[A-Z0-9]{6}$"),       # airline reservation
    re.compile(r"^\d{10}$"),            # retail item / product
    re.compile(r"^[a-z]+_[a-z]+_\d{3,5}$"),  # user id
    re.compile(r"^[A-Z]\d{4}$"),        # telecom customer/line ids (C1001, L1002)
    re.compile(r"^HAT\d{3}$"),          # flight number
    re.compile(r"^[a-z]{2,4}_[a-z0-9_]{4,}$"),  # banking ids (chk_..., dbc_...)
]


def domain_of(row_id: str) -> str:
    d = row_id.split("__")[0]
    return "banking" if d.startswith("banking") else d


def is_id(v) -> bool:
    return isinstance(v, str) and any(p.match(v) for p in ID_PATTERNS)


def perturb_id(v: str, rnd: random.Random) -> str:
    chars = list(v)
    pos = [i for i, c in enumerate(chars) if c.isalnum()][-4:]
    i = rnd.choice(pos)
    c = chars[i]
    if c.isdigit():
        chars[i] = str((int(c) + rnd.randint(1, 9)) % 10)
    elif c.isupper():
        chars[i] = rnd.choice([x for x in "ABCDEFGHJKLMNPQRSTUVWXYZ" if x != c])
    else:
        chars[i] = rnd.choice([x for x in "abcdefghijklmnopqrstuvwxyz" if x != c])
    return "".join(chars)


def replace_call(response: str, k: int, name: str | None = None, args=None) -> str:
    """Rewrite the k-th TOOL_CALL line of the response (new name and/or new JSON args)."""
    ms = list(CALL_LINE_RE.finditer(response))
    m = ms[k]
    new_args = json.dumps(args, ensure_ascii=False) if args is not None else m.group(4)
    line = m.group(1) + (name or m.group(2)) + m.group(3) + new_args
    return response[:m.start()] + line + response[m.end():]


def walk_ids(obj, path=()):
    """Yield (path, value) for id-like string leaves of a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_ids(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_ids(v, path + (i,))
    elif is_id(obj):
        yield path, obj


def set_path(obj, path, value):
    obj = json.loads(json.dumps(obj))
    cur = obj
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value
    return obj


def mutations(row_id: str, prompt: str, response: str, rnd: random.Random):
    """Yield (mtype, new_response, label)."""
    ex = parse_example(prompt, response)
    dom = domain_of(row_id)
    calls = ex.resp_calls
    ctx_ids = set(re.findall(r"#W\d{7}|\b[A-Z0-9]{6}\b|\b\d{10}\b|\b[a-z]+_[a-z]+_\d{3,5}\b", prompt))

    for k, c in enumerate(calls):
        if not isinstance(c.args, dict):
            continue
        # 1. tool that is not available to the agent
        fakes = [t for t in FAKE_TOOLS.get(dom, []) if t not in ex.tools]
        if fakes:
            yield "unavailable_tool", replace_call(response, k, name=rnd.choice(fakes)), 1
        # 2. required schema field dropped / renamed
        sch = ex.tools.get(c.name)
        req = [p for p, r in (sch.params.items() if sch else []) if r and p in c.args]
        if req:
            p = rnd.choice(req)
            a = {kk: vv for kk, vv in c.args.items() if kk != p}
            yield "schema_missing", replace_call(response, k, args=a), 1
            alt = {"user_id": "id", "order_id": "id", "reservation_id": "booking_id", "item_ids": "items",
                   "payment_method_id": "payment_id", "customer_id": "client_id", "line_id": "id",
                   "flight_number": "flight", "card_id": "id", "account_id": "id"}.get(p, p + "_ref")
            a2 = {(alt if kk == p else kk): vv for kk, vv in c.args.items()}
            yield "schema_renamed", replace_call(response, k, args=a2), 1
        # 3. id argument: fabricated (perturbed) or another entity present in the context
        ids = list(walk_ids(c.args))
        if ids:
            path, v = rnd.choice(ids)
            yield "fabricated_id", replace_call(response, k, args=set_path(c.args, path, perturb_id(v, rnd))), 1
            same_kind = [x for x in ctx_ids if x != v and any(p.match(x) and p.match(v) for p in ID_PATTERNS)]
            if same_kind:
                yield "wrong_entity_id", replace_call(
                    response, k, args=set_path(c.args, path, rnd.choice(sorted(same_kind)))), 1
            # 4. placeholder argument
            ph = rnd.choice(["<" + str(path[-1]) + ">", "placeholder", "USER_ID_HERE", "unknown"])
            yield "placeholder", replace_call(response, k, args=set_path(c.args, path, ph)), 1
        # 5. calculator: extra / missing operand
        if c.name == "calculate" and isinstance(c.args.get("expression"), str):
            e = c.args["expression"]
            nums = re.findall(r"\d+(?:\.\d+)?", e)
            if len(nums) >= 2:
                e2 = e.replace(nums[-1], str(round(float(nums[-1]) * rnd.choice([1.1, 0.9, 2]), 2)), 1) \
                    if rnd.random() < 0.5 else e + " + " + rnd.choice(nums)
                yield "calc_operand", replace_call(response, k, args={**c.args, "expression": e2}), 1
    # 6. repeat of a call that failed in the history (tool-only turns)
    if calls and not ex.resp_text:
        failed = [(c, r) for m in ex.history for c, r in zip(m.calls, m.responses) if r.error and c.name == r.name]
        if failed:
            c, _ = rnd.choice(failed)
            head = response.split("\n")[0]
            yield "repeat_failed_call", f"{head}\n\t→ TOOL_CALL {c.name}: {c.raw_args}", 1
    # 7. wrong number in the reply text (a number that appears in the context)
    text = ex.resp_text or ""
    nums = [n for n in re.findall(r"(?<![\w.])\d{2,6}(?:[.,]\d{1,2})?(?![\w])", text) if n in prompt]
    if nums:
        n = rnd.choice(nums)
        val = float(n.replace(",", "."))
        new = val + rnd.choice([-1, 1]) * max(1, round(val * rnd.uniform(0.05, 0.3)))
        new_s = str(int(new)) if "." not in n and "," not in n else f"{new:.2f}"
        if new_s != n:
            yield "wrong_number", response.replace(n, new_s, 1), 1
    dates = [d for d in re.findall(r"\d{4}-\d{2}-\d{2}", text) if d in prompt]
    if dates:
        d = rnd.choice(dates)
        day = int(d[-2:])
        nd = d[:-2] + f"{(day % 27) + 1:02d}"
        if nd != d:
            yield "wrong_date", response.replace(d, nd, 1), 1
    # hard negatives: reordered parallel calls, number formatting
    ms = list(CALL_LINE_RE.finditer(response))
    if len(ms) >= 2:
        lines = [m.group(0) for m in ms]
        sh = lines[:]
        while sh == lines:
            rnd.shuffle(sh)
        out = response
        for a, b in zip(lines, sh):
            out = out.replace(a, "\x00" + str(lines.index(b)), 1)
        for i, l in enumerate(lines):
            out = out.replace("\x00" + str(i), l, 1)
        yield "neg_reorder_calls", out, 0
    big = re.findall(r"(?<![\w.,])(\d{4,6})(?![\w.,])", text)
    if big:
        n = big[0]
        yield "neg_number_format", response.replace(n, f"{int(n):,}".replace(",", " "), 1), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "synth_det.parquet"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    h = pd.read_parquet(ROOT / "experiments" / "history_turns.parquet")
    flag = set()
    for f in ["s9_pred", "s10_pred", "s11_pred"]:
        p = pd.read_parquet(ROOT / "runs" / "a100" / f"{f}.parquet")
        flag |= set(p[p.label == 1].id)
    for f in ["v3f_t0", "fp8b_v3f", "awq_v3f", "sp_fp8_96", "sp_awq96", "sp_fp8_160", "sp_awq192", "v2_t07a", "v3b_t0"]:
        fp = ROOT / "runs" / "a100" / "exp" / f"{f}.jsonl"
        if fp.exists():
            for line in fp.read_text().splitlines():
                r = json.loads(line)
                if (r.get("p") or 0) >= 0.5:
                    flag.add(r["id"])
    clean = h[~h.id.isin(flag)]
    rows = []
    for r in clean.itertuples(index=False):
        traj = r.id.split("::")[0]
        rows.append(dict(id=r.id + "#orig", src=r.id, traj=traj, prompt=r.prompt, response=r.response,
                         label=0, mtype="orig"))
        for j, (mt, resp, lab) in enumerate(mutations(r.id, r.prompt, r.response, rnd)):
            if resp != r.response:
                rows.append(dict(id=f"{r.id}#{mt}{j}", src=r.id, traj=traj, prompt=r.prompt, response=resp,
                                 label=lab, mtype=mt))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(len(clean), "clean turns ->", len(df), "rows")
    print(df.groupby(["label", "mtype"]).size().to_string())


if __name__ == "__main__":
    main()
