"""Translate the dialogue text of Guardian-format rows from English to Russian, the way the test data looks.

Only the free text of ⟦USER⟧ / ⟦ASSISTANT…⟧ blocks (and of the response) is translated; the system block (policy,
tools), tool-call lines (→ TOOL_CALL …) and tool outputs (← TOOL_RESPONSE …) stay in English, as in the test.
Each unique text chunk is translated once (rows share their dialogue prefixes).

python experiments/translate_apb.py --base-url http://127.0.0.1:8000/v1 --model tpro --src apb_eval.parquet --out apb_eval_ru.parquet
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BLOCK_RE = re.compile(r"(⟦[^⟧]*⟧\n)")
RU_NOTE = ("Always communicate with the user in Russian. Keep tool names, tool call arguments, IDs, codes, and other "
           "data values exactly as they appear in the system — do not translate or transliterate them.")
INSTR_END = "Always make sure you generate valid JSON only.\n"


def patch_instructions(prompt: str) -> str:
    """The test data's agent instructions carry this extra Russian-language note (as in valid.parquet)."""
    if RU_NOTE in prompt or INSTR_END not in prompt:
        return prompt
    return prompt.replace(INSTR_END, INSTR_END + "\n" + RU_NOTE + "\n", 1)
SYSTEM = (
    "You are a professional translator. Translate the customer-service chat message from English to natural, fluent "
    "Russian, as a Russian-speaking customer or support agent would write it. Rules: keep EXACTLY unchanged all IDs, "
    "codes, reservation/order/item/user ids, emails, phone numbers, flight numbers, dates, amounts and numbers, tool "
    "and function names, JSON keys, anything in backticks, and people's names (keep them in Latin letters). Keep the "
    "markdown structure (lists, bold) and line breaks. If the message is a JSON object, translate only the "
    "human-readable string values. Output only the translation, nothing else.")


def split_free(body: str) -> tuple[str, str]:
    """Split a message body into its free text and its tool lines (kept verbatim)."""
    free, tools = [], []
    for ln in body.split("\n"):
        (tools if ln.startswith(("\t→", "\t←", "\t  ")) else free).append(ln)
    return "\n".join(free).strip("\n"), "\n".join(tools)


def chunks_of(text: str) -> list[str]:
    out = []
    parts = BLOCK_RE.split(text)
    for k in range(1, len(parts), 2):
        head, body = parts[k], parts[k + 1] if k + 1 < len(parts) else ""
        if head.startswith("⟦SYSTEM"):
            continue
        free, _ = split_free(body)
        if free.strip():
            out.append(free.strip())
    return out


def rebuild(text: str, tr: dict) -> str:
    parts = BLOCK_RE.split(text)
    res = [parts[0]]
    for k in range(1, len(parts), 2):
        head, body = parts[k], parts[k + 1] if k + 1 < len(parts) else ""
        if head.startswith("⟦SYSTEM") or not split_free(body)[0].strip():
            res += [head, body]
            continue
        free, tools = split_free(body)
        new_free = tr.get(free.strip(), free.strip())
        tail = "\n\n" if body.endswith("\n\n") else ("\n" if body.endswith("\n") else "")
        res += [head, new_free + ("\n" + tools if tools else "") + tail]
    return "".join(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="tpro")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    args = ap.parse_args()
    from openai import OpenAI
    cli = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    d = pd.read_parquet(args.src)
    uniq = sorted({c for p, r in zip(d.prompt, d.response) for c in chunks_of(p) + chunks_of(r)})
    print(len(uniq), "unique chunks", flush=True)

    def tr(s):
        for _ in range(3):
            try:
                r = cli.chat.completions.create(
                    model=args.model, temperature=0.0, max_tokens=4096,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": s}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                out = (r.choices[0].message.content or "").strip()
                if out:
                    return out
            except Exception:  # noqa: BLE001
                pass
        return s

    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(tr, uniq))
    table = dict(zip(uniq, res))
    d = d.copy()
    d["prompt"] = [patch_instructions(rebuild(p, table)) for p in d.prompt]
    d["response"] = [rebuild(r, table) for r in d.response]
    d.to_parquet(args.out, index=False)
    same = sum(1 for a, b in table.items() if a == b)
    print("translated", len(table) - same, "of", len(table), "chunks ->", args.out, flush=True)


if __name__ == "__main__":
    main()
