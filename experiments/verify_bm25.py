"""Narrow fact-verification pass on judge-flagged turns: BM25 evidence lines for the auditor's finding, critic v3
decides only FACT objections. python experiments/verify_bm25.py --data X.parquet --flags judge.jsonl --out v.jsonl"""
import argparse, json, math, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardian.judge import Judge, finding_of, fit_context  # noqa: E402

TOK = re.compile(r"[A-Za-zА-Яа-яЁё_]+[0-9A-Za-z_]*|\d+(?:[.,]\d+)?")


def toks(s):
    return [t.lower().replace(",", ".") for t in TOK.findall(s)]


def bm25_lines(ctx, query, k=12, k1=1.2, b=0.75):
    lines = [l.strip() for l in ctx.split("\n") if len(l.strip()) > 3]
    docs = [toks(l) for l in lines]
    N = len(docs) or 1
    avg = sum(map(len, docs)) / N
    df = Counter(t for d in docs for t in set(d))
    q = [t for t in set(toks(query)) if df.get(t)]
    # value-like tokens (numbers, ids) get double weight
    w = {t: (2.0 if any(c.isdigit() for c in t) else 1.0) for t in q}
    sc = []
    for i, d in enumerate(docs):
        tf = Counter(d); s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (N - df[t] + .5) / (df[t] + .5))
                s += w[t] * idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        sc.append(s)
    top = sorted(range(len(lines)), key=lambda i: -sc[i])[:k]
    return [lines[i][:400] for i in sorted(top) if sc[i] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--flags", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1"); ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-ctx-chars", type=int, default=76000)
    a = ap.parse_args()
    df = pd.read_parquet(a.data).set_index("id")
    fl = [json.loads(l) for l in open(a.flags)]
    fl = [r for r in fl if (r.get("p") or 0) >= .5 and r["id"] in df.index]
    out = Path(a.out); done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines()}
    judge = Judge(a.base_url, "qwen", max_tokens=4096, prompt_path=None)
    sysmsg = (ROOT / "src/guardian/critic_prompt_v3.md").read_text()

    def work(r):
        if r["id"] in done:
            return
        row = df.loc[r["id"]]
        finding = finding_of(r.get("reasoning") or "", r.get("text") or "", 1800)
        ev = bm25_lines(row.prompt, finding + "\n" + row.response)
        ctx = fit_context(row.prompt, a.max_ctx_chars)
        user = (f"<context>\n{ctx}\n</context>\n\n<turn_under_review>\n{row.response}\n</turn_under_review>\n\n"
                f"<auditor_finding>\n{finding}\n</auditor_finding>\n\n<evidence>\n" + "\n".join(ev) +
                "\n</evidence>\n\nAnswer now with the two lines TYPE and VERDICT.")
        t0 = time.time()
        res = judge._score([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}])
        m = re.findall(r"TYPE\s*:\s*\**\s*(FACT|OTHER)", (res.get("text") or "") + (res.get("reasoning") or "")[-600:])
        o = dict(id=r["id"], p=res.get("p"), type=m[-1] if m else None, text=res.get("text"),
                 reasoning=(res.get("reasoning") or "")[-3000:], sec=round(time.time() - t0, 1))
        with open(out, "a") as fh:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(o["id"], o["type"], o["p"], o["sec"], flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(work, fl))


if __name__ == "__main__":
    main()
