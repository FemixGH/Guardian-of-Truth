"""How do read-only-tool-call-only turns behave: gold vs Qwen flags (APB-ru, valid)."""
import json, re, pandas as pd
READ = re.compile(r"^(get_|search_|find_|list_|calculate|kb_search|lookup)")
DEVICE = re.compile(r"^(check_|run_speed|can_send|toggle_|reboot|reseat|reset_apn|set_network|grant_app)")
def kind(resp):
    lines=[l for l in resp.split("\n")[1:] if l.strip()]
    free=[l for l in lines if not l.startswith(("\t→","\t←","\t  "))]
    calls=re.findall(r"→ TOOL_CALL (\w+)", resp)
    if free or not calls: return "text"
    if all(READ.match(c) and not DEVICE.match(c) for c in calls): return "readonly"
    return "write/other"
d=pd.read_parquet("runs/apb_eval.parquet")
q={json.loads(l)["id"]:(json.loads(l)["p"] or 0)>=.5 for l in open("runs/cc/ru_qwen_v3f.jsonl")}
d["k"]=d.response.map(kind); d["q"]=d.id.map(q)
print("APB-ru:"); print(d[d.label.isin([0,1])].groupby(["k","label"]).q.agg(["size","sum"]))
v=pd.read_parquet("runs/timing_431_labeled.parquet"); p=pd.read_parquet("runs/a100/s11_pred.parquet").rename(columns={"label":"pred"})
v=v.merge(p,on="id"); v["k"]=v.response.map(kind)
print("valid (gold>=0):"); print(v[v.label>=0].groupby(["k","label"]).pred.agg(["size","sum"]))
print("unlabelled early turns pred rate:"); print(v[v.label<0].groupby("k").pred.agg(["size","sum"]))
for r in v[(v.k=="readonly")&(v.label>=0)].itertuples(): print(r.id,r.label,r.pred,r.response.strip()[:150].replace("\n"," "))
