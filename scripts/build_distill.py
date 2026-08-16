"""Build an ensemble-distillation training set.

Votes (majority + enum-validity) over the strong-6 models' predictions on the
TRAIN inputs, and writes a copy of train_st_aug.jsonl with gold_name/gold_args/
gold_think REPLACED by the ensemble target. The student trained on this mimics
the 6-model consensus in one model (deployable single for the blind set).
Negatives keep gold 'none'. think = the chosen model's trace (Track B).
"""
import sys, json, re
from collections import Counter, defaultdict
sys.path.insert(0, "baselines/leaderboard-code-v1_3")
import normalize

REG = json.load(open("data/processed_v13/tools_registry.json"))
def fmeta(t, k):
    p = ((REG.get(t, {}) or {}).get("parameters", {}) or {}).get("properties", {}).get(k, {}) or {}
    m = re.search(r"Supported values:\s*([^).]+)", p.get("description", "") or "")
    return p.get("type", "string"), ([x.strip() for x in m.group(1).split(",")] if m else None)

MODELS = sys.argv[1:]  # list of pred jsonl paths, strongest first
def load(p):
    d = {}
    for l in open(p):
        l = l.strip()
        if l:
            r = json.loads(l); d[r["id"]] = r
    return d
P = [load(m) for m in MODELS]

def vote(gid):
    preds = [(pf.get(gid) or {}) for pf in P]
    tc = Counter(p.get("tool_called", "none") for p in preds)
    top = max(tc.values())
    tool = next(t for t in [p.get("tool_called", "none") for p in preds] if tc[t] == top)
    sel = [p for p in preds if p.get("tool_called", "none") == tool]
    args = {}; keys = Counter()
    for p in sel:
        for k in (p.get("arguments") or {}): keys[k] += 1
    for k, cnt in keys.items():
        if cnt < max(1, len(sel) / 2): continue
        clu = defaultdict(list)
        for r, p in enumerate(sel):
            a = p.get("arguments") or {}
            if k in a: clu[normalize.canon_value(a[k], k)].append((r, a[k]))
        if not clu: continue
        ck = lambda it: (len(it[1]), -min(r for r, _ in it[1]))
        ranked = sorted(clu.items(), key=ck, reverse=True); chosen = ranked[0]
        typ, en = fmeta(tool, k)
        if en:
            ven = [normalize.canon_value(e, k) for e in en]
            val = [c for c in ranked if c[0] in ven]
            if val: chosen = max(val, key=ck)
        args[k] = chosen[1][0][1]
    # think: from the first selected model that produced one
    think = next((p.get("think") for p in sel if (p.get("think") or "").strip()), "")
    return tool, args, think

rows = [json.loads(l) for l in open("data/processed_v13/train_st_aug_id.jsonl") if l.strip()]
out = open("data/processed_v13/distill_train.jsonl", "w")
n_pos = n_changed = 0
for gid, r in enumerate(rows):
    if str(r.get("requires_function")).lower() == "true" or r.get("gold_name") not in (None, "none", ""):
        tool, args, think = vote(gid)
        if tool == "none":
            # ensemble says no-call; trust gold to avoid dropping a positive
            pass
        else:
            n_pos += 1
            old = (r.get("gold_name"), r.get("gold_args"))
            r["gold_name"] = tool
            r["gold_args"] = json.dumps(args, ensure_ascii=False)
            if think: r["gold_think"] = think
            if old != (r["gold_name"], r["gold_args"]): n_changed += 1
    out.write(json.dumps(r, ensure_ascii=False) + "\n")
out.close()
print(f"distill_train.jsonl written: {len(rows)} rows, {n_pos} ensemble-labeled positives, "
      f"{n_changed} targets differ from gold")
