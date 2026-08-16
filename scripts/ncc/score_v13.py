"""Standalone v1.3 scorer for NCC (no leaderboard server needed).
Usage: python scripts/ncc/score_v13.py results/dev_<tag>.jsonl <tag>
Scores single-model ArgEM/FnAcc/OverallA against the local v1.3 dev parquet.
"""
import sys, json
sys.path.insert(0, "baselines/leaderboard-code-v1_3")
import normalize
import pandas as pd

pred_path = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "model"
df = pd.read_parquet("data/raw/aisa_v1_3/dev-00000-of-00001.parquet") \
     if __import__("os").path.exists("data/raw/aisa_v1_3/dev-00000-of-00001.parquet") \
     else pd.read_parquet("data/raw/aisa_v1_3/data/dev-00000-of-00001.parquet")
gold = []
for i, row in df.iterrows():
    a = [m for m in row["messages"] if m.get("role") == "assistant"][-1]
    tc = a.get("tool_calls")
    if tc is not None and len(tc) > 0:
        fn = tc[0]["function"]; args = {k: v for k, v in (fn.get("arguments") or {}).items() if v not in (None, "")}
        gold.append((int(i), fn.get("name"), args, True))
    else:
        gold.append((int(i), "none", {}, False))
P = {}
for l in open(pred_path):
    l = l.strip()
    if l:
        r = json.loads(l); P[r["id"]] = r
pos = [g for g in gold if g[3]]
argem = sum(normalize.args_match((P.get(i) or {}).get("arguments"), a, t) for i, t, a, _ in pos) / len(pos)
fnacc = sum(((P.get(i) or {}).get("tool_called") or "none") == t for i, t, a, _ in gold) / len(gold)
think = sum(1 for i, _, _, _ in gold if ((P.get(i) or {}).get("think") or "").strip()) / len(gold)
print(f"{tag}: ArgEM={argem:.4f}  FnAcc={fnacc:.4f}  OverallA={0.4*fnacc+0.6*argem:.4f}  "
      f"OverallB={0.3*fnacc+0.5*argem+0.2*think:.4f}  ThinkRate={think:.2f}  "
      f"(beat: vote 0.818 / AFF 0.87 ArgEM)")
