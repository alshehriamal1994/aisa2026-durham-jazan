#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/allam-v10-2ep/final
DONE=runs/v10_autoeval_DONE.txt
LOG=runs/autoeval_v10.log
echo "[autoeval-v10] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 180; done
while pgrep -f "train_qlora.*allam-v10" >/dev/null; do sleep 30; done
sleep 20
python3 scripts/run_infer.py --base ALLaM-AI/ALLaM-7B-Instruct-preview --adapter "$FINAL" \
  --gold data/processed_v11/dev.jsonl --out results/dev_allam_v10.jsonl >> "$LOG" 2>&1
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json, collections
sys.path.insert(0,"baselines/leaderboard-code-v1_1")
from data_loader import load_gold, parse_predictions
from eval_lib import evaluate
from normalize import canon_value
gold=load_gold("dev")
def load(p):
    try: return {x['id']:x for x in parse_predictions(p)[0]}
    except: return {}
files={'v7b':"results/dev_qwen7b_v7b.jsonl",'v7':"results/dev_qwen7b_v7.jsonl",
       'v9':"results/dev_allam_v9.jsonl",'v8':"results/dev_qwen7b_v8.jsonl",
       'v10':"results/dev_allam_v10.jsonl"}
M={k:load(v) for k,v in files.items()}
order=['v10','v7b','v9','v7','v8']
def vote(gid,keys):
    preds=[M[m][gid] for m in keys if gid in M[m] and M[m]]
    if not preds: return None
    tc=collections.Counter(p.get('tool_called') or 'none' for p in preds)
    best=tc.most_common(1)[0][0]
    active=[p for p in preds if (p.get('tool_called') or 'none')==best]
    if best=='none': return {'id':gid,'tool_called':'none','arguments':{},'think':''}
    kc=collections.Counter()
    for p in active:
        for k in (p.get('arguments') or {}): kc[k]+=1
    args={}
    for k,cnt in kc.items():
        if cnt<len(active)/2: continue
        cl=collections.defaultdict(list)
        for p in active:
            v=(p.get('arguments') or {}).get(k)
            if v is not None and str(v).strip(): cl[canon_value(v,k)].append(v)
        if cl: args[k]=max(cl.items(),key=lambda kv:len(kv[1]))[1][0]
    return {'id':gid,'tool_called':best,'arguments':args,'think':''}
# v10 alone
s10=evaluate([M['v10'][g['id']] for g in gold if g['id'] in M['v10']],gold)
print("=== v10 ALLaM-2ep ===")
print(f"v10 alone   OverallA={s10['overall_a']:.4f}  ArgEM={s10['argem']:.4f}")
# 5-model vote
ens=[vote(g['id'],['v7b','v7','v9','v8','v10']) for g in gold]
ens=[e for e in ens if e]
import json as J
open("results/dev_vote5.jsonl","w").write("\n".join(J.dumps(e,ensure_ascii=False) for e in ens)+"\n")
s=evaluate(ens,gold)
print(f"5-MODEL VOTE OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
print("prev 4-vote 0.8577 | target 0.8668")
print("VERDICT:", "*** BEATS TARGET -> SUBMIT results/dev_vote5.jsonl ***" if s['overall_a']>0.8668 else "still climbing, add more models")
PY
echo "[autoeval-v10] DONE $(date)" >> "$LOG"
