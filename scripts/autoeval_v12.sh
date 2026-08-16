#!/usr/bin/env bash
# Watcher for v12 = ALLaM-7B r64 (strong diverse sibling of v10). Wait for the
# training proc to fully exit (GPU free) before inference + 7-vote.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/allam-v12-r64-2ep/final
DONE=runs/v12_autoeval_DONE.txt
LOG=runs/autoeval_v12.log
echo "[autoeval-v12] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 120; done
while pgrep -f "train_qlora.*allam-v12" >/dev/null; do sleep 30; done
sleep 30
python3 scripts/run_infer.py --base ALLaM-AI/ALLaM-7B-Instruct-preview \
  --adapter "$FINAL" --gold data/processed_v11/dev.jsonl \
  --out results/dev_allam_v12.jsonl >> "$LOG" 2>&1
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json, collections
sys.path.insert(0,"baselines/leaderboard-code-v1_1")
from data_loader import load_gold, parse_predictions
from eval_lib import evaluate
from normalize import canon_value, args_match
gold=load_gold("dev"); pos=[g for g in gold if (g.get('tool_called') or 'none')!='none']
def load(p):
    try: return {x['id']:x for x in parse_predictions(p)[0]}
    except: return {}
files={'v7b':"results/dev_qwen7b_v7b.jsonl",'v7':"results/dev_qwen7b_v7.jsonl",
       'v9':"results/dev_allam_v9.jsonl",'v8':"results/dev_qwen7b_v8.jsonl",
       'v10':"results/dev_allam_v10.jsonl",'v11':"results/dev_silma_v11.jsonl",
       'v12':"results/dev_allam_v12.jsonl"}
M={k:load(v) for k,v in files.items()}
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
def score(keys,label,out=None):
    present=[k for k in keys if M.get(k)]
    ens=[vote(g['id'],present) for g in gold]; ens=[e for e in ens if e]
    if out: open(out,"w").write("\n".join(json.dumps(e,ensure_ascii=False) for e in ens)+"\n")
    s=evaluate(ens,gold)
    print(f"{label:16s} [{'+'.join(present)}]  OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}")
    return s['overall_a']
print("=== v12 ALLaM-r64 + 7-vote ===")
if M.get('v12'):
    s=evaluate([M['v12'][g['id']] for g in gold if g['id'] in M['v12']],gold)
    print(f"v12 alone        OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}")
    # does v12 raise the oracle?
    others=['v7b','v7','v9','v8','v10','v11']
    def corr(m,g):
        p=M[m].get(g['id']); return p and p.get('tool_called')==g['tool_called'] and args_match(p.get('arguments'),g.get('arguments'))
    orc6=sum(any(corr(m,g) for m in others) for g in pos)/len(pos)
    orc7=sum(any(corr(m,g) for m in others+['v12']) for g in pos)/len(pos)
    print(f"oracle 6-model={orc6:.4f} -> 7-model={orc7:.4f} (does v12 add new-correct cases?)")
score(['v7b','v7','v9','v8','v10','v11'],"6-vote (prev)")
b=score(['v7b','v7','v9','v8','v10','v11','v12'],"7-vote (ALL)","results/dev_vote7.jsonl")
# also try best strong-subset: drop weak SILMA
b2=score(['v7b','v7','v9','v8','v10','v12'],"6-vote (no SILMA)","results/dev_vote6b.jsonl")
best=max(b,b2)
print(f"baseline 6-vote 0.8601 | target 0.8668 | best now {best:.4f}")
print("VERDICT:", "*** BEATS TARGET -> SUBMIT ***" if best>0.8668 else f"still short by {0.8668-best:.4f}")
PY
echo "[autoeval-v12] DONE $(date)" >> "$LOG"
