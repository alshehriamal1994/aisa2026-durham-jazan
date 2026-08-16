#!/usr/bin/env bash
# Watcher for v11=SILMA: wait for training's final adapter AND the training
# process to fully exit (so the GPU is free) before running inference + 6-vote.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/silma-v11-2ep/final
DONE=runs/v11_autoeval_DONE.txt
LOG=runs/autoeval_v11.log
: > "$DONE"   # clear the bogus prior verdict
echo "[autoeval-v11] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 120; done
while pgrep -f "train_qlora.*silma-v11" >/dev/null; do sleep 30; done
sleep 30   # let CUDA fully release
python3 scripts/run_infer.py --base silma-ai/SILMA-9B-Instruct-v1.0 \
  --adapter "$FINAL" --gold data/processed_v11/dev.jsonl \
  --out results/dev_silma_v11.jsonl >> "$LOG" 2>&1
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
       'v10':"results/dev_allam_v10.jsonl",'v11':"results/dev_silma_v11.jsonl"}
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
def score(keys,label,outfile=None):
    present=[k for k in keys if M.get(k)]
    ens=[vote(g['id'],present) for g in gold]; ens=[e for e in ens if e]
    if outfile: open(outfile,"w").write("\n".join(json.dumps(e,ensure_ascii=False) for e in ens)+"\n")
    s=evaluate(ens,gold)
    print(f"{label:20s} [{'+'.join(present)}]  OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
    return s['overall_a']
print("=== v11 SILMA-9B + 6-vote ===")
if M.get('v11'):
    s11=evaluate([M['v11'][g['id']] for g in gold if g['id'] in M['v11']],gold)
    print(f"v11 SILMA alone      OverallA={s11['overall_a']:.4f}  ArgEM={s11['argem']:.4f}")
a5=score(['v7b','v7','v9','v8','v10'],"5-vote (no SILMA)")
a6=score(['v7b','v7','v9','v8','v10','v11'],"6-vote (ALL)","results/dev_vote6.jsonl")
best=max(a5,a6)
print(f"baseline: 4-vote 0.8577 | 5-vote 0.8589 | target 0.8668 | best now {best:.4f}")
print("VERDICT:", "*** BEATS TARGET -> SUBMIT results/dev_vote6.jsonl ***" if best>0.8668 else f"still short by {0.8668-best:.4f}; add more diverse models")
PY
echo "[autoeval-v11] DONE $(date)" >> "$LOG"
