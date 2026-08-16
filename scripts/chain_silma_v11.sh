#!/usr/bin/env bash
# Unattended chain for v11 = SILMA-9B (6th model for the vote ensemble).
# 1) wait until SILMA download is fully complete (0 .incomplete, all 5 shards)
# 2) wait until v10 releases the GPU
# 3) QLoRA fine-tune SILMA (v7b recipe: r32/a64, 2 epochs) on v1.1 data
# 4) inference --base SILMA -> results/dev_silma_v11.jsonl
# 5) compute 6-model vote (v7b,v7,v9,v8,v10,v11) -> results/dev_vote6.jsonl
# Launched detached (setsid). Re-runnable: training auto-resumes from last ckpt.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/home/amal/.cache/huggingface/hub/models--silma-ai--SILMA-9B-Instruct-v1.0
LOG=runs/chain_silma_v11.log
DONE=runs/v11_autoeval_DONE.txt
echo "[chain-v11] start $(date)" >> "$LOG"

# 1) wait for download: no .incomplete blobs AND all 5 shards resolvable in a snapshot
while : ; do
  inc=$(ls "$B"/blobs/*.incomplete 2>/dev/null | wc -l)
  shards=$(ls -L "$B"/snapshots/*/model-0000?-of-00005.safetensors 2>/dev/null | wc -l)
  [ "$inc" -eq 0 ] && [ "$shards" -eq 5 ] && break
  sleep 60
done
echo "[chain-v11] SILMA download complete (5 shards, 0 incomplete) $(date)" >> "$LOG"

# 2) wait for GPU free (v10 training finished)
while pgrep -f "train_qlora.*allam-v10" >/dev/null; do sleep 60; done
while [ ! -d runs/allam-v10-2ep/final ]; do sleep 60; done   # ensure v10 actually produced its adapter
sleep 30
echo "[chain-v11] GPU free, starting SILMA training $(date)" >> "$LOG"

# 3) fine-tune SILMA (v7b recipe, 2 epochs), checkpoints every 200 steps
python3 scripts/train_qlora.py \
  --model silma-ai/SILMA-9B-Instruct-v1.0 \
  --train-file data/processed_v11/train_st_aug.jsonl \
  --out runs/silma-v11-2ep \
  --epochs 2 --lora-r 32 --lora-alpha 64 \
  --batch 1 --grad-accum 16 --max-len 1024 \
  --save-steps 200 >> "$LOG" 2>&1
echo "[chain-v11] SILMA training done $(date)" >> "$LOG"

# 4) inference
sleep 20
python3 scripts/run_infer.py --base silma-ai/SILMA-9B-Instruct-v1.0 \
  --adapter runs/silma-v11-2ep/final \
  --gold data/processed_v11/dev.jsonl --out results/dev_silma_v11.jsonl >> "$LOG" 2>&1

# 5) 6-model vote
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
# tie-break preference order: strongest singles first
order=['v7b','v10','v11','v7','v9','v8']
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
    ens=[vote(g['id'],keys) for g in gold]; ens=[e for e in ens if e]
    if outfile: open(outfile,"w").write("\n".join(json.dumps(e,ensure_ascii=False) for e in ens)+"\n")
    s=evaluate(ens,gold)
    print(f"{label:18s} OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
    return s['overall_a']
print("=== v11 SILMA-9B + vote sweep ===")
if 'v11' in M and M['v11']:
    s11=evaluate([M['v11'][g['id']] for g in gold if g['id'] in M['v11']],gold)
    print(f"v11 SILMA alone    OverallA={s11['overall_a']:.4f}  ArgEM={s11['argem']:.4f}")
a5=score(['v7b','v7','v9','v8','v10'],"5-vote (no SILMA)")
a6=score(['v7b','v7','v9','v8','v10','v11'],"6-vote (ALL)","results/dev_vote6.jsonl")
best=max(a5,a6)
print(f"prev 4-vote 0.8577 | target 0.8668 | best now {best:.4f}")
print("VERDICT:", "*** BEATS TARGET -> SUBMIT results/dev_vote6.jsonl ***" if best>0.8668 else "still short; add more diverse models")
PY
echo "[chain-v11] DONE $(date)" >> "$LOG"
