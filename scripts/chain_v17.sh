#!/usr/bin/env bash
# v17 = Qwen3-14B QLoRA on CLEAN v1.3 data (ceiling raiser + clean-gold retrain).
# Two justified levers at once: (1) bigger base than our best single (Qwen3-8B v16=0.802),
# (2) clean v1.3 train (331 corrupted stripped-arg rows fixed). Self-gating:
# smoke (256 rows) -> if 14B smoke fails (OOM/format) FALL BACK to Qwen3-8B on clean data
# -> full train -> infer -> score v1.3 -> test add-to-vote vs locked-7 (0.814).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/chain_v17.log
DONE=runs/v17_DONE.txt
TRAIN=data/processed_v13/train_st_aug.jsonl
DEV=data/processed_v13/dev.jsonl
MODEL=Qwen/Qwen3-14B
TAG=v17
: > "$DONE"
echo "[v17] start $(date)" | tee -a "$LOG"

run_smoke () {  # $1=model
  hf download "$1" >> "$LOG" 2>&1 || return 2
  python3 scripts/train_qlora.py --model "$1" --train-file "$TRAIN" \
    --max-train 256 --epochs 1 --max-len 768 --batch 1 --grad-accum 4 --liger \
    --out runs/${TAG}-smoke >> "$LOG" 2>&1 || return 1
  python3 scripts/run_infer.py --base "$1" --adapter runs/${TAG}-smoke/final \
    --gold "$DEV" --out results/dev_${TAG}_smoke.jsonl --limit 60 --batch 4 >> "$LOG" 2>&1 || return 1
  python3 - "$TAG" <<'PY'
import json, sys
sys.path.insert(0, "baselines/leaderboard-code-v1_3")
import pandas as pd
df = pd.read_parquet("data/raw/aisa_v1_3/data/dev-00000-of-00001.parquet")
def gtool(i):
    a=[m for m in df.iloc[i]['messages'] if m.get('role')=='assistant'][-1]
    tc=a.get('tool_calls');
    return (tc[0]['function'].get('name') if tc is not None and len(tc)>0 else 'none')
preds=[json.loads(l) for l in open(f"results/dev_{sys.argv[1]}_smoke.jsonl") if l.strip()]
n=ok=nz=0
for p in preds:
    n+=1
    if (p.get('tool_called') or 'none')==gtool(p['id']): ok+=1
    if p.get('tool_called') not in (None,'','none'): nz+=1
fn=ok/n if n else 0
sys.stderr.write(f"smoke n={n} FnAcc={fn:.2f} calls={nz}\n")
sys.exit(0 if (fn>0.5 and nz>n*0.4) else 3)
PY
}

echo "[v17] SMOKE Qwen3-14B..." | tee -a "$LOG"
if run_smoke "$MODEL"; then
  echo "[v17] 14B smoke PASS" | tee -a "$LOG"
else
  rc=$?
  echo "[v17] 14B smoke FAILED rc=$rc -> FALLBACK Qwen3-8B on clean data" | tee -a "$LOG" "$DONE"
  MODEL=Qwen/Qwen3-8B; TAG=v17b
  : > runs/v17b_DONE.txt
  run_smoke "$MODEL" || { echo "FALLBACK SMOKE ALSO FAILED" | tee -a "$DONE"; exit 1; }
  echo "[v17] fallback 8B smoke PASS" | tee -a "$LOG"
fi

# FULL train (liger keeps the LM-head logits off-VRAM; batch1/ga32 for 16GB)
echo "[v17] FULL train $MODEL on CLEAN v1.3 (tag=$TAG)..." | tee -a "$LOG"
python3 scripts/train_qlora.py --model "$MODEL" --train-file "$TRAIN" \
  --epochs 1 --max-len 1024 --batch 1 --grad-accum 32 --lora-r 32 --lora-alpha 64 --liger \
  --save-steps 200 --out runs/${TAG}-clean >> "$LOG" 2>&1 || { echo "FULL TRAIN CRASHED" | tee -a "$DONE"; exit 1; }

echo "[v17] infer full dev..." | tee -a "$LOG"
python3 scripts/run_infer.py --base "$MODEL" --adapter runs/${TAG}-clean/final \
  --gold "$DEV" --out results/dev_${TAG}.jsonl --batch 6 >> "$LOG" 2>&1 \
  || { echo "FULL INFER CRASHED" | tee -a "$DONE"; exit 1; }

# Score v1.3: single-model + does it help the vote?
TAG="$TAG" python3 - >> "$DONE" 2>&1 <<'PY'
import os, sys, json, itertools
sys.path.insert(0, "baselines/leaderboard-code-v1_3")
import normalize
import pandas as pd
TAG=os.environ["TAG"]
df=pd.read_parquet("data/raw/aisa_v1_3/data/dev-00000-of-00001.parquet")
gold=[]
for i,row in df.iterrows():
    a=[m for m in row['messages'] if m.get('role')=='assistant'][-1]
    tc=a.get('tool_calls')
    if tc is not None and len(tc)>0:
        fn=tc[0]['function']; args={k:v for k,v in (fn.get('arguments') or {}).items() if v not in (None,'')}
        gold.append({'id':int(i),'tool':fn.get('name'),'args':args,'pos':True})
    else: gold.append({'id':int(i),'tool':'none','args':{},'pos':False})
pos=[g for g in gold if g['pos']]
def load(p):
    d={}
    for l in open(p):
        l=l.strip()
        if l: r=json.loads(l); d[r['id']]=r
    return d
files={"v10":"results/dev_allam_v10.jsonl","v12":"results/dev_allam_v12.jsonl",
  "v7b":"results/dev_qwen7b_v7b.jsonl","v7":"results/dev_qwen7b_v7.jsonl",
  "v9":"results/dev_allam_v9.jsonl","v11":"results/dev_silma_v11.jsonl",
  "v8":"results/dev_qwen7b_v8.jsonl", TAG:f"results/dev_{TAG}.jsonl"}
M={k:load(v) for k,v in files.items()}
def em(args,g): return normalize.args_match(args,g['args'],g['tool'])
c=sum(em((M[TAG].get(g['id']) or {}).get('arguments'),g) for g in pos)
print(f"{TAG} SINGLE v1.3 ArgEM={c/len(pos):.4f} ({c}/{len(pos)})  vs best-single v16=0.802")
# simple per-arg majority vote helper
def vote_pred(models,gid):
    from collections import Counter
    tools=Counter((M[m].get(gid) or {}).get('tool_called','none') for m in models)
    tool=tools.most_common(1)[0][0]
    args={}
    keys=Counter()
    for m in models:
        a=(M[m].get(gid) or {}).get('arguments') or {}
        for k in a: keys[k]+=1
    for k,_ in keys.items():
        vals=Counter(normalize.canon_value((M[m].get(gid) or {}).get('arguments',{}).get(k,''),k) for m in models if k in ((M[m].get(gid) or {}).get('arguments') or {}))
        # pick raw value matching the winning canon
        win=vals.most_common(1)[0][0]
        for m in models:
            a=(M[m].get(gid) or {}).get('arguments') or {}
            if k in a and normalize.canon_value(a[k],k)==win: args[k]=a[k]; break
    return {'tool_called':tool,'arguments':args}
LOCKED7=["v10","v12","v7b","v7","v9","v11","v8"]
def vote_em(models):
    c=0
    for g in pos:
        vp=vote_pred(models,g['id'])
        if vp['tool_called']==g['tool'] and em(vp['arguments'],g): c+=1
    return c/len(pos)
base=vote_em(LOCKED7)
withnew=vote_em(LOCKED7+[TAG])
print(f"VOTE locked-7 v1.3 ArgEM={base:.4f}  -> +{TAG} = {withnew:.4f}  ({'HELPS' if withnew>base else 'hurts/flat'})")
PY
echo "[v17] DONE $(date)" | tee -a "$LOG"
