#!/usr/bin/env bash
# Push our best single base hardest: Qwen3-8B, 2 epochs, rank 64, clean v1.3 data.
# If it beats v16 (0.802 single), it's a stronger ensemble member -> lifts the tuned vote.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/q3max.log; DONE=runs/q3max_DONE.txt
: > "$DONE"; echo "[q3max] start $(date)" | tee -a "$LOG"
python3 scripts/train_qlora.py --model Qwen/Qwen3-8B \
  --train-file data/processed_v13/train_st_aug.jsonl \
  --epochs 2 --max-len 1024 --batch 1 --grad-accum 32 --lora-r 64 --lora-alpha 128 --liger \
  --save-steps 300 --out runs/qwen3-q3max >> "$LOG" 2>&1 || { echo "TRAIN CRASH" | tee -a "$DONE"; exit 1; }
python3 scripts/run_infer.py --base Qwen/Qwen3-8B --adapter runs/qwen3-q3max/final \
  --gold data/processed_v13/dev.jsonl --out results/dev_qwen3_q3max.jsonl --batch 6 >> "$LOG" 2>&1 \
  || { echo "INFER CRASH" | tee -a "$DONE"; exit 1; }
python3 - >> "$DONE" 2>&1 <<'PY'
import sys,json
sys.path.insert(0,"baselines/leaderboard-code-v1_3"); import normalize
import pandas as pd
df=pd.read_parquet("data/raw/aisa_v1_3/data/dev-00000-of-00001.parquet")
g=[]
for i,row in df.iterrows():
    a=[m for m in row['messages'] if m.get('role')=='assistant'][-1]; tc=a.get('tool_calls')
    if tc is not None and len(tc)>0:
        fn=tc[0]['function']; args={k:v for k,v in (fn.get('arguments') or {}).items() if v not in (None,'')}
        g.append((int(i),fn.get('name'),args))
P={}
for l in open("results/dev_qwen3_q3max.jsonl"):
    l=l.strip()
    if l: r=json.loads(l); P[r['id']]=r
c=sum(normalize.args_match((P.get(i) or {}).get('arguments'),a,t) for i,t,a in g)
print(f"Qwen3-8B 2ep r64 clean: SINGLE ArgEM={c/len(g):.4f} ({c}/{len(g)})  vs v16=0.802")
PY
echo "[q3max] DONE $(date)" | tee -a "$LOG"
