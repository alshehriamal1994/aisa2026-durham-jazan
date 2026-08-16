#!/usr/bin/env bash
# CROSS-LINGUAL TRANSFER: co-train Qwen2.5-7B on Arabic + 12k English glaive FC,
# then eval on ARABIC dev. Tests if English tool-use competence transfers to Arabic.
# Restores the clean 27-tool registry before inference (apples-to-apples vs qC=0.796).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/xling.log; DONE=runs/xling_DONE.txt
: > "$DONE"; echo "[xling] start $(date)" | tee -a "$LOG"
python3 scripts/train_qlora.py --model Qwen/Qwen2.5-7B-Instruct \
  --train-file data/processed_v13/train_xling.jsonl \
  --epochs 1 --max-len 1024 --batch 1 --grad-accum 32 --lora-r 32 --lora-alpha 64 --liger \
  --save-steps 400 --out runs/qwen25-xling >> "$LOG" 2>&1 || { echo "TRAIN CRASH" | tee -a "$DONE"; exit 1; }
# restore clean 27-tool registry for fair Arabic eval
cp data/processed/tools_registry.json.bak data/processed/tools_registry.json
python3 scripts/run_infer.py --base Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen25-xling/final \
  --gold data/processed_v13/dev.jsonl --out results/dev_qwen25_xling.jsonl --batch 6 >> "$LOG" 2>&1 \
  || { echo "INFER CRASH" | tee -a "$DONE"; exit 1; }
python3 - >> "$DONE" 2>&1 <<'PY'
import sys,json
sys.path.insert(0,"baselines/leaderboard-code-v1_3"); import normalize
import pandas as pd
from collections import defaultdict
df=pd.read_parquet("data/raw/aisa_v1_3/data/dev-00000-of-00001.parquet")
g=[]
for i,row in df.iterrows():
    a=[m for m in row['messages'] if m.get('role')=='assistant'][-1]; tc=a.get('tool_calls')
    d=(row.get('dialect') or '?').lower()
    if tc is not None and len(tc)>0:
        fn=tc[0]['function']; args={k:v for k,v in (fn.get('arguments') or {}).items() if v not in (None,'')}
        g.append((int(i),fn.get('name'),args,d))
P={}
for l in open("results/dev_qwen25_xling.jsonl"):
    l=l.strip()
    if l: r=json.loads(l); P[r['id']]=r
agg=defaultdict(lambda:[0,0]); tot=[0,0]
for i,t,a,d in g:
    ok=normalize.args_match((P.get(i) or {}).get('arguments'),a,t); agg[d][1]+=1; tot[1]+=1
    if ok: agg[d][0]+=1; tot[0]+=1
print(f"CROSS-LINGUAL Qwen2.5 v1.3 ArgEM={tot[0]/tot[1]:.4f}  (vs Arabic-only qC=0.796)")
for d,(c,n) in sorted(agg.items()): print(f"   {d:10s} {c/n:.3f} ({c}/{n})")
print("weak-dialect baseline: egyptian 0.758, levantine 0.761")
PY
echo "[xling] DONE $(date)" | tee -a "$LOG"
