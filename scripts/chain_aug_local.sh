#!/usr/bin/env bash
# Validate the dialectal augmentation (paraphrases generated with our own ALLaM-7B):
# train Qwen2.5-7B on train_st_aug_PLUS. Discarded, see the paper appendix.
# (base+103 aug), infer dev, score v1.3 + per-dialect vs the non-aug clean model (qC=0.796).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/aug_local.log; DONE=runs/aug_local_DONE.txt
: > "$DONE"; echo "[aug] start $(date)" | tee -a "$LOG"
python3 scripts/train_qlora.py --model Qwen/Qwen2.5-7B-Instruct \
  --train-file data/processed_v13/train_st_aug_plus.jsonl \
  --epochs 1 --max-len 1024 --batch 1 --grad-accum 32 --lora-r 32 --lora-alpha 64 --liger \
  --save-steps 300 --out runs/qwen25-aug >> "$LOG" 2>&1 || { echo "TRAIN CRASHED" | tee -a "$DONE"; exit 1; }
python3 scripts/run_infer.py --base Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen25-aug/final \
  --gold data/processed_v13/dev.jsonl --out results/dev_qwen25_aug.jsonl --batch 6 >> "$LOG" 2>&1 \
  || { echo "INFER CRASHED" | tee -a "$DONE"; exit 1; }
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
for l in open("results/dev_qwen25_aug.jsonl"):
    l=l.strip()
    if l: r=json.loads(l); P[r['id']]=r
agg=defaultdict(lambda:[0,0]); tot=[0,0]
for i,t,a,d in g:
    ok=normalize.args_match((P.get(i) or {}).get('arguments'),a,t)
    agg[d][1]+=1; tot[1]+=1
    if ok: agg[d][0]+=1; tot[0]+=1
print(f"AUG-TRAINED Qwen2.5 v1.3 ArgEM={tot[0]/tot[1]:.4f}  (vs non-aug clean qC=0.796)")
for d,(c,n) in sorted(agg.items()):
    print(f"   {d:10s} {c/n:.3f} ({c}/{n})")
print("compare weak dialects vs locked-system: egyptian 0.758, levantine 0.761")
PY
echo "[aug] DONE $(date)" | tee -a "$LOG"
