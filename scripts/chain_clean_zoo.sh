#!/usr/bin/env bash
# Refresh the PROVEN workhorses on CLEAN v1.3 data (331-row corruption fix = uniform
# uplift). Runs AFTER v17b frees the GPU. ALLaM-7B + Qwen2.5-7B, v7/v10 recipe (r32/a64,
# 1 epoch) on data/processed_v13/train_st_aug.jsonl. Infers full dev + scores v1.3.
# Set-and-forget; verdicts in runs/clean_zoo_DONE.txt. These are proven bases (no smoke gate).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/chain_clean_zoo.log
DONE=runs/clean_zoo_DONE.txt
TRAIN=data/processed_v13/train_st_aug.jsonl
DEV=data/processed_v13/dev.jsonl
: > "$DONE"
echo "[clean_zoo] start $(date)" | tee -a "$LOG"

wait_gpu_free () {  # wait until no training proc + <3GB GPU used
  while pgrep -f "train_qlora.py" | grep -qv $$ ; do
    if ! pgrep -f "v17b-clean" >/dev/null && ! pgrep -f "v17-clean" >/dev/null; then break; fi
    sleep 60
  done
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-9999}" -lt 3000 ] && break
    sleep 60
  done
}

train_one () {  # $1=base  $2=tag  $3=epochs
  echo "[clean_zoo] wait GPU for $2..." | tee -a "$LOG"; wait_gpu_free
  echo "[clean_zoo] TRAIN $2 ($1) clean ${3}ep $(date)" | tee -a "$LOG"
  python3 scripts/train_qlora.py --model "$1" --train-file "$TRAIN" \
    --epochs "$3" --max-len 1024 --batch 1 --grad-accum 32 --lora-r 32 --lora-alpha 64 --liger \
    --save-steps 300 --out runs/${2}-clean >> "$LOG" 2>&1 || { echo "$2 TRAIN CRASHED" | tee -a "$DONE"; return 1; }
  echo "[clean_zoo] INFER $2 $(date)" | tee -a "$LOG"
  python3 scripts/run_infer.py --base "$1" --adapter runs/${2}-clean/final \
    --gold "$DEV" --out results/dev_${2}_clean.jsonl --batch 6 >> "$LOG" 2>&1 \
    || { echo "$2 INFER CRASHED" | tee -a "$DONE"; return 1; }
  # score single v1.3
  TAG="$2" python3 - >> "$DONE" 2>&1 <<'PY'
import os,sys,json
sys.path.insert(0,"baselines/leaderboard-code-v1_3"); import normalize
import pandas as pd
df=pd.read_parquet("data/raw/aisa_v1_3/data/dev-00000-of-00001.parquet")
gold=[]
for i,row in df.iterrows():
    a=[m for m in row['messages'] if m.get('role')=='assistant'][-1]; tc=a.get('tool_calls')
    if tc is not None and len(tc)>0:
        fn=tc[0]['function']; args={k:v for k,v in (fn.get('arguments') or {}).items() if v not in (None,'')}
        gold.append((int(i),fn.get('name'),args))
tag=os.environ["TAG"]
P={}
for l in open(f"results/dev_{tag}_clean.jsonl"):
    l=l.strip()
    if l: r=json.loads(l); P[r["id"]]=r
c=sum(normalize.args_match((P.get(i) or {}).get('arguments'),a,t) for i,t,a in gold)
print(f"{tag} CLEAN single v1.3 ArgEM={c/len(gold):.4f} ({c}/{len(gold)})")
PY
  echo "[clean_zoo] $2 done $(date)" | tee -a "$LOG"
}

train_one "ALLaM-AI/ALLaM-7B-Instruct-preview" "allam" 1
train_one "Qwen/Qwen2.5-7B-Instruct" "qwen25" 1
echo "[clean_zoo] ALL DONE $(date)" | tee -a "$LOG" "$DONE"
