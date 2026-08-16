#!/usr/bin/env bash
# Ensemble self-distillation: strong-6 vote -> pseudo-labels on TRAIN -> train one
# Qwen2.5-7B student that mimics the 6-model consensus (deployable single for blind).
# Self-gating on GPU. Verdict in runs/distill_DONE.txt.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/chain_distill.log
DONE=runs/distill_DONE.txt
TRAIN=data/processed_v13/train_st_aug_id.jsonl
DEV=data/processed_v13/dev.jsonl
: > "$DONE"
echo "[distill] start $(date)" | tee -a "$LOG"

# strong-6 = v10,v12 (ALLaM), v7b,v7 (Qwen2.5), aC,qC (clean) — bases + adapters
declare -A BASE=(
 [v10]="ALLaM-AI/ALLaM-7B-Instruct-preview" [v12]="ALLaM-AI/ALLaM-7B-Instruct-preview"
 [v7b]="Qwen/Qwen2.5-7B-Instruct" [v7]="Qwen/Qwen2.5-7B-Instruct"
 [aC]="ALLaM-AI/ALLaM-7B-Instruct-preview" [qC]="Qwen/Qwen2.5-7B-Instruct")
declare -A ADAP=(
 [v10]="runs/allam-v10-2ep/final" [v12]="runs/allam-v12-r64-2ep/final"
 [v7b]="runs/qwen7b-v7b-2ep/final" [v7]="runs/qwen7b-v7-v11/final"
 [aC]="runs/allam-clean/final" [qC]="runs/qwen25-clean/final")
ORDER=(v10 v12 v7b v7 aC qC)

# 1) infer each model on TRAIN (skip if pred exists)
for m in "${ORDER[@]}"; do
  out="results/train_${m}.jsonl"
  if [ -s "$out" ]; then echo "[distill] $out exists, skip" | tee -a "$LOG"; continue; fi
  echo "[distill] infer $m on TRAIN $(date)" | tee -a "$LOG"
  python3 scripts/run_infer.py --base "${BASE[$m]}" --adapter "${ADAP[$m]}" \
    --gold "$TRAIN" --out "$out" --batch 10 >> "$LOG" 2>&1 \
    || { echo "INFER $m CRASHED" | tee -a "$DONE"; exit 1; }
done

# 2) vote -> pseudo-labeled distill train
echo "[distill] build pseudo-labels $(date)" | tee -a "$LOG"
python3 scripts/build_distill.py \
  results/train_v10.jsonl results/train_v12.jsonl results/train_v7b.jsonl \
  results/train_v7.jsonl results/train_aC.jsonl results/train_qC.jsonl >> "$LOG" 2>&1 \
  || { echo "BUILD_DISTILL CRASHED" | tee -a "$DONE"; exit 1; }

# 3) train student (Qwen2.5-7B, proven structured base, v7 recipe)
echo "[distill] train student $(date)" | tee -a "$LOG"
python3 scripts/train_qlora.py --model "Qwen/Qwen2.5-7B-Instruct" \
  --train-file data/processed_v13/distill_train.jsonl \
  --epochs 1 --max-len 1024 --batch 1 --grad-accum 32 --lora-r 32 --lora-alpha 64 --liger \
  --save-steps 300 --out runs/distill-qwen25 >> "$LOG" 2>&1 \
  || { echo "STUDENT TRAIN CRASHED" | tee -a "$DONE"; exit 1; }

# 4) infer dev + score vs vote 0.818
python3 scripts/run_infer.py --base "Qwen/Qwen2.5-7B-Instruct" --adapter runs/distill-qwen25/final \
  --gold "$DEV" --out results/dev_distill.jsonl --batch 6 >> "$LOG" 2>&1 \
  || { echo "STUDENT INFER CRASHED" | tee -a "$DONE"; exit 1; }
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json
sys.path.insert(0, "baselines/leaderboard-code-v1_3"); import normalize
import pandas as pd
df = pd.read_parquet("data/raw/aisa_v1_3/data/dev-00000-of-00001.parquet")
gold = []
for i, row in df.iterrows():
    a = [m for m in row['messages'] if m.get('role') == 'assistant'][-1]; tc = a.get('tool_calls')
    if tc is not None and len(tc) > 0:
        fn = tc[0]['function']; args = {k: v for k, v in (fn.get('arguments') or {}).items() if v not in (None, '')}
        gold.append((int(i), fn.get('name'), args))
P = {}
for l in open("results/dev_distill.jsonl"):
    l = l.strip()
    if l: r = json.loads(l); P[r["id"]] = r
c = sum(normalize.args_match((P.get(i) or {}).get('arguments'), a, t) for i, t, a in gold)
tr = sum(1 for i, _, _ in gold if (P.get(i) or {}).get('think', '').strip())
print(f"DISTILLED STUDENT v1.3 ArgEM={c/len(gold):.4f} ({c}/{len(gold)})  ThinkRate~{tr/len(gold):.2f}  vs vote=0.818 best-single=0.802")
PY
echo "[distill] DONE $(date)" | tee -a "$LOG"
