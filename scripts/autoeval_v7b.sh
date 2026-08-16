#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/qwen7b-v7b-2ep/final
DONE=runs/v7b_autoeval_DONE.txt
LOG=runs/autoeval_v7b.log
echo "[autoeval-v7b] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 120; done
while pgrep -f "train_qlora.*v7b-2ep" >/dev/null; do sleep 30; done
sleep 20
python3 scripts/run_infer.py --adapter "$FINAL" --gold data/processed_v11/dev.jsonl \
  --out results/dev_qwen7b_v7b.jsonl >> "$LOG" 2>&1
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json
sys.path.insert(0,"baselines/leaderboard-code-v1_1")
from data_loader import load_gold, parse_predictions
from eval_lib import evaluate
gold=load_gold("dev")
preds,err=parse_predictions("results/dev_qwen7b_v7b.jsonl")
print("=== v7b (2-epoch) — official v1.1 evaluator ===")
if err: print("ERR",err)
else:
    s=evaluate(preds,gold)
    print(f"v7b   OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
    print("v7 (current #1, submitted) OverallA=0.8493  ArgEM=0.750")
    print("VERDICT:", "BETTER than v7 -> resubmit" if s['overall_a']>0.8493 else "not better -> keep v7")
PY
echo "[autoeval-v7b] DONE $(date)" >> "$LOG"
