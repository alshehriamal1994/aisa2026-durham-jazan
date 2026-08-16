#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/qwen7b-v8-full/final
DONE=runs/v8_autoeval_DONE.txt
LOG=runs/autoeval_v8.log
echo "[autoeval-v8] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 180; done
while pgrep -f "train_qlora.*v8-full" >/dev/null; do sleep 30; done
sleep 20
python3 scripts/run_infer.py --adapter "$FINAL" --gold data/processed_v11/dev.jsonl \
  --out results/dev_qwen7b_v8.jsonl >> "$LOG" 2>&1
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json
sys.path.insert(0,"baselines/leaderboard-code-v1_1")
from data_loader import load_gold, parse_predictions
from eval_lib import evaluate
gold=load_gold("dev")
preds,err=parse_predictions("results/dev_qwen7b_v8.jsonl")
print("=== v8 (FULL data: shared-task + parent, 1 epoch) — official v1.1 evaluator ===")
if err: print("ERR",err)
else:
    s=evaluate(preds,gold)
    print(f"v8    OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
    print("our best v7b   OverallA=0.8529  ArgEM=0.756")
    print("TARGET         OverallA=0.8668  ArgEM=0.778")
    v=("*** BEATS TARGET -> SUBMIT ***" if s['overall_a']>0.8668 else
       "beats our v7b (parent data helped!) -> 2-epoch next" if s['overall_a']>0.8529 else
       "parent data did NOT help -> keep v7b, try stronger base")
    print("VERDICT:",v)
PY
echo "[autoeval-v8] DONE $(date)" >> "$LOG"
