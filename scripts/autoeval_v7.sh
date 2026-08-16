#!/usr/bin/env bash
# Wait for v7 (v1.1 retrain) to finish, run inference, score with the OFFICIAL
# v1.1 fair evaluator, and write a summary vs the current leaderboard target (0.8301).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/qwen7b-v7-v11/final
DONE=runs/v7_autoeval_DONE.txt
LOG=runs/autoeval_v7.log
echo "[autoeval-v7] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 120; done
while pgrep -f "train_qlora.*v7-v11" >/dev/null; do sleep 30; done
sleep 20
echo "[autoeval-v7] v7 done, running inference $(date)" >> "$LOG"
# generate predictions (inputs identical to old dev; gold args differ only)
python3 scripts/run_infer.py --adapter "$FINAL" \
  --gold data/processed_v11/dev.jsonl \
  --out results/dev_qwen7b_v7.jsonl >> "$LOG" 2>&1
# score with the official v1.1 evaluator
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json
sys.path.insert(0, "baselines/leaderboard-code-v1_1")
from data_loader import load_gold, parse_predictions
from eval_lib import evaluate
from normalize import args_match
gold = load_gold("dev")
gmap = {g['id']: g for g in gold}
print("=== v7 (v1.1 retrain) — scored with the OFFICIAL v1.1 fair evaluator ===")
preds, err = parse_predictions("results/dev_qwen7b_v7.jsonl")
if err:
    print("PARSE ERROR:", err)
else:
    s = evaluate(preds, gold)
    print(f"v7        OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}  OverallB={s['overall_b']:.4f}")
    print("---- targets ----")
    print("leaderboard target   OverallA=0.8301  ArgEM=0.718")
    print("second target     OverallA=0.8272  ArgEM=0.712")
    print("our v5 (re-eval) OverallA=0.8248  ArgEM=0.708")
    verdict = "*** BEATS TARGET ***" if s['overall_a'] > 0.8301 else \
              ("#2/#3 zone" if s['overall_a'] > 0.8248 else "no improvement vs v5")
    print("VERDICT:", verdict)
    # end_of_service recovery check (was 0.174 under v1.1 with old model)
    pm = {p['id']: p for p in preds}
    pos = [g for g in gold if g['requires_function'] and g['tool_called']=='calculate_end_of_service']
    ok = sum(1 for g in pos if args_match(pm.get(g['id'],{}).get('arguments'), g['arguments']))
    print(f"calculate_end_of_service ArgEM = {ok}/{len(pos)} = {ok/len(pos):.3f}  (was 0.174 pre-retrain)")
PY
echo "[autoeval-v7] DONE $(date)" >> "$LOG"
