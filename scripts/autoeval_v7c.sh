#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/qwen7b-v7c-copy/final
DONE=runs/v7c_autoeval_DONE.txt
LOG=runs/autoeval_v7c.log
echo "[autoeval-v7c] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 120; done
while pgrep -f "train_qlora.*v7c-copy" >/dev/null; do sleep 30; done
sleep 20
python3 scripts/run_infer.py --adapter "$FINAL" --gold data/processed_v11/dev.jsonl \
  --out results/dev_qwen7b_v7c.jsonl >> "$LOG" 2>&1
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json, collections
sys.path.insert(0,"baselines/leaderboard-code-v1_1")
from data_loader import load_gold, parse_predictions
from eval_lib import evaluate
from normalize import args_match
gold=load_gold("dev")
preds,err=parse_predictions("results/dev_qwen7b_v7c.jsonl")
print("=== v7c (copy-verbatim prompt, 1 epoch) — official v1.1 evaluator ===")
if err: print("ERR",err)
else:
    s=evaluate(preds,gold)
    print(f"v7c   OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
    print("our best v7b   OverallA=0.8529  ArgEM=0.756")
    print("TARGET         OverallA=0.8668  ArgEM=0.778")
    v="*** BEATS TARGET -> SUBMIT ***" if s['overall_a']>0.8668 else ("beats our v7b, escalate to 2ep" if s['overall_a']>0.8529 else "copy-prompt did NOT help (1ep)")
    print("VERDICT:",v)
    # did the copy fields improve?
    pm={p['id']:p for p in preds}
    for fld in ['product_name','medication_name','items','specialty','category']:
        pos=[g for g in gold if g['requires_function'] and fld in (g.get('arguments') or {})]
        ok=sum(1 for g in pos if args_match(pm.get(g['id'],{}).get('arguments'),g['arguments']))
        if pos: print(f"   {fld:16s} {ok}/{len(pos)} = {ok/len(pos):.3f}")
PY
echo "[autoeval-v7c] DONE $(date)" >> "$LOG"
