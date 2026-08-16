#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/allam-v9/final
DONE=runs/v9_autoeval_DONE.txt
LOG=runs/autoeval_v9.log
echo "[autoeval-v9] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 120; done
while pgrep -f "train_qlora.*allam-v9" >/dev/null; do sleep 30; done
sleep 20
python3 scripts/run_infer.py --base ALLaM-AI/ALLaM-7B-Instruct-preview --adapter "$FINAL" \
  --gold data/processed_v11/dev.jsonl --out results/dev_allam_v9.jsonl >> "$LOG" 2>&1
python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json
sys.path.insert(0,"baselines/leaderboard-code-v1_1")
from data_loader import load_gold, parse_predictions
from eval_lib import evaluate
gold=load_gold("dev")
preds,err=parse_predictions("results/dev_allam_v9.jsonl")
print("=== v9 = ALLaM-7B (Arabic-native base), 1 epoch — official v1.1 evaluator ===")
if err: print("ERR",err)
else:
    s=evaluate(preds,gold)
    print(f"v9-ALLaM  OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
    print("our best v7b(Qwen) OverallA=0.8529  ArgEM=0.756")
    print("TARGET                OverallA=0.8668  ArgEM=0.778")
    v=("*** BEATS TARGET -> SUBMIT ***" if s['overall_a']>0.8668 else
       "beats our v7b! ALLaM wins -> 2-epoch next" if s['overall_a']>0.8529 else
       "ALLaM 1ep below v7b -> try 2-epoch or keep Qwen v7b")
    print("VERDICT:",v)
    # per-dialect (ALLaM may help Levantine/Maghrebi where we were weak)
    from normalize import args_match
    import collections
    bd=collections.defaultdict(lambda:[0,0])
    pm={p['id']:p for p in preds}
    for g in gold:
        if not g['requires_function']: continue
        d=g.get('dialect','?'); bd[d][1]+=1
        if (pm.get(g['id'],{}).get('tool_called') or 'none')==g['tool_called'] and args_match(pm.get(g['id'],{}).get('arguments'),g['arguments']): bd[d][0]+=1
    print("per-dialect ArgEM:", {d:round(c/n,3) for d,(c,n) in sorted(bd.items())})
PY
echo "[autoeval-v9] DONE $(date)" >> "$LOG"
