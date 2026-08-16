#!/usr/bin/env bash
# Watcher for v15 = AceGPT-v2-8B (Arabic LLaMA2 base = real diversity). Disciplined:
# reports v15-alone, 7-vote (locked), and 8-vote(+v15); flags v15 ONLY if it helps.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL=runs/acegpt-v15-2ep/final
DONE=runs/v15_autoeval_DONE.txt
LOG=runs/autoeval_v15.log
echo "[autoeval-v15] start $(date)" >> "$LOG"
while [ ! -d "$FINAL" ]; do sleep 120; done
while pgrep -f "train_qlora.*acegpt-v15" >/dev/null; do sleep 30; done
sleep 30
python3 scripts/run_infer.py --base FreedomIntelligence/AceGPT-v2-8B-Chat \
  --adapter "$FINAL" --gold data/processed_v11/dev.jsonl \
  --out results/dev_acegpt_v15.jsonl >> "$LOG" 2>&1
{
echo "=== v15 AceGPT-v2-8B ==="
python3 - <<'PY'
import sys, json
sys.path.insert(0,"baselines/leaderboard-code-v1_1")
from data_loader import load_gold; from eval_lib import evaluate
gold=load_gold("dev"); M={json.loads(l)['id']:json.loads(l) for l in open("results/dev_acegpt_v15.jsonl")}
s=evaluate([M[g['id']] for g in gold if g['id'] in M],gold)
print(f"v15 AceGPT alone  OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}")
PY
P="results/dev_allam_v10.jsonl results/dev_allam_v12.jsonl results/dev_qwen7b_v7b.jsonl results/dev_qwen7b_v7.jsonl results/dev_allam_v9.jsonl results/dev_silma_v11.jsonl results/dev_qwen7b_v8.jsonl"
echo "-- 7-vote (locked) --"
python3 scripts/ensemble_vote.py --preds $P --names v10 v12 v7b v7 v9 v11 v8 \
  --out results/dev_vote7_chk.jsonl --score data/processed_v11/dev.jsonl
echo "-- 8-vote (+v15) --"
python3 scripts/ensemble_vote.py --preds $P results/dev_acegpt_v15.jsonl --names v10 v12 v7b v7 v9 v11 v8 v15 \
  --out results/dev_vote8_acegpt.jsonl --score data/processed_v11/dev.jsonl
echo "(7-vote baseline 0.8637 | target 0.8668) -> keep v15 ONLY if 8-vote > 0.8637"
} >> "$DONE" 2>&1
echo "[autoeval-v15] DONE $(date)" >> "$LOG"
