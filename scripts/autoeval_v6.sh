#!/usr/bin/env bash
# Wait for v6 training to finish, then eval BOTH the mid (~1.3ep) and final
# (2ep) checkpoints on dev and write a compact summary + DONE sentinel.
# Durable: setsid+nohup so it survives terminal/session close.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FINAL=runs/qwen7b-v6-r64-2ep/final
MID=runs/qwen7b-v6-mid-keep
DONE=runs/v6_autoeval_DONE.txt
LOG=runs/autoeval_v6.log

echo "[autoeval] start $(date)" >> "$LOG"

# 1) wait for training to write the final adapter
while [ ! -d "$FINAL" ]; do sleep 120; done
# settle: make sure training process released the GPU
while pgrep -f "train_qlora.py" >/dev/null; do sleep 30; done
sleep 20
echo "[autoeval] v6 finished, GPU free $(date)" >> "$LOG"

# 2) eval mid (~1.3 epoch) then final (2 epoch)
python3 scripts/run_infer.py --adapter "$MID" \
  --out results/dev_qwen7b_v6_mid.jsonl   >> "$LOG" 2>&1
python3 scripts/run_infer.py --adapter "$FINAL" \
  --out results/dev_qwen7b_v6_final.jsonl >> "$LOG" 2>&1

# 3) compact summary (Overall A is the prize metric; v5 baseline = 0.808)
python3 - <<'PY' >> "$DONE" 2>&1
import json
def row(tag, path):
    try:
        s = json.load(open(path))
        return f"{tag:12s} OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}  OverallB={s['overall_b']:.4f}"
    except Exception as e:
        return f"{tag:12s} ERROR: {e}"
print("=== v6 auto-eval results ===")
print(row("v5 (#1)",   "results/dev_qwen7b_v5.scores.json"))
print(row("v6-mid~1.3ep","results/dev_qwen7b_v6_mid.scores.json"))
print(row("v6-final 2ep","results/dev_qwen7b_v6_final.scores.json"))
print("PRIZE METRIC = Overall A. Keep v6 only if it beats v5's 0.8080.")
PY
echo "[autoeval] DONE $(date)" >> "$LOG"
