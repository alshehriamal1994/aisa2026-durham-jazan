#!/usr/bin/env bash
# Blind-test driver for AISA-ArabicFC. Runs the 6 locked members on an input
# file, then per-argument majority votes -> a submission JSONL.
#
#   bash scripts/run_blind.sh <input.jsonl> <out_submission.jsonl>
#
# <input.jsonl> = the (blind) test split with the same INPUT fields as dev
#   (id, user, context, candidate_tools, dialect...). Gold not required.
# On the development split this reproduces ArgEM 0.822 and Overall A 0.8925
# against v1.4 gold with the official evaluator.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IN="${1:-data/processed_v11/dev.jsonl}"
OUT="${2:-results/blind_submission.jsonl}"
BATCH="${BATCH:-8}"   # conservative; 16GB OOMs if another GPU process co-tenants. Override: BATCH=16 ...
PD="results/blind_preds"
mkdir -p "$PD"
echo "[blind] input=$IN  out=$OUT  batch=$BATCH"

# Refuse to start if the GPU is already heavily occupied (the 06-18 OOM cause).
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${FREE_MIB:-0}" -lt 10000 ]; then
  echo "[blind] WARNING: only ${FREE_MIB}MiB GPU free (<10GB). Another process may be running; risk of OOM." >&2
fi

# (name|base|adapter|predfile) — must match scripts/ensemble_config.json
MODELS=(
  "v10|ALLaM-AI/ALLaM-7B-Instruct-preview|runs/allam-v10-2ep/final|$PD/dev_allam_v10.jsonl"
  "v12|ALLaM-AI/ALLaM-7B-Instruct-preview|runs/allam-v12-r64-2ep/final|$PD/dev_allam_v12.jsonl"
  "v7b|Qwen/Qwen2.5-7B-Instruct|runs/qwen7b-v7b-2ep/final|$PD/dev_qwen7b_v7b.jsonl"
  "v7|Qwen/Qwen2.5-7B-Instruct|runs/qwen7b-v7-v11/final|$PD/dev_qwen7b_v7.jsonl"
  "aC|ALLaM-AI/ALLaM-7B-Instruct-preview|runs/allam-clean/final|$PD/dev_allam_clean.jsonl"
  "qC|Qwen/Qwen2.5-7B-Instruct|runs/qwen25-clean/final|$PD/dev_qwen25_clean.jsonl"
)

PREDS=(); NAMES=()
for m in "${MODELS[@]}"; do
  IFS="|" read -r name base adapter out <<< "$m"
  if [ ! -f "$out" ]; then
    echo "[blind] inferring $name ($base)"
    python3 scripts/run_infer.py --base "$base" --adapter "$adapter" --gold "$IN" --out "$out" --batch "$BATCH"
  else
    echo "[blind] $name already done ($out) — skip"
  fi
  PREDS+=("$out"); NAMES+=("$name")
done

echo "[blind] voting -> $OUT"
# --train-conventions enables BOTH train-derived rules: the date-year rule (+0.004 dev)
# and the omit rule (+0.022 dev), together +0.026 against v1.4 gold and -0.002 against
# v1.1/v1.3 gold. Drop the flag for the pure vote. Set RULES="--no-omit-rule" (or
# --no-year-rule) to reproduce the individual submission variants in predictions/.
python3 scripts/ensemble_vote.py --preds "${PREDS[@]}" --names "${NAMES[@]}" --out "$OUT" \
  --train-conventions ${RULES:-} --input "$IN"
echo "[blind] DONE -> $OUT"
