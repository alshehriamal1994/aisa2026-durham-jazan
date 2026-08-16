#!/usr/bin/env bash
#SBATCH --job-name=aisa-bigfc
#SBATCH --partition=cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --time=1-00:00:00
#SBATCH --output=aisa_%j.log
# Hamilton 8 — 1 whole H200 NVL (144GB GPU, 16 cores, 273GB RAM auto-allocated;
# do NOT set --cpus-per-task/--mem on the cuda queue). cuda walltime cap = 3 days
# (raise -t to 3-00:00:00 for 72B if needed). Trains a BIG base QLoRA on CLEAN
# v1.3 data, infers dev, scores under the v1.3 evaluator.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python/3.12.6 cuda/12.4.1
source "${VENV:-/nobackup/$USER/aisa-venv}/bin/activate"
export HF_HOME="${HF_HOME:-/nobackup/$USER/hf_cache}"
export HF_HUB_OFFLINE=1                  # weights pre-staged on the login node
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
TAG="${TAG:-qwen32b}"
TRAIN=data/processed_v13/train_st_aug.jsonl
DEV=data/processed_v13/dev.jsonl
# 144GB H200 -> big batch for speed.
BATCH="${BATCH:-8}"; GA="${GA:-4}"; MAXLEN="${MAXLEN:-1024}"

echo "[ham] $(date) model=$MODEL tag=$TAG batch=$BATCH ga=$GA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 scripts/train_qlora.py --model "$MODEL" --train-file "$TRAIN" \
  --epochs 1 --max-len "$MAXLEN" --batch "$BATCH" --grad-accum "$GA" \
  --lora-r 32 --lora-alpha 64 --liger --save-steps 200 --out "runs/${TAG}-clean"

python3 scripts/run_infer.py --base "$MODEL" --adapter "runs/${TAG}-clean/final" \
  --gold "$DEV" --out "results/dev_${TAG}.jsonl" --batch "$BATCH"

python3 scripts/ncc/score_v13.py "results/dev_${TAG}.jsonl" "$TAG"
echo "[ham] DONE $(date) -> results/dev_${TAG}.jsonl ; adapter runs/${TAG}-clean/final"
