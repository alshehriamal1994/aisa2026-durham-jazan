#!/usr/bin/env bash
# Unattended chain: wait until (a) ALLaM weights finished downloading AND
# (b) v6 training released the GPU, then fine-tune ALLaM on the winning v5
# recipe. Launched detached (setsid) so it survives terminal/SSH close.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BLOBS=/home/amal/.cache/huggingface/hub/models--ALLaM-AI--ALLaM-7B-Instruct-preview/blobs

echo "[chain] start $(date)"

# 1) wait for ALLaM download (no .incomplete blobs left)
while ls "$BLOBS"/*.incomplete >/dev/null 2>&1; do sleep 60; done
echo "[chain] ALLaM download complete $(date)"

# 2) wait for v6 to finish so the GPU is free (final adapter written)
while [ ! -d runs/qwen7b-v6-r64-2ep/final ]; do sleep 120; done
echo "[chain] v6 finished, GPU free $(date)"
sleep 30

# 3) fine-tune ALLaM (v5 winning recipe: r32/a64, 1 epoch), checkpoints every 100
python3 scripts/train_qlora.py \
  --model ALLaM-AI/ALLaM-7B-Instruct-preview \
  --train-file data/processed/train_st_aug.jsonl \
  --out runs/allam7b-v1-st-aug \
  --epochs 1 --lora-r 32 --lora-alpha 64 \
  --batch 1 --grad-accum 16 --max-len 1024 \
  --save-steps 100
echo "[chain] ALLaM training done $(date)"
