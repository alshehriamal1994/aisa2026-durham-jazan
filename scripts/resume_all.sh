#!/usr/bin/env bash
# Resume the two jobs the terminal crash killed, fully detached from any
# controlling terminal (setsid + nohup + </dev/null) so a closed terminal /
# dropped SSH cannot kill them again. Re-runnable: both steps resume in place.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HXET=0

# 1) ALLaM-7B weight download (network/disk only — resumes the 3 partial shards)
setsid nohup hf download ALLaM-AI/ALLaM-7B-Instruct-preview \
  --include "model-*.safetensors" "model.safetensors.index.json" \
            "config.json" "generation_config.json" \
            "tokenizer*" "special_tokens_map.json" \
  > runs/download_allam_resume.log 2>&1 < /dev/null &
echo "allam-download PID $!"

# 2) v6 training (GPU) — checkpoints every 100 steps, auto-resumes from last
setsid nohup python3 scripts/train_qlora.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train-file data/processed/train_st_aug.jsonl \
  --out runs/qwen7b-v6-r64-2ep \
  --epochs 2 --lora-r 64 --lora-alpha 128 \
  --batch 1 --grad-accum 16 --max-len 1024 \
  --save-steps 100 \
  > runs/qwen7b-v6-r64-2ep.log 2>&1 < /dev/null &
echo "v6-train PID $!"
