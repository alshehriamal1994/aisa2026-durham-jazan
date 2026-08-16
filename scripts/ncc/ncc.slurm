#!/bin/bash
#SBATCH --job-name=aisa32b
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=aisa-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
mkdir -p data/processed; cp -n data/processed_v13/tools_registry.json data/processed/
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SIF=$HOME/pytorch.sif
M=${MODEL:-Qwen/Qwen2.5-32B-Instruct}; T=${TAG:-qwen32b}; B=${BATCH:-2}; G=${GA:-16}
singularity exec --nv $SIF python3 scripts/train_qlora.py --model "$M" --train-file data/processed_v13/train_st_aug.jsonl --epochs 1 --max-len 1024 --batch $B --grad-accum $G --lora-r 32 --lora-alpha 64 --save-steps 2000 --out runs/${T}-clean
singularity exec --nv $SIF python3 scripts/run_infer.py --base "$M" --adapter runs/${T}-clean/final --gold data/processed_v13/dev.jsonl --out results/dev_${T}.jsonl --batch $B
singularity exec --nv $SIF python3 scripts/ncc/score_v13.py results/dev_${T}.jsonl $T
