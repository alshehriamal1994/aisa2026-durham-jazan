#!/usr/bin/env bash
# Hamilton 8 — build a venv (pinned to match the machine that produced our 0.818
# system). Run ONCE on the login node (it has internet).
set -e
module load python/3.12.6
module load cuda/12.4.1
VENV="${1:-/nobackup/$USER/aisa-venv}"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
# CUDA 12.4 wheels — match the local env exactly
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==5.5.0 peft==0.18.0 bitsandbytes==0.49.0 \
            datasets==4.5.0 accelerate==1.12.0 sentencepiece protobuf pandas pyarrow
pip install liger-kernel || echo "liger optional"
echo "venv ready at $VENV"
echo "activate with: module load python/3.12.6 cuda/12.4.1 && source $VENV/bin/activate"
