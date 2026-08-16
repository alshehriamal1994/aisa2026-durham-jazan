#!/usr/bin/env bash
# Run on a LOGIN node (compute nodes usually have no internet). Pre-fetches the
# base weights into a scratch HF cache so the Slurm job can run fully OFFLINE.
# Qwen2.5-32B ~65GB, Qwen2.5-72B ~145GB on disk — point HF_HOME at scratch.
set -e
module load python/3.12.6 2>/dev/null || true
source "${VENV:-/nobackup/$USER/aisa-venv}/bin/activate" 2>/dev/null || true
export HF_HOME="${HF_HOME:-/nobackup/$USER/hf_cache}"   # Hamilton Lustre scratch
mkdir -p "$HF_HOME"
MODEL="${1:-Qwen/Qwen2.5-32B-Instruct}"
echo "downloading $MODEL into $HF_HOME ..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
echo "done. In the Slurm job, set HF_HOME=$HF_HOME and HF_HUB_OFFLINE=1"
