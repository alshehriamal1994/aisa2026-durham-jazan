# AISA-ArabicFC — Big-model run on Durham NCC

Goal: train a **32B–72B** base (QLoRA, clean v1.3 data) to crack the entity walls
our 8B models miss → honest ceiling-raiser for dev **and** the blind test.
Target: single-model ArgEM 0.80 → ~0.84–0.87 (vote/oracle up accordingly).

## 0. Transfer the bundle
```bash
scp aisa_ncc_bundle.tar.gz <user>@ncc-login:~/
ssh <user>@ncc-login && tar xzf aisa_ncc_bundle.tar.gz && cd aisa_ncc
```

## 1. Build the env (login node, once)
```bash
bash scripts/ncc/setup_env.sh aisa      # adjust the CUDA module line to NCC's
```
Discover NCC specifics if unsure:
```bash
sinfo -o "%P %G %l"        # partitions, GPU types, max walltime
module avail cuda 2>&1 | head
```

## 2. Pre-download weights (login node — compute nodes have no internet)
```bash
export HF_HOME=$HOME/scratch/hf_cache          # point at SCRATCH (72B ~145GB!)
bash scripts/ncc/predownload.sh Qwen/Qwen2.5-32B-Instruct
# later: bash scripts/ncc/predownload.sh Qwen/Qwen2.5-72B-Instruct
```

## 3. Submit the training job
```bash
export HF_HOME=$HOME/scratch/hf_cache ENV_NAME=aisa
# 32B (recommended first — validates the gain in ~half the time):
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-32B-Instruct,TAG=qwen32b,BATCH=4,GA=8 scripts/ncc/train.sbatch
# 72B (the stretch — set batch low):
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct,TAG=qwen72b,BATCH=1,GA=16 scripts/ncc/train.sbatch
# Qwen3-32B (reasoning / Track B + ensemble diversity):
sbatch --export=ALL,MODEL=Qwen/Qwen3-32B,TAG=qwen3-32b,BATCH=2,GA=16 scripts/ncc/train.sbatch
```
The job trains → infers dev → prints the v1.3 score in `aisa_<jobid>.log`.

## 4. Bring results back
```bash
scp -r <user>@ncc-login:~/aisa_ncc/runs/qwen32b-clean/final  ./runs/qwen32b-clean-final
scp     <user>@ncc-login:~/aisa_ncc/results/dev_qwen32b.jsonl ./results/
```
Then locally: add it to `scripts/ensemble_config.json` and re-vote — if it lifts
the 0.818 vote, it's in. If single ArgEM ≥ ~0.85, it may even stand alone.

## Recipe notes (80GB)
| Model | 4-bit weights | BATCH/GA | ~time (1ep, 12k rows) |
|---|---|---|---|
| Qwen2.5-32B | ~18 GB | 4 / 8 | ~6–10 h |
| Qwen2.5-72B | ~38 GB | 1 / 16 | ~16–24 h |
| Qwen3-32B | ~18 GB | 2 / 16 | ~8–12 h |

- `--liger` keeps the 151k-vocab logits off-VRAM (already in the sbatch).
- Multi-GPU: add `#SBATCH --gres=gpu:2`; HF `device_map="auto"` shards 72B across both.
- Walltime hit? `train_qlora` saves every 200 steps → resume from the last ckpt.

## Compliance
Open-weights base (Qwen, Apache-2.0), official train data only, no dev/test gold
leakage. Fully disclosable in the system paper. Unconstrained task — legal.
