#!/usr/bin/env bash
# BIG-IDEA TEST: does training on CANONICAL targets transfer better to an UNSEEN
# dialect than SURFACE targets? Both models train on ZERO Maghrebi; the 140 held-out
# Maghrebi rows are the test. A=surface gold, B=canonical gold. Self-gating on GPU.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/canon_exp.log; DONE=runs/canon_exp_DONE.txt
: > "$DONE"; echo "[canon] start $(date)" | tee -a "$LOG"

wait_gpu(){ while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${u:-9999}" -lt 3000 ] && break; sleep 60; done; }

train_infer(){ # $1=trainfile $2=tag
  echo "[canon] wait GPU for $2..." | tee -a "$LOG"; wait_gpu
  echo "[canon] TRAIN $2 ($1) $(date)" | tee -a "$LOG"
  python3 scripts/train_qlora.py --model Qwen/Qwen2.5-7B-Instruct --train-file "$1" \
    --epochs 1 --max-len 1024 --batch 1 --grad-accum 32 --lora-r 32 --lora-alpha 64 --liger \
    --save-steps 400 --out "runs/$2" >> "$LOG" 2>&1 || { echo "$2 TRAIN CRASH" | tee -a "$DONE"; return 1; }
  python3 scripts/run_infer.py --base Qwen/Qwen2.5-7B-Instruct --adapter "runs/$2/final" \
    --gold data/processed_v13/test_mag.jsonl --out "results/test_mag_$2.jsonl" --batch 6 >> "$LOG" 2>&1 \
    || { echo "$2 INFER CRASH" | tee -a "$DONE"; return 1; }
}

train_infer data/processed_v13/train_no_mag.jsonl       canon_A_surface || exit 1
train_infer data/processed_v13/train_no_mag_canon.jsonl canon_B_canon   || exit 1

python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json
sys.path.insert(0,"baselines/leaderboard-code-v1_3"); import normalize
gold=[json.loads(l) for l in open("data/processed_v13/test_mag.jsonl")]
G={r['id']:r for r in gold}
def score(tag):
    P={}
    for l in open(f"results/test_mag_{tag}.jsonl"):
        l=l.strip()
        if l: r=json.loads(l); P[r['id']]=r
    c=sum(normalize.args_match((P.get(i) or {}).get('arguments'), G[i]['gold_args'], G[i]['gold_name']) for i in G)
    return c/len(G), c
sa,sc=score("canon_A_surface"); ba,bc=score("canon_B_canon")
print("=== HELD-OUT MAGHREBI (zero-shot dialect transfer, 140 rows) ===")
print(f"  A surface-trained : ArgEM={sa:.4f} ({sc}/140)")
print(f"  B canonical-trained: ArgEM={ba:.4f} ({bc}/140)")
print(f"  => canonical {'HELPS' if ba>sa else 'does NOT help'} unseen-dialect transfer by {ba-sa:+.4f}")
print("  (if B>A: training on meaning, not spelling, generalizes to a dialect never seen — the big-idea signal)")
PY
echo "[canon] DONE $(date)" | tee -a "$LOG"
