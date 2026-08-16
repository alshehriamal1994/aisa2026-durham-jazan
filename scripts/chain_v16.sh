#!/usr/bin/env bash
# v16 = Qwen3-8B QLoRA (generational upgrade of our Qwen2.5 workhorse; oracle-lift swing).
# Self-gating: wait download -> SMOKE (256 rows) -> abort if format/template broken
# -> full train (v7 recipe) -> infer -> score v1.1 -> 8-vote vs locked 7-vote 0.8637.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=Qwen/Qwen3-8B
LOG=runs/chain_v16.log
DONE=runs/v16_DONE.txt
TRAIN=data/processed_v11/train_st_aug.jsonl
: > "$DONE"
echo "[v16] start $(date)" | tee -a "$LOG"

# 1) ensure weights present (idempotent; returns cached path when complete)
echo "[v16] ensuring $MODEL is downloaded..." | tee -a "$LOG"
hf download "$MODEL" >> "$LOG" 2>&1 || { echo "DOWNLOAD FAILED" | tee -a "$DONE"; exit 1; }

# 2) SMOKE: tiny train + infer, verify our format survives Qwen3's chat template
echo "[v16] SMOKE train (256 rows)..." | tee -a "$LOG"
python3 scripts/train_qlora.py --model "$MODEL" --train-file "$TRAIN" \
  --max-train 256 --epochs 1 --max-len 1024 --batch 1 --grad-accum 4 \
  --out runs/qwen3-smoke >> "$LOG" 2>&1 || { echo "SMOKE TRAIN CRASHED" | tee -a "$DONE"; exit 1; }
python3 scripts/run_infer.py --base "$MODEL" --adapter runs/qwen3-smoke/final \
  --gold data/processed_v11/dev.jsonl --out results/dev_qwen3_smoke.jsonl --limit 60 >> "$LOG" 2>&1 \
  || { echo "SMOKE INFER CRASHED" | tee -a "$DONE"; exit 1; }

SMOKE_OK=$(python3 - <<'PY'
import json, sys
sys.path.insert(0, "baselines/leaderboard-code-v1_1")
from data_loader import load_gold
gold = {g["id"]: g for g in load_gold("dev")}
preds = [json.loads(l) for l in open("results/dev_qwen3_smoke.jsonl", encoding="utf-8") if l.strip()]
n = ok = nonempty = 0
for p in preds:
    g = gold.get(p["id"])
    if not g: continue
    n += 1
    if (p.get("tool_called") or "none") == g["tool_called"]: ok += 1
    if p.get("tool_called") not in (None, "", "none"): nonempty += 1
fnacc = ok / n if n else 0
print(f"smoke n={n} FnAcc={fnacc:.2f} nonempty_calls={nonempty}", file=sys.stderr)
# format is healthy if it parses real calls and gets a decent chunk of tools right
print(1 if (fnacc > 0.5 and nonempty > n * 0.4) else 0)
PY
)
echo "[v16] SMOKE_OK=$SMOKE_OK" | tee -a "$LOG"
if [ "$SMOKE_OK" != "1" ]; then
  echo "SMOKE FAILED — Qwen3 chat-template likely fights our <think>{json} format (needs enable_thinking=False). NOT running full train. See $LOG." | tee -a "$DONE"
  exit 1
fi

# 3) FULL train: v7 recipe (r32/a64, 1 epoch), conservative VRAM for 8B on 16GB
echo "[v16] FULL train Qwen3-8B r32/a64 1ep..." | tee -a "$LOG"
python3 scripts/train_qlora.py --model "$MODEL" --train-file "$TRAIN" \
  --epochs 1 --max-len 1024 --batch 1 --grad-accum 32 --lora-r 32 --lora-alpha 64 \
  --save-steps 200 --out runs/qwen3-v16 >> "$LOG" 2>&1 || { echo "FULL TRAIN CRASHED" | tee -a "$DONE"; exit 1; }

# 4) infer on full dev + score + 8-vote
python3 scripts/run_infer.py --base "$MODEL" --adapter runs/qwen3-v16/final \
  --gold data/processed_v11/dev.jsonl --out results/dev_qwen3_v16.jsonl >> "$LOG" 2>&1 \
  || { echo "FULL INFER CRASHED" | tee -a "$DONE"; exit 1; }

python3 - >> "$DONE" 2>&1 <<'PY'
import sys, json, collections
sys.path.insert(0, "baselines/leaderboard-code-v1_1")
from data_loader import load_gold
from eval_lib import evaluate
from normalize import canon_value
gold = load_gold("dev")
def load(p):
    try: return {x["id"]: x for x in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}
    except Exception: return {}
files = {"v10":"results/dev_allam_v10.jsonl","v12":"results/dev_allam_v12.jsonl",
         "v7b":"results/dev_qwen7b_v7b.jsonl","v7":"results/dev_qwen7b_v7.jsonl",
         "v9":"results/dev_allam_v9.jsonl","v11":"results/dev_silma_v11.jsonl",
         "v8":"results/dev_qwen7b_v8.jsonl","v16":"results/dev_qwen3_v16.jsonl"}
M = {k: load(v) for k, v in files.items()}
order = ["v16","v10","v12","v7b","v7","v9","v11","v8"]
def vote(gid, keys):
    preds = [M[m][gid] for m in keys if M.get(m) and gid in M[m]]
    if not preds: return None
    tc = collections.Counter(p.get("tool_called") or "none" for p in preds)
    best = tc.most_common(1)[0][0]
    active = [p for p in preds if (p.get("tool_called") or "none") == best]
    if best == "none": return {"id":gid,"tool_called":"none","arguments":{},"think":""}
    kc = collections.Counter()
    for p in active:
        for k in (p.get("arguments") or {}): kc[k]+=1
    args = {}
    for k, cnt in kc.items():
        if cnt < len(active)/2: continue
        cl = collections.defaultdict(list)
        for p in active:
            v = (p.get("arguments") or {}).get(k)
            if v is not None and str(v).strip(): cl[canon_value(v,k)].append(v)
        if cl: args[k] = max(cl.items(), key=lambda kv: len(kv[1]))[1][0]
    return {"id":gid,"tool_called":best,"arguments":args,"think":""}
def score(keys, label):
    present = [k for k in keys if M.get(k)]
    ens = [e for e in (vote(g["id"], present) for g in gold) if e]
    s = evaluate(ens, gold)
    print(f"{label:18s} [{'+'.join(present)}]  OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")
    return s
print("=== v16 Qwen3-8B verdict ===")
s16 = evaluate([M["v16"][g["id"]] for g in gold if g["id"] in M["v16"]], gold)
print(f"v16 ALONE           OverallA={s16['overall_a']:.4f}  ArgEM={s16['argem']:.4f}  FnAcc={s16['fnacc']:.4f}  (best single so far=0.7580)")
s7 = score(["v10","v12","v7b","v7","v9","v11","v8"], "7-vote (locked)")
s8 = score(order, "8-vote (+v16)")
print(f"\n7-vote ArgEM={s7['argem']:.4f} -> 8-vote ArgEM={s8['argem']:.4f}  (delta {s8['argem']-s7['argem']:+.4f})")
print("KEEP v16 in ensemble ONLY if 8-vote OverallA > 0.8637." )
print("VERDICT:", "ADD v16 (helps)" if s8["overall_a"] > 0.8637 else "DROP v16 (no help)")
PY
echo "[v16] DONE $(date)" | tee -a "$LOG"
