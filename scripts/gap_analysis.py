#!/usr/bin/env python3
"""Gap analysis vs the current development leader (ArgEM 0.808).

Answers, on the official v1.1 evaluator:
  1. Single-model ArgEM for each locked member (historical: this script predates the six-member submission).
  2. Our robust vote ArgEM, and the ORACLE ArgEM (any model right).
  3. Of the positives our VOTE misses: how many are oracle-recoverable
     (>=1 model right) split by agreement = majority-recoverable vs
     minority-only, vs all-wrong (truly irreducible by selection).
  4. For the oracle-recoverable-but-vote-missed cases, which (tool,arg)
     keys differ and whether the winning models share a base => is there a
     PRINCIPLED selection rule (not dev-tuned weights)?
"""
import json, sys, collections
sys.path.insert(0, "baselines/leaderboard-code-v1_1")
from normalize import args_match  # noqa

CONFIG = json.load(open("scripts/ensemble_config.json"))
MODELS = CONFIG["models"]
ORDER = CONFIG["tiebreak_order"]
BASE = {m["name"]: m["base"].split("/")[-1] for m in MODELS}

# map model name -> pred file
PRED = {
    "v10": "results/dev_allam_v10.jsonl",
    "v12": "results/dev_allam_v12.jsonl",
    "v7b": "results/dev_qwen7b_v7b.jsonl",
    "v7":  "results/dev_qwen7b_v7.jsonl",
    "v9":  "results/dev_allam_v9.jsonl",
    "v11": "results/dev_silma_v11.jsonl",
    "v8":  "results/dev_qwen7b_v8.jsonl",
}

def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            x = json.loads(line)
            out[x["id"]] = x
    return out

gold = load("data/processed_v11/dev.jsonl")
positives = [g for g in gold.values() if g.get("requires_function")]
npos = len(positives)

# load models that exist
preds = {}
for name, path in PRED.items():
    try:
        preds[name] = load(path)
    except FileNotFoundError:
        print(f"!! missing pred file for {name}: {path}")

print(f"positives = {npos}; models loaded = {list(preds)}\n")

# per-model argem
print("== single-model ArgEM ==")
model_right = {}  # name -> set(ids it gets right)
for name in preds:
    right = set()
    for g in positives:
        gid = g["id"]
        p = preds[name].get(gid)
        if p and args_match(p.get("arguments"), g.get("arguments")):
            right.add(gid)
    model_right[name] = right
    print(f"  {name:4s} ({BASE[name]:22s}) ArgEM = {len(right)/npos:.4f}  ({len(right)}/{npos})")

# oracle
oracle = set()
for g in positives:
    gid = g["id"]
    if any(gid in model_right[n] for n in preds):
        oracle.add(gid)
print(f"\n== ORACLE ArgEM = {len(oracle)/npos:.4f} ({len(oracle)}/{npos}) ==")
print(f"   leader target = 0.808 ({round(0.808*npos)}/{npos})  |  our robust vote ~0.774\n")

# vote (replicate plain per-arg majority via the same logic as ensemble_vote at arg level
# but here we only need: did the MAJORITY answer match gold? Approximate vote-correct as
# "the value chosen by >=ceil(active/2) models matches gold". We instead reuse the produced
# vote file for ground truth of what the combiner actually output.
vote = load("results/dev_vote7_locked.jsonl")
vote_right = set()
for g in positives:
    gid = g["id"]
    p = vote.get(gid)
    if p and args_match(p.get("arguments"), g.get("arguments")):
        vote_right.add(gid)
print(f"== locked 7-vote ArgEM = {len(vote_right)/npos:.4f} ({len(vote_right)}/{npos}) ==\n")

# cases vote misses but oracle has => the recoverable-by-better-selection set
gap = oracle - vote_right
print(f"== VOTE-MISSED but ORACLE-RECOVERABLE = {len(gap)} cases (max honest headroom from selection) ==")
maj_recoverable = []   # majority of models right -> vote SHOULD get; investigate why it didn't
minority_only = []     # <half right -> majority vote structurally cannot reach
for gid in gap:
    n_right = sum(1 for n in preds if gid in model_right[n])
    if n_right * 2 > len(preds):
        maj_recoverable.append((gid, n_right))
    else:
        minority_only.append((gid, n_right))
print(f"   majority-right but vote missed (BUG/tiebreak fixable) = {len(maj_recoverable)}")
print(f"   minority-only (vote structurally cannot reach)        = {len(minority_only)}\n")

# all-wrong = truly irreducible
allwrong = set(g["id"] for g in positives) - oracle
print(f"== ALL-WRONG (no model right; irreducible by any selection) = {len(allwrong)} ==")
tool_keys = collections.Counter()
for gid in allwrong:
    g = gold[gid]
    for k in (g.get("arguments") or {}):
        tool_keys[(g["tool_called"], k)] += 1
print("   top (tool,arg) among all-wrong:")
for (t, k), c in tool_keys.most_common(12):
    print(f"     {c:3d}  {t}.{k}")

# which (tool,arg) drive the minority-only set, and which base solves them
print("\n== minority-only recoverable: which base model is the one getting it right? ==")
base_solver = collections.Counter()
mo_keys = collections.Counter()
for gid, nr in minority_only:
    g = gold[gid]
    solvers = [n for n in preds if gid in model_right[n]]
    for n in solvers:
        base_solver[BASE[n]] += 1
    for k in (g.get("arguments") or {}):
        mo_keys[(g["tool_called"], k)] += 1
print("   base of the solving model (counts, may double-count multi-solver):")
for b, c in base_solver.most_common():
    print(f"     {c:3d}  {b}")
print("   top (tool,arg) in minority-only set:")
for (t, k), c in mo_keys.most_common(12):
    print(f"     {c:3d}  {t}.{k}")
