#!/usr/bin/env python3
"""Dump the oracle-recoverable-but-vote-missed cases (minority-only) with full
query + every model's candidate answer + gold, so a reasoning judge can assess
how many are recoverable from the query (verifier ceiling) vs arbitrary noise.
"""
import json, sys, collections
sys.path.insert(0, "baselines/leaderboard-code-v1_1")
from normalize import args_match  # noqa

def load(p):
    o = {}
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if l: x = json.loads(l); o[x["id"]] = x
    return o

gold = load("data/processed_v11/dev.jsonl")
PRED = {"v10":"results/dev_allam_v10.jsonl","v12":"results/dev_allam_v12.jsonl",
        "v7b":"results/dev_qwen7b_v7b.jsonl","v7":"results/dev_qwen7b_v7.jsonl",
        "v9":"results/dev_allam_v9.jsonl","v11":"results/dev_silma_v11.jsonl",
        "v8":"results/dev_qwen7b_v8.jsonl"}
preds = {n: load(p) for n, p in PRED.items()}
vote = load("results/dev_vote7_locked.jsonl")

positives = [g for g in gold.values() if g.get("requires_function")]
out = []
for g in positives:
    gid = g["id"]
    # vote wrong?
    pv = vote.get(gid, {})
    if args_match(pv.get("arguments"), g.get("arguments")):
        continue
    # oracle right? (>=1 model)
    right_models = [n for n in preds if preds[n].get(gid) and args_match(preds[n][gid].get("arguments"), g.get("arguments"))]
    if not right_models:
        continue  # all-wrong, not in scope
    out.append((gid, g, right_models))

print(f"=== {len(out)} oracle-recoverable, vote-missed cases ===\n")
for gid, g, right_models in out:
    print(f"id {gid}  [{g.get('dialect')}]  tool={g['tool_called']}  (correct models: {','.join(right_models)})")
    print(f"  USER: {g.get('user','')[:200]}")
    # distinct candidate arg-dicts
    by_args = collections.defaultdict(list)
    for n in preds:
        p = preds[n].get(gid, {})
        key = json.dumps(p.get("arguments", {}), ensure_ascii=False, sort_keys=True)
        by_args[key].append(n)
    for argstr, models in sorted(by_args.items(), key=lambda x: -len(x[1])):
        mark = "  <-- VOTE" if any(m in models for m in []) else ""
        print(f"    [{len(models)}x {','.join(models)}] {argstr}")
    print(f"  GOLD: {json.dumps(g.get('arguments',{}), ensure_ascii=False, sort_keys=True)}")
    print(f"  VOTE PICKED: {json.dumps(pv.get('arguments',{}), ensure_ascii=False, sort_keys=True)}")
    print()
