#!/usr/bin/env python3
"""Convert the official blind TEST split (1,125 rows, no gold) into our inference
input format — the same record shape prepare_data.py emits for dev, minus gold.

id = row index in the test split (0..1124), per the Space's "How it works" tab.
candidate_tools comes from `tools_sampled`, exactly as load_dev_gold() does.

Usage: python3 scripts/prepare_blind.py [out.jsonl]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import extract_messages, to_pylist  # noqa: E402

from datasets import load_dataset  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed_blind/test.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("TuwaiqAcademy/AISA-ArabicFC", split="test")
print(f"[1/2] loaded test split: {len(ds)} rows")

recs = []
for i, r in enumerate(ds):
    context, user, _ = extract_messages(r["messages"])
    cands = [
        (t.get("function") or {}).get("name")
        for t in to_pylist(r["tools_sampled"])
        if (t.get("function") or {}).get("name")
    ]
    recs.append({
        "id": i,
        "dialect": str(r["dialect"]).lower(),
        "context": context,
        "user": user,
        "candidate_tools": cands,
    })

with open(OUT, "w", encoding="utf-8") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

n_cands = [len(r["candidate_tools"]) for r in recs]
n_empty_user = sum(1 for r in recs if not r["user"])
print(f"[2/2] wrote {len(recs)} rows -> {OUT}")
print(f"      candidate_tools per row: min={min(n_cands)} max={max(n_cands)}")
print(f"      empty user text: {n_empty_user}")
print(f"      ids contiguous 0..{len(recs)-1}: {[r['id'] for r in recs] == list(range(len(recs)))}")
