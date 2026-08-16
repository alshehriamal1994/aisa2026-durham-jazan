#!/usr/bin/env python3
"""Sanity-check a submission JSONL before upload. Usage:
    python3 scripts/check_submission.py results/blind_submission.jsonl [--expect N]
Exits non-zero if anything looks wrong."""
import argparse, json, sys, collections

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--expect", type=int, default=None, help="expected row count")
args = ap.parse_args()

rows, problems = [], []
ids = []
for ln, line in enumerate(open(args.file, encoding="utf-8"), 1):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception as e:
        problems.append(f"line {ln}: bad JSON ({e})"); continue
    rows.append(r)
    for k in ("id", "tool_called", "arguments", "think"):
        if k not in r:
            problems.append(f"id {r.get('id','?')}: missing field '{k}'")
    ids.append(r.get("id"))
    tc = r.get("tool_called")
    if not isinstance(tc, str) or not tc:
        problems.append(f"id {r.get('id','?')}: empty/invalid tool_called")
    if not isinstance(r.get("arguments"), dict):
        problems.append(f"id {r.get('id','?')}: arguments not a dict")
    if tc == "none" and r.get("arguments"):
        problems.append(f"id {r.get('id','?')}: tool 'none' but arguments non-empty")

dup = [i for i, c in collections.Counter(ids).items() if c > 1]
if dup:
    problems.append(f"duplicate ids: {dup[:10]}{'...' if len(dup) > 10 else ''}")
if args.expect is not None and len(rows) != args.expect:
    problems.append(f"row count {len(rows)} != expected {args.expect}")

calls = sum(1 for r in rows if r.get("tool_called") not in (None, "", "none"))
print(f"rows={len(rows)}  calls={calls}  none={len(rows)-calls}  unique_ids={len(set(ids))}")
if problems:
    print(f"\n❌ {len(problems)} PROBLEM(S):")
    for p in problems[:40]:
        print("  -", p)
    sys.exit(1)
print("✅ submission looks valid")
