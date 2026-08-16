"""
Reusable evaluation harness — score any predictions file against the official
dev gold using the organisers' eval_lib. Every experiment scores identically.

Usage:
    python scripts/evaluate.py results/<preds>.jsonl [--gold data/processed/dev.jsonl]

Predictions: JSONL of {"id", "tool_called", "arguments", "think"?}.
Prints FnAcc / ArgEM / ThinkRate / Overall A+B / per-dialect, and writes
<preds>.scores.json next to the input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("baselines/leaderboard-code")))
from eval_lib import evaluate  # type: ignore


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--gold", default="data/processed/dev.jsonl")
    args = ap.parse_args()

    preds = load_jsonl(args.preds)
    gold = load_jsonl(args.gold)
    s = evaluate(preds, gold)

    print(f"\n=== {args.preds} ===")
    print(f"FnAcc        {s['fnacc']:.4f}")
    print(f"ArgEM   ★    {s['argem']:.4f}")
    print(f"ThinkRate    {s['thinkrate']:.4f}")
    print(f"Overall A    {s['overall_a']:.4f}   (0.40·FnAcc + 0.60·ArgEM)")
    print(f"Overall B    {s['overall_b']:.4f}   (0.30·FnAcc + 0.50·ArgEM + 0.20·ThinkRate)")
    print(f"missing      {s['missing']}   n_pos={s['n_positive']}  n_neg={s['n_negative']}")
    print("per-dialect (FnAcc / ArgEM):")
    for d, v in sorted(s["dialect_breakdown"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {d:<10} n={v['n']:>3} pos={v['n_positive']:>3}  {v['fnacc']:.3f} / {v['argem']:.3f}")

    out = Path(args.preds).with_suffix(".scores.json")
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
