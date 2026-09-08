"""
AISA-ArabicFC official evaluation library (v2 weights).

Metrics:
  FnAcc       = #(pred.tool_called == gold.tool_called) / N_total
                Negatives have gold.tool_called = "none", so FnAcc also
                penalises hallucinated calls and missed calls.
  ArgEM       = #(pred.arguments == gold.arguments) / N_positive
                Strict exact match over key-value pairs (None values filtered).
                Only computed on positive samples.
  ThinkRate   = #(non-empty pred.think) / N_total

Track A: 0.40 * FnAcc + 0.60 * ArgEM
Track B: 0.30 * FnAcc + 0.50 * ArgEM + 0.20 * ThinkRate

Dialect diagnostic (Track C): per-dialect FnAcc + ArgEM + gap (max - min).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from normalize import args_match


DIALECTS = ["msa", "gulf", "egyptian", "levantine", "maghrebi"]


def _norm_args(args: dict | None) -> dict:
    """Drop None values and stringify everything for comparison."""
    if not args:
        return {}
    return {str(k): str(v).strip() for k, v in args.items() if v is not None and v != ""}


def evaluate(predictions: list[dict], gold: list[dict]) -> dict[str, Any]:
    """
    Score predictions against gold.

    predictions: [{"id": int, "tool_called": str, "arguments": dict, "think"?: str}, ...]
    gold:        [{"id": int, "tool_called": str, "arguments": dict,
                   "dialect": str, "requires_function": bool}, ...]

    Returns a dict with overall + per-dialect metrics.
    """
    by_id = {p["id"]: p for p in predictions}

    n_total = len(gold)
    fn_correct = 0
    arg_em_correct = 0
    n_positive = 0
    think_present = 0
    missing = 0

    dialect_stats: dict[str, dict] = defaultdict(
        lambda: {"fn_correct": 0, "arg_em_correct": 0, "pos_total": 0, "total": 0}
    )

    for g in gold:
        gid = g["id"]
        dialect = (g.get("dialect") or "unknown").lower()
        dstat = dialect_stats[dialect]
        dstat["total"] += 1

        p = by_id.get(gid)
        gold_fn = g["tool_called"]
        gold_args = _norm_args(g.get("arguments"))
        is_positive = bool(g.get("requires_function"))

        if is_positive:
            n_positive += 1
            dstat["pos_total"] += 1

        if p is None:
            missing += 1
            continue

        # FnAcc — across ALL samples
        pred_fn = (p.get("tool_called") or "none").strip() or "none"
        if pred_fn == gold_fn:
            fn_correct += 1
            dstat["fn_correct"] += 1

        # ArgEM — only on positives. Normalised, fair matching (see normalize.py):
        # numbers (5000==5000.0, Arabic-Indic digits), Arabic orthography,
        # number-words, list/set fields, and closed-class bilingual aliases.
        if is_positive:
            if args_match(p.get("arguments"), g.get("arguments"), gold_fn):
                arg_em_correct += 1
                dstat["arg_em_correct"] += 1

        # ThinkRate — non-empty think text counts (>5 chars)
        think = p.get("think") or ""
        if isinstance(think, str) and len(think.strip()) > 5:
            think_present += 1

    fnacc = fn_correct / n_total if n_total else 0.0
    argem = arg_em_correct / n_positive if n_positive else 0.0
    thinkrate = think_present / n_total if n_total else 0.0

    overall_a = 0.40 * fnacc + 0.60 * argem
    overall_b = 0.30 * fnacc + 0.50 * argem + 0.20 * thinkrate

    # Per-dialect
    dialect_breakdown: dict[str, dict] = {}
    for d, s in dialect_stats.items():
        dialect_breakdown[d] = {
            "fnacc": (s["fn_correct"] / s["total"]) if s["total"] else 0.0,
            "argem": (s["arg_em_correct"] / s["pos_total"]) if s["pos_total"] else 0.0,
            "n": s["total"],
            "n_positive": s["pos_total"],
        }

    fn_values = [v["fnacc"] for v in dialect_breakdown.values() if v["n"] > 0]
    ar_values = [v["argem"] for v in dialect_breakdown.values() if v["n_positive"] > 0]
    gap_fnacc = (max(fn_values) - min(fn_values)) if fn_values else 0.0
    gap_argem = (max(ar_values) - min(ar_values)) if ar_values else 0.0

    return {
        "fnacc": fnacc,
        "argem": argem,
        "thinkrate": thinkrate,
        "overall_a": overall_a,
        "overall_b": overall_b,
        "dialect_breakdown": dialect_breakdown,
        "gap_fnacc": gap_fnacc,
        "gap_argem": gap_argem,
        "n_total": n_total,
        "n_positive": n_positive,
        "n_negative": n_total - n_positive,
        "n_predictions": len(predictions),
        "missing": missing,
    }
