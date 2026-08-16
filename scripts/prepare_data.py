"""
Phase 0 — unify AISA-ArabicFC training data.

Merges the shared-task train (10.5k, carries <think> traces) with the larger,
cleaner, better-balanced parent train (45.7k), into one normalized record schema.

Outputs (data/processed/):
  train.jsonl            deduped training records (dev rows excluded)
  internal_val.jsonl     held-out slice of train for quick checkpoint selection
  dev.jsonl              the OFFICIAL dev (545) gold — our real eval, never trained on
  tools_registry.json    canonical {name: {description, parameters}} JSON schemas
  prepare_stats.json     counts + dialect breakdown + leakage report

Record schema (one JSON per line):
  {source, dialect, domain, requires_function, context, user,
   candidate_tools:[name,...], gold_name, gold_args:{...}, gold_think}

Run:  python scripts/prepare_data.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import glob
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Real task presents ~4 candidate tools per query (dev is exactly 4). The parent
# dataset lists all 27; subsample to match the dev/test distribution AND keep
# prompts short enough that every row fits the training context.
_RNG = random.Random(42)
N_CANDIDATES = 4


def sample_candidates(all_names, gold_name, k=N_CANDIDATES):
    pool = [n for n in dict.fromkeys(all_names) if n != gold_name]
    _RNG.shuffle(pool)
    if gold_name and gold_name != "none":
        chosen = [gold_name] + pool[:k - 1]
    else:
        chosen = pool[:k]
    _RNG.shuffle(chosen)
    return chosen

import os
RAW_ST = Path(os.environ.get("RAW_ST", "data/raw/aisa-arabicfc-sharedtask/data"))
RAW_PARENT = Path("data/raw/aisa-ar-functioncall-parent/data")
OUT = Path(os.environ.get("OUT_DIR", "data/processed"))
OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────
def to_pylist(v):
    """Coerce a messages/tools cell (ndarray of dicts, list, or repr-string) to a list."""
    if v is None:
        return []
    if isinstance(v, str):
        return ast.literal_eval(v) if v.strip() else []
    if hasattr(v, "tolist"):
        v = v.tolist()
    return list(v)


def clean_args(raw) -> dict:
    """Drop None/'' values; keep order; values left exactly as gold (quirks preserved)."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = ast.literal_eval(raw) if raw.strip() else {}
    return {str(k): v for k, v in raw.items() if v is not None and v != ""}


def first_tool_call(tcs):
    """Return (name, args) from a tool_calls cell, or (None, None)."""
    tcs = to_pylist(tcs)
    if not tcs:
        return None, None
    fn = (tcs[0] or {}).get("function") or {}
    return fn.get("name"), clean_args(fn.get("arguments"))


def clean_tool_schema(fn: dict) -> dict:
    """Strip None-valued property slots (shared-task pads every key with None)."""
    params = fn.get("parameters") or {}
    props = {k: v for k, v in (params.get("properties") or {}).items() if v is not None}
    req = params.get("required", [])
    if isinstance(req, str):
        try:
            req = ast.literal_eval(req)
        except (ValueError, SyntaxError):
            req = []
    return {
        "description": fn.get("description", ""),
        "parameters": {"type": params.get("type", "object"), "properties": props, "required": req},
    }


def extract_messages(msgs):
    """Return (context_developer_text, user_text, gold_think) from a messages list."""
    msgs = to_pylist(msgs)
    context, user, think = "", "", ""
    for m in msgs:
        role = m.get("role")
        if role == "developer" and not context:
            context = (m.get("content") or "").strip()
        elif role == "user":
            user = (m.get("content") or "").strip()
        elif role == "assistant":
            think = (m.get("think") or m.get("_think_for_train") or "") or ""
            think = think.strip()
    return context, user, think


# ── load + normalize each source ───────────────────────────────────────
def load_sharedtask_train():
    df = pd.read_parquet(RAW_ST / "train-00000-of-00001.parquet")
    recs, registry = [], {}
    for _, r in df.iterrows():
        context, user, think = extract_messages(r["messages"])
        name, args = first_tool_call(
            next((m.get("tool_calls") for m in to_pylist(r["messages"]) if m.get("role") == "assistant"), None)
        )
        cands = []
        for t in to_pylist(r["tools_sampled"]):
            fn = t.get("function") or {}
            nm = fn.get("name")
            if nm:
                cands.append(nm)
                registry.setdefault(nm, clean_tool_schema(fn))
        rf = bool(r["requires_function"])
        recs.append({
            "source": "sharedtask", "dialect": str(r["dialect"]).lower(),
            "domain": None, "requires_function": rf,
            "context": context, "user": user, "candidate_tools": cands,
            "gold_name": (name or "none") if rf else "none",
            "gold_args": args if rf else {}, "gold_think": think,
        })
    return recs, registry


def load_parent_train():
    recs, registry = [], {}
    for f in sorted(glob.glob(str(RAW_PARENT / "train-*.parquet"))):
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            context, user, think = extract_messages(r["messages"])
            name, args = first_tool_call(
                next((m.get("tool_calls") for m in to_pylist(r["messages"]) if m.get("role") == "assistant"), None)
            )
            cands = []
            for t in to_pylist(r["tools"]):
                fn = t.get("function") or {}
                nm = fn.get("name")
                if nm:
                    cands.append(nm)
                    # parent schemas are clean → preferred source of truth
                    registry[nm] = clean_tool_schema(fn)
            rf = bool(r["requires_function"])
            gold_name = (name or "none") if rf else "none"
            recs.append({
                "source": "parent", "dialect": str(r["dialect"]).lower(),
                "domain": r.get("domain"), "requires_function": rf,
                "context": context, "user": user,
                "candidate_tools": sample_candidates(cands, gold_name),
                "gold_name": gold_name,
                "gold_args": args if rf else {}, "gold_think": think,
            })
    return recs, registry


def load_dev_gold():
    """Official dev (545) → gold records + the set of user queries to exclude from train."""
    df = pd.read_parquet(RAW_ST / "dev-00000-of-00001.parquet")
    gold, queries = [], set()
    for i, r in df.iterrows():
        context, user, _ = extract_messages(r["messages"])
        name, args = first_tool_call(
            next((m.get("tool_calls") for m in to_pylist(r["messages"]) if m.get("role") == "assistant"), None)
        )
        rf = bool(r["requires_function"])
        gold.append({
            "id": int(i), "dialect": str(r["dialect"]).lower(), "requires_function": rf,
            "tool_called": (name or "none") if rf else "none",
            "arguments": args if rf else {}, "context": context, "user": user,
            "candidate_tools": [ (t.get("function") or {}).get("name")
                                 for t in to_pylist(r["tools_sampled"]) if (t.get("function") or {}).get("name") ],
        })
        queries.add(user)
    return gold, queries


# ── main ───────────────────────────────────────────────────────────────
def main():
    print("[..] loading sources")
    st_recs, st_reg = load_sharedtask_train()
    pa_recs, pa_reg = load_parent_train()
    dev_gold, dev_queries = load_dev_gold()
    print(f"     sharedtask={len(st_recs):,}  parent={len(pa_recs):,}  dev={len(dev_gold):,}")

    # registry: parent (clean) wins; fill gaps from shared-task
    registry = dict(st_reg)
    registry.update(pa_reg)
    print(f"     tool registry: {len(registry)} tools")

    # merge, exclude dev leakage, dedup
    all_recs = st_recs + pa_recs
    seen, kept, n_leak = set(), [], 0
    for rec in all_recs:
        if rec["user"] in dev_queries:
            n_leak += 1
            continue
        key = (rec["user"], rec["gold_name"], json.dumps(rec["gold_args"], sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)
    print(f"     merged={len(all_recs):,}  dev-leak-dropped={n_leak:,}  after-dedup={len(kept):,}")

    # deterministic held-out internal val (~800, stratified-ish by hashing user)
    val, train = [], []
    for rec in kept:
        bucket = int(hashlib.blake2b(rec["user"].encode("utf-8"), digest_size=8).hexdigest(), 16) % 1000
        (val if bucket < 17 else train).append(rec)  # ~1.7%

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(OUT / "train.jsonl", train)
    dump(OUT / "internal_val.jsonl", val)
    dump(OUT / "dev.jsonl", dev_gold)
    with open(OUT / "tools_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    # stats
    def dialect_counts(rows):
        return dict(Counter(r["dialect"] for r in rows))
    stats = {
        "train": len(train), "internal_val": len(val), "dev": len(dev_gold),
        "dev_leak_dropped": n_leak, "n_tools": len(registry),
        "train_dialects": dialect_counts(train),
        "train_requires_function": dict(Counter(r["requires_function"] for r in train)),
        "train_has_think": sum(1 for r in train if r["gold_think"]),
        "tools": sorted(registry.keys()),
    }
    with open(OUT / "prepare_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(f"train={len(train):,}  val={len(val):,}  dev={len(dev_gold)}")
    print("train dialects:", stats["train_dialects"])
    print("train requires_function:", stats["train_requires_function"])
    print(f"train rows with gold <think>: {stats['train_has_think']:,}")
    print(f"tools: {stats['n_tools']}")


if __name__ == "__main__":
    main()
