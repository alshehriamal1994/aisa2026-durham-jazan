"""
Reproduce the Track A baseline (AISA-AR-FunctionCall-FT) on the dev split.

Reads the pre-formatted `text` field, cuts at the model turn, greedy-generates,
parses the FunctionGemma DSL output, writes predictions JSONL, and scores
locally with the official eval_lib.

Usage:
    python scripts/run_baseline_track_a.py [--limit N] [--max-new-tokens 200]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path("baselines/aisa-ar-functioncall-ft")
DATA_PATH = Path("data/raw/aisa-arabicfc-sharedtask/data/dev-00000-of-00001.parquet")
OUT_PATH = Path("results/dev_track_a_baseline.jsonl")


# Reused from the Track B README (works for Track A too — think section will just be empty)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
CALL_RE = re.compile(r"<start_function_call>\s*call:(\w+)\{(.*?)\}\s*<end_function_call>", re.DOTALL)
ARG_RE = re.compile(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]+))")


def parse_output(text: str) -> dict:
    """Parse one raw model generation into the submission schema."""
    out = {"requires_function": False, "tool_called": "none", "arguments": {}, "think": ""}
    if (m := THINK_RE.search(text)):
        out["think"] = m.group(1).strip()
    if (m := CALL_RE.search(text)):
        out["requires_function"] = True
        out["tool_called"] = m.group(1)
        for key, str_val, num_val in ARG_RE.findall(m.group(2)):
            val: object = str_val if str_val else num_val
            if val == "":
                continue
            # Drop literal None/null/empty placeholders the model emits for optional fields.
            if isinstance(val, str) and val.strip().lower() in {"none", "null", ""}:
                continue
            if str_val == "":  # numeric coercion only when it wasn't an <escape> string
                s = str(val).strip()
                try:
                    val = float(s) if "." in s else int(s)
                except ValueError:
                    pass
            out["arguments"][key] = val
    return out


def cut_prompt(text: str) -> str:
    """Take the `text` field and cut it at the model turn — what the model has to complete."""
    marker = "<start_of_turn>model\n"
    if marker in text:
        return text.split(marker, 1)[0] + marker
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N dev rows")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    print(f"[..] Loading dev from {DATA_PATH}")
    dev = pd.read_parquet(DATA_PATH)
    if args.limit:
        dev = dev.head(args.limit)
    print(f"[ok] {len(dev):,} rows")

    print(f"[..] Loading tokenizer + model from {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    print(f"[ok] device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    preds: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for i, row in dev.iterrows():
            prompt = cut_prompt(row["text"])
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            raw = tok.decode(new_tokens, skip_special_tokens=False)
            parsed = parse_output(raw)

            preds.append({
                "id": int(i),
                "tool_called": parsed["tool_called"],
                "arguments": parsed["arguments"],
                "think": parsed["think"],
            })

            if (i + 1) % 25 == 0 or i + 1 == len(dev):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(dev) - i - 1) / rate
                print(f"  [{i+1:>3}/{len(dev)}]  {rate:.2f} rows/s  ETA {eta:.0f}s")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[ok] wrote {len(preds)} predictions to {OUT_PATH}")

    # ── Score locally with the official eval ──
    sys.path.insert(0, str(Path("baselines/leaderboard-code")))
    from eval_lib import evaluate  # type: ignore

    print("\n[..] Scoring locally with official eval_lib")
    gold = []
    for i, row in dev.iterrows():
        assistant = next((m for m in row["messages"] if m.get("role") == "assistant"), None)
        tcs = assistant.get("tool_calls") if assistant else None
        if tcs is not None and len(tcs) > 0:
            fn = tcs[0].get("function") or {}
            gold_args = {k: v for k, v in (fn.get("arguments") or {}).items() if v is not None and v != ""}
            gold.append({
                "id": int(i),
                "tool_called": fn.get("name", row["tool_called"]),
                "arguments": gold_args,
                "dialect": row["dialect"],
                "requires_function": True,
            })
        else:
            gold.append({
                "id": int(i),
                "tool_called": "none",
                "arguments": {},
                "dialect": row["dialect"],
                "requires_function": False,
            })

    scores = evaluate(preds, gold)
    print("\n=== Scores ===")
    print(f"FnAcc:        {scores['fnacc']:.4f}")
    print(f"ArgEM:        {scores['argem']:.4f}")
    print(f"ThinkRate:    {scores['thinkrate']:.4f}")
    print(f"Overall (A):  {scores['overall_a']:.4f}")
    print(f"Overall (B):  {scores['overall_b']:.4f}")
    print(f"Dialect gap (FnAcc): {scores['gap_fnacc']:.4f}")
    print(f"Dialect gap (ArgEM): {scores['gap_argem']:.4f}")
    print(f"\nPer-dialect:")
    for d, s in scores["dialect_breakdown"].items():
        print(f"  {d:<10} n={s['n']:>3}  pos={s['n_positive']:>3}  FnAcc={s['fnacc']:.3f}  ArgEM={s['argem']:.3f}")

    # Save scores
    score_path = OUT_PATH.with_suffix(".scores.json")
    with score_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"\nScores saved to {score_path}")


if __name__ == "__main__":
    main()
