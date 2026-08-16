"""
Inference for a QLoRA AISA-ArabicFC model: generate predictions on a gold split,
parse to submission JSON, score with the official eval.

    python scripts/run_infer.py --adapter runs/qwen7b-v1/final --gold data/processed/dev.jsonl \
        --out results/dev_qwen7b_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, "src")
from aisa.prompt import build_messages, parse_output  # noqa: E402

REG = json.load(open("data/processed/tools_registry.json", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base model)")
    ap.add_argument("--gold", default="data/processed/dev.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if tok.chat_template is None:  # match train_qlora fallback (AceGPT-v2 / Llama-3)
        tok.chat_template = (
            "{{- bos_token }}{%- for message in messages %}"
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
            "+ message['content'] | trim + '<|eot_id|>' }}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
    if args.limit:
        gold = gold[:args.limit]

    preds = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(gold), args.batch):
            chunk = gold[s:s + args.batch]
            prompts = [
                tok.apply_chat_template(build_messages(r, REG), tokenize=False, add_generation_prompt=True)
                for r in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for r, row_ids, inp in zip(chunk, gen, enc["input_ids"]):
                raw = tok.decode(row_ids[inp.shape[0]:], skip_special_tokens=True)
                p = parse_output(raw)
                preds.append({"id": r["id"], "tool_called": p["tool_called"],
                              "arguments": p["arguments"], "think": p["think"]})
            done = min(s + args.batch, len(gold))
            rate = done / (time.time() - t0)
            print(f"  [{done}/{len(gold)}] {rate:.1f} rows/s", end="\r")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n[ok] wrote {len(preds)} preds to {args.out}")

    # score — only if the input carries gold labels (blind test input does NOT)
    if gold and all("tool_called" in g for g in gold):
        sys.path.insert(0, "baselines/leaderboard-code")
        from eval_lib import evaluate  # type: ignore
        s = evaluate(preds, gold)
        print(f"FnAcc {s['fnacc']:.4f} | ArgEM {s['argem']:.4f} | Think {s['thinkrate']:.4f} "
              f"| OverallA {s['overall_a']:.4f} | OverallB {s['overall_b']:.4f}")
        Path(args.out).with_suffix(".scores.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print("[info] input has no gold labels (blind test) — skipping scoring")


if __name__ == "__main__":
    main()
