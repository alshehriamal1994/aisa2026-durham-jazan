"""
Phase 1 — QLoRA SFT for AISA-ArabicFC (Tracks A+B).

Fine-tunes a strong instruct base (default Qwen2.5-7B-Instruct, 4-bit) to emit
<think>..</think> + a JSON tool call. Uses explicit label masking (train on the
assistant turn only) for version-stable, bulletproof completion-only loss.

Smoke test (a few steps, tiny subset):
    python scripts/train_qlora.py --max-train 256 --epochs 1 --out runs/smoke

Full run:
    python scripts/train_qlora.py --model Qwen/Qwen2.5-7B-Instruct --epochs 2 --out runs/qwen7b-v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, "src")
from aisa.prompt import build_messages, build_target  # noqa: E402

REG = json.load(open("data/processed/tools_registry.json", encoding="utf-8"))


def load_records(path, limit=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


# Minimal Llama-3 chat template — fallback for bases that ship without one
# (AceGPT-v2-8B-Chat). Standard system/user/assistant roles.
LLAMA3_CHAT_TEMPLATE = (
    "{{- bos_token }}{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"
)


def make_dataset(rows, tok, max_len):
    # Materialize explicitly via from_list — from_generator's opaque caching of
    # no-arg generators silently truncated a prior run to 10k examples.
    examples, skipped = [], 0
    for r in rows:
        msgs = build_messages(r, REG)
        target = build_target(r)
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_str = tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": target}],
            tokenize=False, add_generation_prompt=False,
        )
        if not full_str.startswith(prompt_str):
            skipped += 1
            continue  # template mismatch guard (should not happen)
        p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
        f_ids = tok(full_str, add_special_tokens=False)["input_ids"]
        if len(f_ids) > max_len:
            skipped += 1
            continue  # skip the rare over-long example rather than truncate the target
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        examples.append({"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels})
    print(f"     tokenized {len(examples):,} examples (skipped {skipped})")
    return Dataset.from_list(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--out", default="runs/qwen7b-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=None, help="cap training rows (smoke test)")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--liger", action="store_true", help="use Liger fused kernels (saves VRAM on the LM head)")
    args = ap.parse_args()

    print(f"[..] tokenizer + 4-bit model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:  # e.g. AceGPT-v2 (Llama-3 base) ships no template
        tok.chat_template = LLAMA3_CHAT_TEMPLATE

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("[..] building datasets")
    train_rows = load_records(args.train_file, args.max_train)
    val_rows = load_records("data/processed/internal_val.jsonl", 400)
    train_ds = make_dataset(train_rows, tok, args.max_len)
    val_ds = make_dataset(val_rows, tok, args.max_len)
    print(f"     train={len(train_ds):,}  val={len(val_ds):,}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=4,
        use_liger_kernel=args.liger,
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)

    print("[..] training")
    from transformers.trainer_utils import get_last_checkpoint
    last_ckpt = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    if last_ckpt:
        print(f"[..] resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    final = Path(args.out) / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    print(f"[ok] saved adapter to {final}")


if __name__ == "__main__":
    main()
