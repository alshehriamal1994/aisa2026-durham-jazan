"""Dialectal augmentation PILOT — teacher-paraphrase for blind robustness.

Uses our own ALLaM-7B (open, license-clean, disclosable) to rewrite existing
train queries into under-served dialects (Egyptian/Levantine/Maghrebi), KEEPING
the gold tool+args unchanged. Output rows can be appended to train_st_aug for a
more dialect-robust ensemble. PILOT: small N + prints samples to judge quality
and speed before committing a full run.
"""
import sys, json, argparse, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--targets", default="egyptian,levantine,maghrebi")
ap.add_argument("--out", default="data/processed_v13/aug_pilot.jsonl")
ap.add_argument("--src-dialects", default="msa")  # paraphrase FROM these
A = ap.parse_args()

MODEL = "ALLaM-AI/ALLaM-7B-Instruct-preview"
DIA_AR = {"egyptian": "المصرية", "levantine": "الشامية (اللهجة السورية/اللبنانية)",
          "maghrebi": "المغاربية", "gulf": "الخليجية"}
targets = A.targets.split(",")
srcs = set(A.src_dialects.split(","))

rows = [json.loads(l) for l in open("data/processed_v13/train_st_aug.jsonl") if l.strip()]
pos = [r for r in rows if (str(r.get("requires_function")).lower() == "true"
       or r.get("gold_name") not in (None, "none", "")) and r.get("dialect") in srcs]
pool = pos[:A.n]
print(f"loading {MODEL} (4-bit)...", file=sys.stderr)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model.eval()

def paraphrase(q, dia):
    sys_p = (f"أعد صياغة طلب المستخدم التالي باللهجة العربية {DIA_AR[dia]} مع الحفاظ التام على نفس المعنى "
             f"وكل الأسماء والأرقام والكيانات (المدن، العملات، المنتجات) كما هي دون تغيير أو ترجمة. "
             f"اكتب الصياغة الجديدة فقط دون أي شرح.")
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": q}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask")
    attn = attn.to(model.device) if attn is not None else None
    with torch.no_grad():
        out = model.generate(ids, attention_mask=attn, max_new_tokens=128, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

import time
t0 = time.time(); n = 0
with open(A.out, "w") as f:
    for i, r in enumerate(pool):
        dia = targets[i % len(targets)]
        newq = paraphrase(r["user"], dia)
        if not newq or len(newq) < 4:
            continue
        nr = dict(r); nr["user"] = newq; nr["dialect"] = dia; nr["source"] = "aug_dialect"
        f.write(json.dumps(nr, ensure_ascii=False) + "\n"); n += 1
        if i < 6:
            print(f"\n[{dia}] ORIG: {r['user']}\n      NEW : {newq}\n      gold: {r.get('gold_name')} {r.get('gold_args')}", file=sys.stderr)
rate = n / (time.time() - t0)
print(f"\nPILOT: wrote {n} augmented rows -> {A.out}  ({rate:.2f} rows/s, full ~10k would take ~{10000/rate/3600:.1f}h)", file=sys.stderr)
