"""
Negative augmentation — the path to FnAcc≈1.0.

v1 got 0/500 positives wrong but 37/45 negatives wrong (it almost never says
"none"), because training had only ~60 negative examples. Dev negatives span 11
categories. We add two kinds of negatives:

1. CANDIDATE-SWAP (free, real, multi-dialect): take real positive queries and
   present 4 candidate tools that EXCLUDE the gold tool → answer becomes "none".
   Teaches "only call a candidate that actually matches" — covers off_domain /
   ambiguous / out-of-scope dev cases with genuine Arabic across all dialects.

2. NO-ACTION templates: chitchat / thanks / opinion / general-knowledge /
   past-tense / feelings / meta across 5 dialects — covers the purely
   conversational dev negatives that have no actionable intent at all.

Output: data/processed/train_aug.jsonl  (positives + originals + new negatives)
Run:    python scripts/augment_negatives.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path
import os

OUT = Path(os.environ.get("OUT_DIR", "data/processed"))
RNG = random.Random(7)

# ── no-action negative templates per category, with dialect variants ──────
# Each entry is a ready Arabic utterance needing NO tool. Mixed dialects.
NO_ACTION = {
    "chitchat": [
        "السلام عليكم ورحمة الله", "أهلين كيفك اليوم؟", "صباح الخير يا صديقي",
        "مساء الخير، شو الأخبار؟", "هلا والله، نورت", "شكراً جزيلاً على مساعدتك",
        "الله يعطيك العافية", "تسلم إيدك، مشكور", "يعطيك ألف عافية يا غالي",
        "مع السلامة، نشوفك بخير", "تمام التمام، يسلمو", "كتر خيرك يا باشا",
    ],
    "opinion": [
        "شو رأيك بأفضل وجهة سياحية؟", "أيش أحلى مدينة في العالم العربي؟",
        "برأيك إيه أحسن أكلة مصرية؟", "وش تنصحني أقرأ من الكتب؟",
        "شو بتفضّل، القهوة ولا الشاي؟", "أيهما أفضل في رأيك الصيف أم الشتاء؟",
    ],
    "general_knowledge": [
        "مين مؤسس شركة أبل؟", "شنو عاصمة اليابان؟", "ما هو أطول نهر في العالم؟",
        "كم عدد سكان مصر؟", "إيش هو أكبر محيط في العالم؟", "متى انتهت الحرب العالمية الثانية؟",
        "مين كتب رواية البؤساء؟", "ما هي سرعة الضوء؟",
    ],
    "off_domain": [
        "حل لي هذه المعادلة: ٢س + ٥ = ١١", "اكتب لي كود بايثون يطبع الأعداد الأولية",
        "ترجم لي معنى الحياة فلسفياً", "صمم لي شعار لمشروعي",
        "اكتب لي قصيدة عن الوطن", "علّمني كيف أعزف على العود",
    ],
    "past_tense_completed": [
        "قارنت أسعار الجوالات وقررت آخذ سامسونج", "حجزت الفندق امبارح والحمد لله",
        "حوّلت المبلغ وخلصت المعاملة", "رحت للدكتور وكل شي تمام الحين",
        "طلبت الأكل ووصل بسرعة", "خلصت تجديد الإقامة الأسبوع اللي فات",
    ],
    "feelings_experiences": [
        "مبسوط إني حجزت الفندق من بدري", "فرحان لأن مكافأة نهاية الخدمة طلعت كويسة",
        "زعلان شوي من زحمة المرور اليوم", "متحمس لرحلة العمرة الجاية",
        "مرتاح بعد ما خلصت كل المعاملات", "قلقان شوي بخصوص نتيجة التحاليل",
    ],
    "meta_questions": [
        "شو الخدمات اللي تقدر تساعدني فيها؟", "إيش الأشياء اللي بتعرف تعملها؟",
        "وش قدراتك بالضبط؟", "كيف ممكن تساعدني؟", "مين أنت وشو وظيفتك؟",
        "إنت بوت ولا إنسان؟",
    ],
    "about_domain_no_action": [
        "ما الفرق بين الطقس والمناخ؟", "شو يعني سعر الصرف بالظبط؟",
        "إيه الفرق بين العمرة والحج؟", "وش معنى الزكاة؟",
        "ليش الذهب غالي؟", "شنو الفرق بين الإقامة والتأشيرة؟",
    ],
    "ambiguous_underspecified": [
        "محتاج مساعدة بشغلة", "في إمكانية تساعدني بموضوع؟", "عندي طلب صغير",
        "ممكن سؤال؟", "أبغى أستفسر عن شي", "لو سمحت محتاج خدمة",
    ],
    "comparative_evaluation": [
        "أيهما أفضل للسفر الطيران أم القطار؟", "إيه أحسن، الدفع كاش ولا بالبطاقة؟",
        "وش الأفضل أسكن فندق ولا شقة؟",
    ],
    "hypothetical_conditional": [
        "لو عندي مليون ريال شو تنصحني أسوي فيها؟", "افترض إني بسافر بكرة، إيش ألبس؟",
        "لو كنت مكاني شو كنت تختار؟",
    ],
}

NO_THINK = [
    "لا تتطلب رسالة المستخدم استدعاء أي أداة من الأدوات المتاحة.",
    "هذه رسالة عامة لا تحتاج إلى استدعاء أداة.",
    "لا توجد أداة مناسبة بين الأدوات المتاحة لتلبية هذا الطلب، لذا لا حاجة للاستدعاء.",
    "الطلب لا يستلزم تنفيذ أي وظيفة.",
]


def make_neg(user, tools, dialect):
    return {
        "source": "augment", "dialect": dialect, "domain": None,
        "requires_function": False, "context": "", "user": user,
        "candidate_tools": tools, "gold_name": "none", "gold_args": {},
        "gold_think": RNG.choice(NO_THINK),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="train.jsonl")
    ap.add_argument("--output", default="train_aug.jsonl")
    ap.add_argument("--source", default=None, help="keep only this source (e.g. sharedtask)")
    ap.add_argument("--n-swap", type=int, default=1800)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(OUT / a.input, encoding="utf-8")]
    if a.source:
        rows = [r for r in rows if r.get("source") == a.source]
    all_tools = sorted({t for r in rows for t in r["candidate_tools"]})
    # registry of all tools (for distractors) — from full registry, not just filtered rows
    all_tools = sorted(json.load(open(OUT / "tools_registry.json", encoding="utf-8")).keys())
    positives = [r for r in rows if r["requires_function"]]
    existing_neg = [r for r in rows if not r["requires_function"]]
    dialects = ["msa", "gulf", "egyptian", "levantine", "maghrebi"]

    new_negs = []

    # 1) candidate-swap negatives from real positive queries
    n_swap = a.n_swap
    for r in RNG.sample(positives, min(n_swap, len(positives))):
        pool = [t for t in all_tools if t != r["gold_name"]]
        RNG.shuffle(pool)
        tools = pool[:4]
        new_negs.append(make_neg(r["user"], tools, r["dialect"]))

    # 2) no-action template negatives (each query repeated with a few random tool sets)
    for cat, utts in NO_ACTION.items():
        for u in utts:
            for _ in range(6):  # vary candidate sets / dialect tag
                RNG.shuffle(all_tools)
                tools = all_tools[:4]
                new_negs.append(make_neg(u, tools, RNG.choice(dialects)))

    # 3) upsample the ~60 genuine dataset negatives
    upsampled = existing_neg * 4

    aug = positives + existing_neg + upsampled + new_negs
    RNG.shuffle(aug)
    with open(OUT / a.output, "w", encoding="utf-8") as f:
        for r in aug:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_neg = sum(1 for r in aug if not r["requires_function"])
    print(f"positives={len(positives):,}  orig_neg={len(existing_neg)}  "
          f"swap_neg={n_swap}  template_neg={sum(len(v) for v in NO_ACTION.values())*6}  "
          f"upsampled={len(upsampled)}")
    print(f"TOTAL train_aug={len(aug):,}  negatives={n_neg:,} ({n_neg/len(aug):.1%})")


if __name__ == "__main__":
    main()
