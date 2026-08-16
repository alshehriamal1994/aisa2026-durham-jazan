"""
Canonicalization layer — SAFE post-processing of predicted argument values.

Data shows the gold is highly consistent for 44/64 (tool,arg) pairs (>90% one
form) but irreducibly split for ~10 (currency, dates, type, termination_type).
So we ONLY snap values for high-purity pairs, using a per-(tool,arg) dominant
form learned from training + small curated AR<->LAT entity dictionaries. We
NEVER touch low-purity pairs (would break as many as it fixes).

Build the form table from training once; apply at inference.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, Counter

from .normalize import float_like, ar_digits_to_ascii

PURITY_MIN = 0.90  # only snap (tool,arg) pairs at least this consistent in script


def _script(s):
    s = str(s)
    if re.search(r"[؀-ۿ]", s):
        return "AR"
    if re.search(r"[A-Za-z]", s):
        return "LAT"
    return "NUM"


# curated AR<->LAT canonical equivalents (mined from gold value lists + domain).
# direction-agnostic groups; we pick the member matching the target script.
_GROUPS = [
    # languages: iso / english / arabic
    {"LAT": ["en", "English"], "AR": ["الإنجليزية", "انجليزية", "الانجليزية"]},
    {"LAT": ["fr", "French"], "AR": ["الفرنسية", "فرنسية"]},
    {"LAT": ["es", "Spanish"], "AR": ["الإسبانية", "الاسبانية", "إسبانية"]},
    {"LAT": ["de", "German"], "AR": ["الألمانية", "ألمانية"]},
    {"LAT": ["it", "Italian"], "AR": ["الإيطالية", "إيطالية"]},
    {"LAT": ["ar", "Arabic"], "AR": ["العربية", "عربية"]},
    # zakat types
    {"LAT": ["money"], "AR": ["مال", "المال"]},
    {"LAT": ["gold"], "AR": ["ذهب", "الذهب"]},
    {"LAT": ["silver"], "AR": ["فضة", "الفضة"]},
    # countries (common)
    {"LAT": ["Saudi Arabia"], "AR": ["السعودية"]},
    {"LAT": ["Egypt"], "AR": ["مصر"]},
    {"LAT": ["UAE"], "AR": ["الإمارات"]},
    {"LAT": ["Kuwait"], "AR": ["الكويت"]},
    {"LAT": ["Jordan"], "AR": ["الأردن"]},
    {"LAT": ["Qatar"], "AR": ["قطر"]},
    {"LAT": ["Bahrain"], "AR": ["البحرين"]},
    {"LAT": ["Lebanon"], "AR": ["لبنان"]},
]


def _convert(value, target_script):
    """Map value to its equivalent in target_script if known; else None."""
    v = str(value).strip()
    for g in _GROUPS:
        for forms in g.values():
            if v in forms:
                cand = g.get(target_script)
                return cand[0] if cand else None
    return None


def build_form_table(train_path="data/processed/train.jsonl"):
    """Learn dominant script per (tool,arg) from training gold."""
    counts = defaultdict(Counter)
    for line in open(train_path, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("requires_function"):
            continue
        t = r["gold_name"]
        for k, v in r["gold_args"].items():
            counts[(t, k)][_script(v)] += 1
    table = {}
    for key, c in counts.items():
        n = sum(c.values())
        form, top = c.most_common(1)[0]
        if n >= 20 and top / n >= PURITY_MIN:
            table[key] = form  # the dominant script we trust
    return table


class Canonicalizer:
    def __init__(self, table):
        self.table = table

    def fix(self, tool, args):
        out = {}
        for k, v in args.items():
            dom = self.table.get((tool, k))
            nv = v
            if dom == "NUM":
                nv = float_like(v) if _script(v) in ("NUM",) or re.search(r"\d", str(v)) else v
            elif dom in ("AR", "LAT") and _script(v) != dom:
                conv = _convert(v, dom)
                if conv is not None:
                    nv = conv
            out[k] = nv
        return out
