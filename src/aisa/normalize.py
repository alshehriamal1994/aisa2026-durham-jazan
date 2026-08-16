"""
Argument-value normalization for AISA-ArabicFC.

ArgEM is a *strict* exact match after the official `_norm_args` (which only
str()-stringifies and strips). The gold labels apply several recurring
canonicalizations inconsistently; learning + reproducing the *dominant* ones
recovers points the baseline drops. These helpers are used both to (optionally)
clean training targets and to post-process model output before submission.

Keep this conservative: only transform when the dominant gold pattern is clear.
Over-normalizing can DESTROY matches when the gold itself kept the raw form.
Every transform here is measured against dev before being switched on.
"""
from __future__ import annotations

import re

# Arabic-Indic and Eastern-Arabic-Indic digits -> ASCII
_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"          # U+0660..0669
_EXT_DIGITS = "۰۱۲۳۴۵۶۷۸۹"        # U+06F0..06F9 (Persian/Urdu, appear occasionally)
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_AR_DIGITS)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_EXT_DIGITS)})

# Arabic tatweel + common diacritics (harakat) — gold sometimes strips these.
_TATWEEL = "ـ"
_HARAKAT = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")


def ar_digits_to_ascii(s: str) -> str:
    """٥٤٣ -> 543. Pure conversion, no other change."""
    return s.translate(_DIGIT_MAP)


def strip_tatweel_harakat(s: str) -> str:
    return _HARAKAT.sub("", s.replace(_TATWEEL, ""))


def normalize_number(val) -> str:
    """
    The gold stores numeric args as Python floats/ints, which str() renders as
    '3.0', '10.0', '543210'. The eval str()-compares, so a model emitting '3'
    when gold is 3.0 -> '3.0' MISMATCHES. Mirror the gold's float rendering:
    integers that came through as floats keep the '.0'.
    Returns the value unchanged (as str) when it isn't numeric.
    """
    s = str(val).strip()
    s_ascii = ar_digits_to_ascii(s)
    # Plain integer written by the model where gold likely used a float.
    if re.fullmatch(r"-?\d+", s_ascii):
        return s_ascii  # caller decides int-vs-float; see float_like()
    if re.fullmatch(r"-?\d+\.\d+", s_ascii):
        return s_ascii
    return s


def float_like(val) -> str:
    """Render a numeric value the way the gold's str(float) would: 3 -> '3.0'."""
    s = ar_digits_to_ascii(str(val).strip())
    try:
        f = float(s)
    except ValueError:
        return s
    # Gold uses str(float); ints-as-floats become 'N.0'.
    return repr(f) if f != int(f) else f"{int(f)}.0"


def normalize_value(val) -> str:
    """
    Default conservative text normalization for string args:
      - Arabic-Indic digits -> ASCII (gold does this for id/visa/iqama numbers)
      - strip surrounding whitespace and a single trailing sentence period
    Diacritic stripping is left OFF by default (measured separately) because the
    gold keeps diacritics in some text fields.
    """
    s = str(val).strip()
    s = ar_digits_to_ascii(s)
    s = s.rstrip("。.۔").strip()  # trailing latin/arabic full stop
    return s


# Quick self-check when run directly.
if __name__ == "__main__":
    assert ar_digits_to_ascii("٥٤٣٢١٠") == "543210"
    assert ar_digits_to_ascii("١٥ أكتوبر") == "15 أكتوبر"
    assert float_like(3) == "3.0"
    assert float_like("10") == "10.0"
    assert float_like("3.5") == "3.5"
    assert normalize_value("أنا أحب القهوة.") == "أنا أحب القهوة"
    print("normalize.py self-check OK")
