"""
AISA-ArabicFC — argument normalization & fair matching for ArgEM.
=================================================================
Applied IDENTICALLY to predictions and gold, deterministic, and published so
participants know the rules. It only treats different *writings of the same
answer* as equal — never two different answers.

Layers
  0. Numbers      Arabic-Indic/Persian digits -> ASCII; whole-number int==float
                  (quantity fields only; identifier fields stay strict).
  1. Orthography  strip Arabic diacritics + tatweel; unify alef أإآٱ->ا,
                  alef-maqsura ى->ي, ta-marbuta ة->ه, hamza seats.
  2. Number words Arabic cardinals (واحد..ألف) -> digits.
  3. List fields  split on ، , ; & / و and ⏎ -> compare as an unordered SET.
  4. Aliases      closed classes: countries, cities, currencies, languages,
                  zakat-type, quran search-type  (bilingual / ISO).

`termination_type` is intentionally NOT semantically aliased (168 uncontrolled
variants) — see SEMANTIC_REVIEW_FIELDS. Only safe orthographic normalization is
applied there until the organizers approve a canonical map.
"""
from __future__ import annotations
import re
import unicodedata

# ────────────────────────── Layer 0: digits / numbers ──────────────────────
_DIGITS = {**{ord("٠") + i: str(i) for i in range(10)},
           **{ord("۰") + i: str(i) for i in range(10)}}

def to_ascii_digits(s: str) -> str:
    return s.translate(_DIGITS)

# Identifier-like fields: keep strict (leading zeros & exact form matter).
ID_FIELDS = {
    "id_number", "iqama_number", "visa_number", "recipient_iban", "iban",
    "insurance_number", "passport_number", "phone", "phone_number",
    "national_id", "account_number", "reference_number",
}

def _as_number(v):
    try:
        return float(to_ascii_digits(str(v)).replace(",", "").strip())
    except (ValueError, TypeError):
        return None

# ────────────────────────── Layer 1: Arabic orthography ─────────────────────
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

def arabic_ortho(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _DIAC.sub("", s)               # harakat, tanwin, shadda, sukun, dagger-alef, tatweel
    s = re.sub("[إأآٱ]", "ا", s)       # alef variants
    s = s.replace("ى", "ي").replace("ة", "ه")
    s = s.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
    return s

# ────────────────────────── Layer 2: number words ──────────────────────────
# keys already in orthographically-normalised form (no ة, bare alef, …)
_NUMWORDS = {
    "صفر": "0",
    "واحد": "1", "واحده": "1", "احد": "1", "احدي": "1",
    "اثنين": "2", "اثنان": "2", "اثنتين": "2", "اثنتان": "2",
    "ثلاث": "3", "ثلاثه": "3",
    "اربع": "4", "اربعه": "4",
    "خمس": "5", "خمسه": "5",
    "ست": "6", "سته": "6",
    "سبع": "7", "سبعه": "7",
    "ثمان": "8", "ثمانيه": "8", "ثماني": "8",
    "تسع": "9", "تسعه": "9",
    "عشر": "10", "عشره": "10",
    "عشرين": "20", "ثلاثين": "30", "اربعين": "40", "خمسين": "50",
    "ستين": "60", "سبعين": "70", "ثمانين": "80", "تسعين": "90",
    "مايه": "100", "مئه": "100", "ميه": "100", "الف": "1000",
}

def words_to_digits(s: str) -> str:
    return " ".join(_NUMWORDS.get(t, t) for t in s.split())

# ────────────────────────── Layer 4: closed-class aliases ───────────────────
# canonical -> list of surface forms (Arabic + English/ISO). Normalised on load.
_COUNTRIES = {
    "saudi": ["السعودية", "المملكة العربية السعودية", "المملكة", "Saudi Arabia", "KSA", "Saudi"],
    "uae": ["الإمارات", "الامارات", "الإمارات العربية المتحدة", "UAE", "United Arab Emirates", "Emirates"],
    "egypt": ["مصر", "Egypt"],
    "jordan": ["الأردن", "Jordan"],
    "kuwait": ["الكويت", "Kuwait"],
    "qatar": ["قطر", "Qatar"],
    "bahrain": ["البحرين", "Bahrain"],
    "oman": ["عمان", "سلطنة عمان", "Oman"],          # country sense (city Amman handled in _CITIES)
    "yemen": ["اليمن", "Yemen"],
    "syria": ["سوريا", "سورية", "الشام", "Syria"],
    "lebanon": ["لبنان", "Lebanon"],
    "iraq": ["العراق", "Iraq"],
    "palestine": ["فلسطين", "Palestine"],
    "tunisia": ["تونس", "Tunisia"],
    "algeria": ["الجزائر", "Algeria"],
    "morocco": ["المغرب", "Morocco"],
    "libya": ["ليبيا", "Libya"],
    "sudan": ["السودان", "Sudan"],
    "china": ["الصين", "China"],
    "usa": ["أمريكا", "امريكا", "الولايات المتحدة", "USA", "US", "United States", "America"],
    "uk": ["بريطانيا", "المملكة المتحدة", "UK", "United Kingdom", "Britain"],
    "germany": ["ألمانيا", "Germany"],
    "france": ["فرنسا", "France"],
    "turkey": ["تركيا", "Turkey", "Türkiye"],
    "spain": ["إسبانيا", "اسبانيا", "Spain"],
    "italy": ["إيطاليا", "ايطاليا", "Italy"],
    "canada": ["كندا", "Canada"],
    "india": ["الهند", "India"],
    "japan": ["اليابان", "Japan"],
    "switzerland": ["سويسرا", "Switzerland"],
    "russia": ["روسيا", "Russia"],
}
_CITIES = {
    "amman": ["عمّان", "عمان", "Amman"],             # city sense
    "dubai": ["دبي", "Dubai"],
    "beirut": ["بيروت", "Beirut"],
    "damascus": ["دمشق", "Damascus"],
    "cairo": ["القاهرة", "Cairo"],
    "ankara": ["أنقرة", "Ankara"],
    "jerusalem": ["القدس", "Jerusalem"],
    "irbid": ["إربد", "Irbid"],
    "homs": ["حمص", "Homs"],
    "tripoli": ["طرابلس", "Tripoli"],
    "riyadh": ["الرياض", "Riyadh"],
    "london": ["لندن", "London"],
    "ceuta": ["سبتة", "Ceuta"],
    "mecca": ["مكة", "مكة المكرمة", "Mecca", "Makkah"],
    "medina": ["المدينة", "المدينة المنورة", "Medina", "Madinah"],
    "jeddah": ["جدة", "Jeddah"],
}
# currency: ISO canonical. Only ISO + *qualified* Arabic forms — bare ambiguous
# words (ريال / دينار / درهم / ليرة / جنيه) are deliberately NOT aliased.
_CURRENCIES = {
    "sar": ["SAR", "ريال سعودي", "سعودي ريال", "ريال"],
    "aed": ["AED", "درهم إماراتي", "درهم"],
    "egp": ["EGP", "جنيه مصري", "الجنيه المصري", "جنيه"],
    "kwd": ["KWD", "دينار كويتي"],
    "bhd": ["BHD", "دينار بحريني"],
    "qar": ["QAR", "ريال قطري"],
    "omr": ["OMR", "ريال عماني"],
    "yer": ["YER", "ريال يمني"],
    "usd": ["USD", "دولار أمريكي", "دولار امريكي", "دولار"],
    "eur": ["EUR", "يورو", "أورو"],
    "gbp": ["GBP", "جنيه استرليني"],
    "syp": ["SYP", "ليرة سورية", "ليرة سوري", "الليرة السورية"],
    "lbp": ["LBP", "ليرة لبنانية"],
    "mad": ["MAD", "درهم مغربي"],
    "tnd": ["TND", "دينار تونسي"],
    "jod": ["JOD", "دينار أردني"],
    "dzd": ["DZD", "دينار جزائري"],
    "lyd": ["LYD", "دينار ليبي"],
    "cad": ["CAD", "دولار كندي"],
    "jpy": ["JPY"], "try": ["TRY", "ليرة تركية"], "ils": ["ILS"],
    "iqd": ["IQD"], "inr": ["INR"], "pkr": ["PKR"],
    "gram": ["g", "gram", "grams", "جرام", "غرام", "جم"],
    "kg": ["kg", "كيلو", "كجم", "كيلوجرام", "كيلوغرام"],
}
_LANGUAGES = {
    "en": ["en", "english", "الإنجليزية", "الانجليزية", "انجليزية", "انجليزي", "الانجليزي"],
    "ar": ["ar", "arabic", "العربية", "العربي", "عربي"],
    "fr": ["fr", "french", "الفرنسية", "الفرنسي", "فرنسية", "فرنسي"],
    "es": ["es", "spanish", "الإسبانية", "الاسبانية", "اسباني"],
    "de": ["de", "german", "الألمانية", "الالمانية", "الماني"],
    "it": ["it", "italian", "الإيطالية", "الايطالية"],
    "tr": ["tr", "turkish", "التركية"],
    "zh": ["zh", "chinese", "الصينية"],
    "ja": ["ja", "japanese", "اليابانية"],
    "ko": ["ko", "korean", "الكورية"],
    "fa": ["fa", "persian", "الفارسية"],
    "ur": ["ur", "urdu", "الأردية"],
    "hi": ["hi", "hindi", "الهندية"],
    "ru": ["ru", "russian", "الروسية"],
    "pt": ["pt", "portuguese", "البرتغالية"],
}
# zakat asset type — clean enough to alias
_ZTYPE = {
    "gold": ["gold", "ذهب", "الذهب"],
    "silver": ["silver", "فضة", "الفضة"],
    "cash": ["cash", "money", "maal", "cash_in_bank", "نقد", "النقد", "نقود", "النقود",
             "مال", "المال", "أموال", "الأموال", "أموال نقدية", "المال النقدي", "مدخرات", "المدخرات"],
    "trade": ["trade", "goods", "عروض تجارية", "تجارة", "التجارة", "بضائع"],
    "crops": ["crops", "زرع", "زروع", "الزروع", "محاصيل", "المحاصيل الزراعية"],
    "livestock": ["livestock", "أغنام", "ماشية", "الماشية", "أنعام"],
    "salary": ["salary", "income", "راتب", "دخل"],
    "shares": ["shares", "stocks", "أسهم"],
    "realestate": ["عقارات", "land", "أرض"],
    "fitr": ["زكاة الفطر", "فطر", "الفطر", "fitr"],
}
# quran search type — clean enough to alias
_SEARCHTYPE = {
    "verse": ["verse", "آية", "آيات", "اية", "ايات"],
    "surah": ["surah", "سورة", "سوره"],
    "tafseer": ["tafseer", "interpretation", "تفسير"],
    "meaning": ["meaning", "معنى", "تقصي"],
    "word": ["word", "كلمة", "كلمه"],
    "topic": ["topic", "موضوع"],
    "exact": ["exact", "بداية", "start", "starts_with", "beginning"],
    "any": ["any"],
}

def _build(*tables):
    out = {}
    for table in tables:
        for canon, forms in table.items():
            for f in forms:
                out[arabic_ortho(to_ascii_digits(f)).strip().casefold()] = canon
    return out

_ALIAS = {
    "country": _build(_COUNTRIES, _CITIES),
    "city": _build(_CITIES, _COUNTRIES),
    "currency": _build(_CURRENCIES),
    "language": _build(_LANGUAGES),
    "ztype": _build(_ZTYPE),
    "searchtype": _build(_SEARCHTYPE),
}
FIELD_CLASS = {
    "country": "country", "destination_country": "country", "departure_country": "country",
    "nationality": "country",
    "city": "city", "departure_city": "city", "destination_city": "city", "arrival_city": "city",
    "currency": "currency", "from_currency": "currency", "to_currency": "currency",
    "target_language": "language", "source_language": "language", "language": "language",
    "type": "ztype",
    "search_type": "searchtype",
}

# Fields left to orthographic normalisation only (uncontrolled vocabulary —
# pending an organizer-approved canonical map). NOT semantically aliased.
SEMANTIC_REVIEW_FIELDS = {"termination_type"}

LIST_FIELDS = {"items", "country", "destination_country"}
# quantity fields where 5000 == 5000.0 (everything castable, except ID_FIELDS)
_SEP = re.compile(r"\s*(،|,|;|&|/|\bو\b|\band\b|\n)\s*")
_WAW_ATTACHED = re.compile(r"\sو(?=ال\S)")   # "مصر والإمارات" -> split (scoped to list fields)

# ── Open / semi-closed free-text fields ─────────────────────────────────────
# product_name: strip generic device/carrier words + the definite article, map
# common brand transliterations to a canonical latin token, keep model numbers
# (so iPhone 13 != iPhone 14). Unifies آيفون / الآيفون / هاتف آيفون / iPhone /
# آيفون ١٣ etc. without merging distinct products.
PRODUCT_FIELDS = {"product_name"}
_CARRIER = {arabic_ortho(w) for w in
            ["هاتف", "هواتف", "موبايل", "موبايلات", "جوال", "جوالات",
             "تليفون", "تلفون", "تيليفون", "جهاز", "اجهزة", "محمول"]}
_PROD_BRAND = {arabic_ortho(k): v for k, v in {
    "ايفون": "iphone", "ابل": "apple", "ايباد": "ipad", "جالكسي": "galaxy",
    "سامسونج": "samsung", "سامسونغ": "samsung", "هواوي": "huawei", "شاومي": "xiaomi",
    "ريدمي": "redmi", "ديل": "dell", "لينوفو": "lenovo", "اسوس": "asus", "ايسر": "acer",
    "سوني": "sony", "نوكيا": "nokia", "شارب": "sharp", "بلايستيشن": "playstation",
    "اكسبوكس": "xbox", "نينتندو": "nintendo", "ابو": "oppo",
}.items()}

def _norm_product(s: str) -> str:
    out = []
    for t in s.split():
        stripped = t[2:] if t.startswith("ال") and len(t) > 4 else t
        if t in _CARRIER or stripped in _CARRIER:
            continue
        out.append(_PROD_BRAND.get(stripped, t))
    return " ".join(out).strip() or s

# specialty: drop honorific/qualifier prefixes (طبيب/دكتور/طب/أمراض…) + article,
# then map the medical-specialty core (Arabic + English) to a canonical token.
SPECIALTY_FIELDS = {"specialty"}
_SPEC_PREFIX = {arabic_ortho(w) for w in
                ["طبيب", "دكتور", "اخصائي", "اختصاصي", "استشاري", "طب", "امراض", "قسم", "عيادة"]}
_SPEC_ALIAS = {arabic_ortho(k): v for k, v in {
    "اطفال": "pediatrics", "قلب": "cardiology", "عيون": "ophthalmology",
    "اسنان": "dentistry", "جلدية": "dermatology", "جلد": "dermatology",
    "انف واذن وحنجرة": "ent", "انف واذن": "ent", "عظام": "orthopedics",
    "اعصاب": "neurology", "مخ واعصاب": "neurology", "باطنية": "internal", "باطنة": "internal",
    "نساء": "gynecology", "نسائية": "gynecology", "نساء وتوليد": "gynecology",
    "نساء وولادة": "gynecology", "نساء ولادة": "gynecology", "نفسي": "psychiatry",
    "نفسية": "psychiatry", "عام": "general", "اورام": "oncology",
    "جهاز هضمي": "gastroenterology", "جهاز تنفسي": "pulmonology", "غدد صماء": "endocrinology",
}.items()}
_SPEC_ALIAS.update({  # English forms
    "pediatrician": "pediatrics", "pediatrics": "pediatrics", "cardiology": "cardiology",
    "cardiologist": "cardiology", "dermatology": "dermatology", "dermatologist": "dermatology",
    "dentistry": "dentistry", "dentist": "dentistry", "ophthalmology": "ophthalmology",
    "neurologist": "neurology", "gynecologist": "gynecology", "obstetrician-gynecologist": "gynecology",
    "general practitioner": "general",
})

def _norm_specialty(s: str) -> str:
    toks = [t[2:] if t.startswith("ال") and len(t) > 3 else t for t in s.split()]
    toks = [t for t in toks if t not in _SPEC_PREFIX]
    core = " ".join(toks).strip()
    return _SPEC_ALIAS.get(core, core or s)

# category: map the dense, high-frequency core (Arabic spellings + English) to a
# canonical token. Genuinely distinct concepts stay distinct (watch != smartwatch,
# phone != smartphone); the long tail of one-offs is left unmapped (no merge).
CATEGORY_FIELDS = {"category"}
_CATEGORY = {
    "laptop": ["لابتوب", "لاب توب", "اللاب توب", "كمبيوتر محمول", "حاسوب محمول",
               "كومبيوتر محمول", "جهاز لابتوب", "ابتوب", "laptop"],
    "computer": ["كمبيوتر", "كومبيوتر", "حاسوب", "كمبيوتر مكتبي", "جهاز كمبيوتر", "computer"],
    "electronics": ["الكترونيات", "اجهزة الكترونية", "جهاز الكتروني", "اجهزة الكترونيه",
                    "electronics", "electronic device", "device"],
    "clothes": ["ملابس", "لبس", "لبسة", "حوايج", "clothes"],
    "fashion": ["ازياء", "موضة", "fashion"],
    "camera": ["كاميرا", "كاميرات", "الات التصوير", "اجهزة تصوير", "camera"],
    "watch": ["ساعة", "ساعة يد", "ساعات", "watch", "watches"],
    "smartwatch": ["ساعة ذكية", "smartwatch"],
    "phone": ["موبايل", "جوال", "جوالات", "هاتف", "هواتف", "هاتف محمول", "phone", "mobile phone"],
    "smartphone": ["هاتف ذكي", "smartphone"],
    "tablet": ["تابلت", "جهاز لوحي", "tablet"],
    "tv": ["تلفزيون", "تليفزيون", "تلفاز", "tv"],
    "gaming": ["جهاز العاب", "بلايستيشن", "كونسول", "playstation", "xbox"],
    "bag": ["حقيبة", "حقيبة يد", "شنطة", "bag"],
    "jewelry": ["مجوهرات", "jewelry"],
    "accessories": ["اكسسوارات", "اكسسوار"],
}
CAT_ALIAS = {}
for _c, _forms in _CATEGORY.items():
    for _f in _forms:
        CAT_ALIAS[arabic_ortho(_f).casefold()] = _c

# date fields: normalize named days + relative terms + month names across Arabic
# and English so "الخميس"="يوم الخميس"="Friday", "بكرة"="tomorrow",
# "الأسبوع الجاي"="next week"="next_week". Absolute ISO dates (2023-11-10) are
# left exact. Does NOT reconcile a natural date vs a resolved ISO date.
DATE_FIELDS = {"date", "check_in", "check_out", "departure_date", "return_date", "appointment_date"}
_DAY = {
    "sunday": ["الاحد", "sunday"], "monday": ["الاثنين", "monday"], "tuesday": ["الثلاثاء", "الثلاثا", "tuesday"],
    "wednesday": ["الاربعاء", "wednesday"], "thursday": ["الخميس", "thursday"],
    "friday": ["الجمعة", "friday"], "saturday": ["السبت", "saturday"],
}
_REL = {
    "today": ["اليوم", "today"],
    "tomorrow": ["غدا", "بكرة", "باكر", "باجر", "بكره", "tomorrow"],
    "day_after_tomorrow": ["بعد غد", "بعد غدا", "بعد بكرة", "بعد بكره", "بعد باجر", "بعد باكر",
                            "the day after tomorrow", "day after tomorrow", "after tomorrow"],
    "next_week": ["الاسبوع القادم", "الاسبوع الجاي", "الاسبوع المقبل", "next week", "coming week"],
    "next_month": ["الشهر القادم", "الشهر الجاي", "الشهر المقبل", "next month"],
    "weekend": ["نهاية الاسبوع", "عطلة نهاية الاسبوع", "weekend", "the weekend"],
}
_MONTH = {m: [m, m[:3]] + ars for m, ars in {
    "january": ["يناير"], "february": ["فبراير"], "march": ["مارس"], "april": ["ابريل", "إبريل"],
    "may": ["مايو", "ماي"], "june": ["يونيو", "يونيه"], "july": ["يوليو", "يوليوز", "يوليه"],
    "august": ["اغسطس", "غشت"], "september": ["سبتمبر", "شتنبر"], "october": ["اكتوبر"],
    "november": ["نوفمبر", "نونبر"], "december": ["ديسمبر", "دجنبر"]}.items()}
_norm = lambda s: arabic_ortho(to_ascii_digits(s)).strip().casefold()
DATE_ALIAS = {}
for _t in (_DAY, _REL):
    for _c, _fs in _t.items():
        for _f in _fs: DATE_ALIAS[_norm(_f)] = _c
_DAY_TOK = {_norm(f): c for c, fs in _DAY.items() for f in fs}
_MONTH_TOK = {_norm(f): c for c, fs in _MONTH.items() for f in fs}
_QUAL = re.compile(r"\s+(القادم|القادمه|الجاي|الجايه|المقبل|المقبله|القادمة|الجاية|المقبلة)$")

def _norm_date(s: str) -> str:
    s = s.replace("_", " ").strip()
    if s in DATE_ALIAS: return DATE_ALIAS[s]
    s = re.sub(r"^يوم\s+", "", s)
    s2 = _QUAL.sub("", s)
    if s2 in DATE_ALIAS: return DATE_ALIAS[s2]
    if s2 in _DAY_TOK: return _DAY_TOK[s2]
    return " ".join(_MONTH_TOK.get(t, _DAY_TOK.get(t, t)) for t in s2.split()).strip() or s

# name fields: map common Arabic personal names to/from their English
# transliterations so "أحمد" = "Ahmed", "صديقي" = "my friend". Token-wise, so
# compound names ("فاطمة حسين" = "Fatima Hussein") work.
NAME_FIELDS = {"recipient_name", "doctor_name"}
_NAMES = {
    "احمد": ["ahmed", "ahmad"], "محمد": ["mohammed", "mohamed", "muhammad", "mohammad", "mohamad"],
    "علي": ["ali"], "عبدالله": ["abdullah", "abdallah", "abdulla"], "حسن": ["hassan", "hasan"],
    "حسين": ["hussein", "hussain", "husain"], "فاطمه": ["fatima", "fatimah"], "عمر": ["omar", "umar"],
    "خالد": ["khalid", "khaled"], "ساره": ["sara", "sarah"], "مريم": ["maryam", "mariam"],
    "ماريا": ["maria"], "نوره": ["noura", "nora"], "يوسف": ["youssef", "yusuf", "yousef"],
    "ابراهيم": ["ibrahim"], "سعد": ["saad"], "فهد": ["fahd", "fahad"], "عبدالرحمن": ["abdulrahman", "abdelrahman"],
}
_NAME_TOK = {}
for _c, _fs in _NAMES.items():
    key = arabic_ortho(_c).casefold()
    _NAME_TOK[key] = key
    for _f in _fs: _NAME_TOK[_f.casefold()] = key
_NAME_PHRASE = {"my friend": "صديقي", "my brother": "اخي", "my sister": "اختي", "my father": "والدي",
                "my mother": "والدتي", "my wife": "زوجتي", "my husband": "زوجي", "my son": "ابني"}
_NAME_PHRASE = {k: arabic_ortho(v).casefold() for k, v in _NAME_PHRASE.items()}

def _norm_name(s: str) -> str:
    if s in _NAME_PHRASE: return _NAME_PHRASE[s]
    return " ".join(_NAME_TOK.get(t, t) for t in s.split()).strip() or s

# ────────────────────────── public matching API ────────────────────────────
def canon_value(v, field: str = "") -> str:
    s = to_ascii_digits(str(v))
    s = arabic_ortho(s)
    s = words_to_digits(s)
    s = re.sub(r"\s+", " ", s).strip().casefold()
    cls = FIELD_CLASS.get(field)
    if cls and s in _ALIAS[cls]:
        return _ALIAS[cls][s]
    if field in PRODUCT_FIELDS:
        return _norm_product(s)
    if field in SPECIALTY_FIELDS:
        return _norm_specialty(s)
    if field in CATEGORY_FIELDS:
        return CAT_ALIAS.get(s, s)
    if field in DATE_FIELDS:
        return _norm_date(s)
    if field in NAME_FIELDS:
        return _norm_name(s)
    return s

def _split_set(v, field: str) -> frozenset:
    s = to_ascii_digits(str(v))
    s = _WAW_ATTACHED.sub(" ، ", s)
    parts = [p for p in _SEP.split(s) if p and p not in ("،", ",", ";", "&", "/", "و", "and")]
    if not parts:
        parts = [s]
    return frozenset(canon_value(p, field) for p in parts if p.strip())

def value_match(pred, gold, field: str = "") -> bool:
    """True iff pred and gold are the same answer under the normalisation rules."""
    # numeric equivalence (quantity fields only)
    if field not in ID_FIELDS:
        pn, gn = _as_number(pred), _as_number(gold)
        if pn is not None and gn is not None:
            return pn == gn
    if field in LIST_FIELDS:
        return _split_set(pred, field) == _split_set(gold, field)
    return canon_value(pred, field) == canon_value(gold, field)

# ── Optional parameters that are declared in a tool's schema but never appear in
# any gold answer (derived from the released train+dev gold). They are neither
# required nor scored: stripped from BOTH prediction and gold before matching, so
# a model may freely emit OR omit them (e.g. a model that infers `source_language`
# for translate_text is not penalised). Exact-set matching is unchanged for every
# real argument, and this is tool-scoped — a key ignored for one tool can still be
# a scored argument for another (e.g. `country` is ignored for get_weather but
# scored for compare_prices). NOT a blanket "ignore all extras" rule.
OPTIONAL_IGNORE: dict[str, set[str]] = {
    "translate_text":           {"source_language"},
    "check_traffic_violations": {"plate_number"},
    "get_qibla_direction":      {"latitude", "longitude"},
    "get_weather":              {"country"},
    "calculate_end_of_service": {"country"},
    "calculate_zakat":          {"weight_unit"},
    "check_iqama_status":       {"border_number"},
    "order_food":               {"delivery_address"},
    "search_hotels":            {"stars"},
    # search_type: annotated in <5% of search_quran gold (inconsistently) while a
    # type-word (آية/سورة/تفسير…) appears in most queries — unlearnable either way,
    # so it is not scored (same rationale as source_language).
    "search_quran":             {"surah_number", "search_type"},
    "search_umrah_packages":    {"duration_days", "hotel_rating"},
}

def args_match(pred_args: dict, gold_args: dict, tool: str | None = None) -> bool:
    """All-or-nothing ArgEM over a row, with per-value normalised matching.

    Optional parameters in OPTIONAL_IGNORE[tool] (declared in schema but never in
    gold) are dropped from both sides first, so emitting them is neither rewarded
    nor penalised. All other keys must match exactly (set equality + value_match).
    """
    ignore = OPTIONAL_IGNORE.get(tool or "", set())
    pg = {k: v for k, v in (pred_args or {}).items()
          if v is not None and str(v).strip() != "" and k not in ignore}
    gg = {k: v for k, v in (gold_args or {}).items()
          if v is not None and str(v).strip() != "" and k not in ignore}
    if set(pg.keys()) != set(gg.keys()):
        return False
    return all(value_match(pg[k], gg[k], k) for k in gg)
