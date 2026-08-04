"""
scheduler/classifier.py — Arabic text → medical specialty routing.

Uses regex-based rules on normalized Arabic text to match patient
complaints to one of 8 clinic specialties. Falls back to Gemini
LLM when no rule matches (async path).
"""
import re
import logging
from nlp.normalizer import normalize

logger = logging.getLogger(__name__)

ROUTING_RULES: list[tuple[str, str]] = [
        
    # # ── CARDIOLOGY ─ heart / chest ──
    # (
    #     r"قلب|صدر|خفقان|ذبح[هة]\s*صدر|"
    #     r"ضغط\s*القلب|ألم\s*صدر|الم\s*صدر|"
    #     r"قصور\s*قلبي|جلط[هة]\s*قلب",
    #     "cardiology",
    # ),

    # ── NEUROLOGY ─ acute neuro symptoms + breathing difficulty ──
    (
        r"صداع.*شديد|شلل|رعشه|"
        r"تنميل|اعصاب|اغماء|تشنج|صرع|"
        r"ضعف\s*مفاجئ|صعوبه\s*نطق|تشوش.*كلام|"
        r"فقدان\s*توازن|جلطه|سكت[هة]|"
        r"دوخه\s*شديده|دوخه.*تنميل|"
        r"ضيق\s*تنفس.*دوخه|صعوبه\s*تنفس.*دوخه|"
        r"دوخه.*شلل|تشنجات\s*قويه",
        "neurology",
    ),
    
    # ── ORTHOPEDICS ─ musculoskeletal (fixed bare "وجع/الم" patterns) ──
    (
        r"كسر|التواء|عظام|مفاصل|"
        r"(الم|وجع)(?:\s*في)?\s*(?:ال)?(?:ظهر|رقبه|ركبه|مفصل|كتف|عضله|ورك|ساق|طرف|ذراع|يد|رجل)|"
        r"الركبه|الورك|الكتف|الساق|الظهر|"
        r"عمود\s*فقري|وتر|وقعه|سقوط|"
        r"لا.*يستطيع.*مشي|لا.*استطيع.*الحركه|"
        r"الم.*مزمن.*(ظهر|رقبه|عظام)|"
        r"تمزق|تورم.*(ركبه|ورك|كتف)|"
        r"التواء.*(ركبه|ورك|كتف)",
        "orthopedics",
    ),
    
    # ── GYNECOLOGY ─ OB/GYN context (added "عياده النساء") ──
    (
        r"حمل|ولاده|دوره\s*شهريه|اضطرابات.*دوره|"
        r"(الم|وجع)\s*رحم|(الم|وجع)\s*مبيض|"
        r"(الم|وجع)\s*بطن.*حامل|الم.*حمل|"
        r"نزيف\s*رحمي|نزيف.*حمل|"
        r"مشاكل\s*حمل|ولاده\s*مبكره|تقلصات.*حمل|"
        r"عياد[هة]?\s*نساء|عياد[هة]?\s*توليد|"
        r"توليد|مهبل|فحص\s*مهبلي|"
        r"تنظيم\s*اسره|مشاكل.*نسائيه|صحه.*انجابيه",
        "gynecology",
    ),
    
    # # ── PEDIATRICS ─ child-specific ──
    # (
    #     r"طفل|طفله|طفلي|رضيع|"
    #     r"ابني\s*صغير|ابنتي\s*صغيره|"
    #     r"حمى\s*اطفال|كح[eh]\s*اطفال|"
    #     r"عمر[اه]?\s*(?:[1-9]|1[0-7])\s*سنه",
    #     "pediatrics",
    # ),

    # # ── OPHTHALMOLOGY ─ eye ──
    # (
    #     r"عيون|عين|ضعف\s*نظر|"
    #     r"احمرار\s*عين|الم\s*عين|"
    #     r"عدم\s*وضوح\s*الرؤيه|رمد",
    #     "ophthalmology",
    # ),

    # ── DERMATOLOGY ─ skin-specific (fixed bare patterns) ──
    (
        r"طفح.*جلد|طفح\s*حاد|"
        r"(حكه|حكه.*جلد|احمرار.*جلد|احمرار|"
        r"التهاب.*جلد|التهاب|"
        r"حساسيه.*جلد|حساسيه|"
        r"جفاف\s*جلد|جفاف|"
        r"اكزيما|صدفيه|"
        r"بقع\s*جلد|بقع.*جلد|بقع|"
        r"شرى|قشره.*جلد|قشره|"
        r"حب\s*الشباب|حب|"
        r"تورم.*وجه|تورم.*جسم)",
        "dermatology",
    ),
    
    # # ── DENTISTRY ─ dental ──
    # (
    #     r"اسنان|سن\s*مكسور|ضرس|"
    #     r"الم\s*اسنان|التهاب\s*لث[eh]|"
    #     r"خلع\s*ضرس|تسوس",
    #     "dentistry",
    # ),

    # # Pulmonology and respiratory
    # (
    #     r"رئة|تنفس|ربو|"
    #     r"كحة مزمنة|اختناق|بلغم|التهاب رئة", 
    #     "pulmonology",
    # ),
    
    # ── GASTROENTEROLOGY ─ GI symptoms ──
    (
        r"(الم|وجع)\s*معده|(الم|وجع)\s*بطن|"
        r"مغص|اسهال|امساك|"
        r"قولون|بواسير|كبد|"
        r"(ترجيع|قيء|غثيان).*مستمر|"
        r"حرقه\s*معده|انتفاخ.*بطن|"
        r"عسر\s*هضم|سوء\s*هضم|"
        r"تسمم\s*غذائي|نزيف.*جهاز.*هضمي|"
        r"الم\s*كبد",
        "gastroenterology",
    ),
    
    # ── CHRONIC DISEASES ─ disease program + critical values ──
    (
        r"سكري|سكر\s*مرتفع|السكر|ارتفاع\s*سكر|هبوط\s*سكر|"
        r"انسولين|السكر\s*التراكمي|"
        r"متابع[هة]\s*سكر|فحص\s*سكر|"
        r"ضغط\s*الدم|ضغط\s*مرتفع|"
        r"قراءات?\s*ضغط|متابع[هة].*ضغط|"
        r"ادويه.*ضغط|"
        r"ربو|ازمه\s*ربو|"
        r"كولسترول|دهون|"
        r"غده\s*درقيه|"
        r"مرض\s*مزمن|امراض\s*مزمنه|"
        r"متابع[هة].*مزمنه|"
        r"تجديد.*وصفه|تجديد.*ادويه",
        "chronic_diseases",
    ),
    
    # ── ELDERLY ─ geriatric program (better age patterns) ──
    (
        r"جدي|جدتي|والدي.*مسن|والدتي.*مسن[هة]|"
        r"(كبار\s*السن|كبير\s*سن|كبيره\s*سن|"
        r"مسن|مسنه|شيخوخه)|"
        r"خرف|ذاكره|نسيان|اختلاط\s*ادويه|"
        r"عمر[اه]?\s*([6789]\d)|"  # عمره 60-99
        r"ضعف\s*عام.*مسن|عدم.*القدره.*على.*الاعتناء",
        "elderly",
    ),

    # ── CARDIOLOGY ─ heart / chest ──
    (
        r"قلب|صدر|خفقان|ذبح[هة]\s*صدر|"
        r"ضغط\s*القلب|ألم\s*صدر|الم\s*صدر|"
        r"قصور\s*قلبي|جلط[هة]\s*قلب",
        "cardiology",
    ),

    # ── PEDIATRICS ─ child-specific ──
    (
        r"طفل|طفله|طفلي|رضيع|"
        r"ابني\s*صغير|ابنتي\s*صغيره|"
        r"حمى\s*اطفال|كح[eh]\s*اطفال|"
        r"عمر[اه]?\s*(?:[1-9]|1[0-7])\s*سنه",
        "pediatrics",
    ),

    # ── DENTISTRY ─ dental ──
    (
        r"اسنان|سن\s*مكسور|ضرس|"
        r"الم\s*اسنان|التهاب\s*لث[eh]|"
        r"خلع\s*ضرس|تسوس",
        "dentistry",
    ),

    # ── OPHTHALMOLOGY ─ eye ──
    (
        r"عيون|عين|ضعف\s*نظر|"
        r"احمرار\s*عين|الم\s*عين|"
        r"عدم\s*وضوح\s*الرؤيه|رمد",
        "ophthalmology",
    ),

    # ── GENERAL PRACTICE ─ routine / mild acute (last resort) ──
    (
        r"حمى|زكام|كحه.*خفيفه|كحه\s*بسيطه|"
        r"ارهاق|وهن|تعب\s*عام|"
        r"(فحص|كشف)\s*روتيني|مراجع[هة]\s*عامه|"
        r"استشاره.*عامه|اعراض\s*بسيطه|"
        r"متابع[هة]\s*عامه",
        "general_practice",
    ),
]

# ── Human-readable Arabic specialty names ─────────────────────────────────────
SPECIALTY_NAMES_AR = {
    # "cardiology":        "القلب والأوعية",
    "neurology":         "طب الأعصاب",
    "orthopedics":       "العظام والمفاصل",
    "gynecology":        "النساء والتوليد",
    # "pediatrics":        "طب الأطفال",
    # "ophthalmology":     "طب العيون",
    "dermatology":       "الأمراض الجلدية",
    # "dentistry":         "طب الأسنان",
    # "pulmonology":       "طب الصدر والجهاز التنفسي",
    "gastroenterology":  "الجهاز الهضمي",
    "chronic_diseases":  "الأمراض المزمنة",
    "elderly":           "كبار السن",
    "general_practice":  "الطب العام",
}

SPECIALTY_KEYS = set(SPECIALTY_NAMES_AR.keys())
_VALID_KEY_PATTERN = re.compile(r"[a-z_]+")

# Clinics the product does not operate — detect patient requests early.
UNSUPPORTED_SPECIALTY_PATTERNS: list[tuple[str, str]] = [
    (r"اسنان|أسنان|ضرس|لث[eh]|تسوس|خلع\s*ضرس", "طب الأسنان"),
    (r"صدر(?:يه|ية)?|رئ[eh]|تنفس.*مزمن|التهاب\s*رئ[eh]|pulmonology", "طب الصدر"),
    (r"قلب|cardiology|ذبح[هة]\s*صدر|خفقان\s*شديد", "القلب والأوعية"),
    (r"اطفال|أطفال|pediatrics|رضيع|طفل\s*صغير", "طب الأطفال"),
    (r"عيون|ophthalmology|ضعف\s*نظر|احمرار\s*عين", "طب العيون"),
    (r"ent\b|اذن|أذن|انف|أنف|حلق\s*واذن", "ENT"),
]


def detect_unsupported_specialty(text: str) -> str | None:
    """Return Arabic clinic label if text requests a clinic we do not offer."""
    normalized = normalize(text or "")
    if not normalized:
        return None
    for pattern, label in UNSUPPORTED_SPECIALTY_PATTERNS:
        if re.search(pattern, normalized):
            return label
    return None


def is_supported_specialty(key: str | None) -> bool:
    return bool(key) and key in SPECIALTY_KEYS


def _parse_gemini_specialty_key(raw: str) -> str | None:
    """Extract a valid specialty key from a Gemini free-text response."""
    if not raw or not raw.strip():
        return None
    text = raw.strip().lower()
    for token in _VALID_KEY_PATTERN.findall(text):
        if token in SPECIALTY_KEYS:
            return token
    compact = re.sub(r"[^a-z_]", "", text)
    for key in sorted(SPECIALTY_KEYS, key=len, reverse=True):
        if key in compact:
            return key
    return None


def classify_specialty(text: str) -> dict:
    """
    Rule-based classifier. Always normalizes input text first.
    
    IMPROVEMENTS:
    - More robust pattern matching
    - Better handling of edge cases
    - Returns specialty key + Arabic name
    - Fallback: general_practice if no pattern matches
    """
    normalized_text = normalize(text or "")
    
    # Iterate through rules and check for matches
    for pattern, specialty in ROUTING_RULES:
        if re.search(pattern, normalized_text):
            return {
                "specialty":    specialty,
                "specialty_ar": SPECIALTY_NAMES_AR.get(specialty, specialty),
                "method":       "rule",
                "confidence":   0.9,
            }
    
    # Fallback to general practice
    return {
        "specialty":    "general_practice",
        "specialty_ar": SPECIALTY_NAMES_AR["general_practice"],
        "method":       "default",
        "confidence":   0.5,
    }


def auto_resolve_specialty(result: dict) -> dict:
    """Pick a clinic automatically — never leave booking blocked on patient specialty choice."""
    if result.get("method") != "default":
        return result
    return {
        "specialty": "general_practice",
        "specialty_ar": SPECIALTY_NAMES_AR["general_practice"],
        "method": "auto_fallback",
        "confidence": result.get("confidence", 0.4),
    }


async def classify_with_gemini_fallback(text: str, gemini_client) -> dict:
    """
    Try rules first. If unresolved, ask Gemini and validate response.
    Use this for ambiguous or multi-complaint messages.
    """
    result = classify_specialty(text)

    if result["method"] == "default":
        normalized_text = normalize(text or "")
        prompt = (
            f"المريض يقول: '{normalized_text}'\n\n"
            f"اختر التخصص الأنسب من هذه المفاتيح فقط:\n"
            + "\n".join(f"- {k}: {v}" for k, v in SPECIALTY_NAMES_AR.items())
            + "\n\nأجب بمفتاح واحد فقط من القائمة بالإنجليزية (مثل: neurology). بدون أي نص إضافي."
        )
        try:
            gemini_result = await gemini_client.ask(prompt, max_tokens=20)
            if not (gemini_result or "").strip():
                raise ValueError("Empty response from Gemini")
            specialty_key = _parse_gemini_specialty_key(gemini_result)
            if specialty_key:
                result = {
                    "specialty":    specialty_key,
                    "specialty_ar": SPECIALTY_NAMES_AR[specialty_key],
                    "method":       "gemini",
                    "confidence":   0.85,
                }
            else:
                logger.warning("Gemini returned unknown specialty key: %s", gemini_result)
        except Exception as exc:
            logger.exception("Gemini fallback failed: %s", exc)

    return auto_resolve_specialty(result)