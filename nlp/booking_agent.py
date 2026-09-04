"""
nlp/booking_agent.py — LLM-first patient booking conversation turn.

Each turn returns a natural Arabic reply plus structured field extraction.
Falls back to rule-based extractor when the LLM is unavailable.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from nlp.extractor import extract_patient_fields
from nlp.gemini_client import OFF_TOPIC_REPLY, GeminiClient, gemini
from nlp.normalizer import normalize

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 12

VALID_INTENTS = frozenset({
    "continue",
    "confirm",
    "decline",
    "cancel",
    "accept_gp",
    "reject_gp",
    "inquiry",
    "contact",
    "new_booking",
    "next_slot",
    "slot_list",
    "edit_time",
    "off_topic",
})

FIELD_QUESTIONS_AR = {
    "name": "ما اسمك الكريم؟ 😊",
    "complaint": "سلامتك! شو الأعراض أو سبب الزيارة؟ 🩺",
    "urgency_score": "كيف شايف حالة المريض؟ (عاجل / متوسط / روتيني)",
    "time_pref": "متى يناسبك الموعد؟ (اليوم، بكرا، الأسبوع الجاي...)",
}


@dataclass
class BookingTurnResult:
    reply: str
    intent: str = "continue"
    extracted: dict[str, Any] = field(default_factory=dict)
    off_topic: bool = False


def trim_chat_history(history: list[dict]) -> list[dict]:
    if len(history) <= MAX_HISTORY_TURNS * 2:
        return history
    return history[-(MAX_HISTORY_TURNS * 2) :]


def append_history(history: list[dict], role: str, content: str) -> list[dict]:
    text = (content or "").strip()
    if not text:
        return history
    out = list(history) + [{"role": role, "content": text}]
    return trim_chat_history(out)


def _parse_json_response(raw: str) -> dict:
    clean = (raw or "").strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        repaired = _repair_truncated_booking_json(clean)
        if repaired is not None:
            return repaired
        raise


def _repair_truncated_booking_json(raw: str) -> dict | None:
    """Best-effort parse when Gemini truncates mid-JSON (common with 2.5 thinking)."""
    if not raw or '"reply"' not in raw:
        return None
    # Extract reply string even if the closing quote/braces were cut off.
    m = re.search(r'"reply"\s*:\s*"((?:\\.|[^"\\])*)', raw)
    reply = ""
    if m:
        reply = m.group(1)
        try:
            reply = json.loads(f'"{reply}"')
        except json.JSONDecodeError:
            reply = reply.replace('\\"', '"').replace("\\n", "\n")
    intent_m = re.search(r'"intent"\s*:\s*"([a-z_]+)"', raw)
    intent = intent_m.group(1) if intent_m else "continue"
    if intent not in VALID_INTENTS:
        intent = "continue"
    # Prefer a short usable reply over failing the whole turn.
    if not reply.strip():
        return None
    return {
        "reply": reply.strip()[:400],
        "intent": intent,
        "off_topic": False,
        "extracted": {"name": None, "complaint": None, "urgency": None, "time_pref": None},
    }


def _urgency_label_to_score(label: str | None) -> float | None:
    if not label:
        return None
    norm = normalize(label).lower()
    if norm in ("urgent", "عاجل", "طارئ", "فوري"):
        return 0.9
    if norm in ("medium", "متوسط", "عادي"):
        return 0.5
    if norm in ("routine", "روتيني"):
        return 0.2
    if any(w in norm for w in ("عاجل", "طارئ", "فوري", "خطير")):
        return 0.9
    if any(w in norm for w in ("متوسط", "خلال اسبوع", "عادي")):
        return 0.5
    if any(w in norm for w in ("روتيني", "مش عاجل", "اي وقت")):
        return 0.2
    return None


def _parse_time_phrase(text: str) -> dict | None:
    if not text:
        return None
    from datetime import date, timedelta

    t = normalize(text).lower()
    today = date.today()
    if "بعد بكرا" in t or "بعد غد" in t:
        return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
    if "اليوم" in t or "هلق" in t or "الان" in t:
        return {"date": str(today), "phrase": "اليوم"}
    if "بكرا" in t or "غدا" in t:
        return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
    if "اسبوع" in t or "أسبوع" in t:
        return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
    if "اي وقت" in t or "أي وقت" in t or "لا يهم" in t:
        return {"date": None, "phrase": "أي وقت متاح"}
    return {"date": None, "phrase": text.strip()}


def _validate_name(name: str | None) -> str | None:
    if not name:
        return None
    raw = name.strip()
    if not raw or raw.startswith("/"):
        return None
    if any(ch.isdigit() for ch in raw):
        return None
    words = raw.split()
    if not (1 <= len(words) <= 4) or len(raw) > 40:
        return None
    norm = normalize(raw)
    if any(w in norm for w in ("موعد", "حجز", "عاجل", "روتيني", "اليوم", "بكرا")):
        return None
    return raw


def _message_has_urgency_signal(text: str) -> bool:
    norm = normalize(text or "").lower()
    tokens = (
        "عاجل", "طارئ", "فوري", "خطير", "روتيني", "متوسط", "عادي",
        "p1", "p2", "p3", "🔴", "🟡", "🟢",
    )
    return any(t in norm for t in tokens)


_COMPLAINT_HINT_WORDS = (
    "الم", "وجع", "مرض", "صداع", "كحة", "سكري", "شكوى", "عندي",
    "تنميل", "خدر", "دوخه", "دوخة", "حكه", "حكة", "طفح", "كسر",
    "نزيف", "حراره", "حرارة", "غثيان", "اقياء", "إقياء", "ضيق",
    "خفقان", "رعشه", "رعشة", "تورم", "انتفاخ", "مغص", "اسهال",
)


def _message_looks_like_complaint(text: str) -> bool:
    """Heuristic: symptom-like free text even when not in symptoms.json."""
    raw = (text or "").strip()
    if len(raw) < 3:
        return False
    if _message_has_urgency_signal(raw) and len(raw.split()) <= 2:
        return False
    norm = normalize(raw)
    if any(w in norm for w in _COMPLAINT_HINT_WORDS):
        return True
    # «ايدي / ايدي اليسار / رجلي» alone after short context
    body = ("ايد", "يد", "رجل", "راس", "بطن", "صدر", "ظهر", "رقبه", "رقبة")
    return any(w in norm for w in body) and len(norm.split()) >= 2


_BOOKING_DONE_MARKERS = (
    "تم تاكيد حجز", "تم تأكيد حجز", "تم الحجز", "تم حجزك", "تم تثبيت الموعد",
    "حجزك صار", "الموعد محجوز", "رقم الحجز", "حفظ ملفك في النظام",
    "سيظهر الموعد تلقائيا", "confirmed your appointment", "booking confirmed",
)


def reply_claims_booking_done(reply: str) -> bool:
    """True when the LLM invents a successful booking confirmation."""
    norm = normalize(reply or "")
    if not norm:
        return False
    return any(normalize(m) in norm for m in _BOOKING_DONE_MARKERS)


def merge_rule_extracted(user_message: str, collected: dict) -> dict[str, Any]:
    """Rule-based extraction to merge with or replace LLM output."""
    fields = extract_patient_fields(user_message)
    out: dict[str, Any] = {}

    if not collected.get("name") and fields.get("name"):
        name = _validate_name(fields["name"])
        if name:
            out["name"] = name

    if not collected.get("complaint") and fields.get("complaint"):
        out["complaint"] = fields["complaint"].get("raw") or fields["complaint"]
    elif not collected.get("complaint") and _message_looks_like_complaint(user_message):
        out["complaint"] = user_message.strip()

    if _message_has_urgency_signal(user_message):
        label_score = _urgency_label_to_score(user_message)
        if label_score is not None:
            out["urgency_score"] = label_score
        elif collected.get("urgency_score") is None and fields.get("urgency_score") is not None:
            out["urgency_score"] = fields["urgency_score"]

    tp = fields.get("time_pref") or {}
    if not collected.get("time_pref") and (tp.get("date") or tp.get("phrase")):
        out["time_pref"] = tp.get("phrase") or tp.get("date")

    return out


def apply_extracted_to_data(
    data: dict,
    extracted: dict[str, Any],
    *,
    score_from_label,
) -> None:
    """Merge agent/rule extracted fields into FSM data dict."""
    name = extracted.get("name")
    if name and not data.get("name"):
        valid = _validate_name(str(name))
        if valid:
            data["name"] = valid

    complaint = extracted.get("complaint")
    if complaint and not data.get("complaint"):
        raw = str(complaint).strip()
        if len(raw) >= 2:
            data["complaint"] = {
                "raw": raw,
                "category": "general",
                "urgency_score": 0.3,
                "specialty": "general_practice",
            }

    urgency = extracted.get("urgency") or extracted.get("urgency_score")
    if urgency is not None:
        if isinstance(urgency, (int, float)):
            data["urgency_score"] = float(urgency)
        else:
            score = _urgency_label_to_score(str(urgency))
            if score is None and score_from_label:
                score = score_from_label(str(urgency))
            if score is not None:
                data["urgency_score"] = score

    time_pref = extracted.get("time_pref")
    if not data.get("time_pref") and time_pref:
        if isinstance(time_pref, dict):
            data["time_pref"] = time_pref
        else:
            mapped = _parse_time_phrase(str(time_pref))
            if mapped:
                data["time_pref"] = mapped

    if data.get("time_pref") and isinstance(data["time_pref"], str):
        mapped = _parse_time_phrase(data["time_pref"])
        if mapped:
            data["time_pref"] = mapped


def missing_required_fields(data: dict, required: list[str]) -> list[str]:
    missing = []
    for field_name in required:
        val = data.get(field_name)
        if val is None:
            missing.append(field_name)
        elif field_name == "time_pref" and isinstance(val, dict) and not (val.get("date") or val.get("phrase")):
            missing.append(field_name)
        elif field_name == "complaint" and not val:
            missing.append(field_name)
    return missing


def fallback_reply(
    collected: dict,
    required: list[str],
    user_message: str,
    *,
    rule_hint: str | None = None,
    phase: str = "CHATTING",
) -> BookingTurnResult:
    """Template reply when LLM is unavailable."""
    merged = merge_rule_extracted(user_message, collected)
    missing = missing_required_fields(collected, required)
    rule_intent = detect_rule_intent(user_message, rule_hint=rule_hint, phase=phase)


    if phase == "CONFIRM":
        confirm_intent = detect_confirm_intent(user_message) or rule_intent
        if confirm_intent in VALID_INTENTS:
            return BookingTurnResult(reply="", intent=confirm_intent, extracted=merged)
        return BookingTurnResult(
            reply="لسا معك بالحجز. بدك تأكيد الموعد، تشوف وقت ثاني، أو نلغي؟",
            intent="continue",
            extracted=merged,
        )

    if phase in ("TERMINAL", "GP_FALLBACK"):
        if rule_intent in VALID_INTENTS:
            return BookingTurnResult(reply="", intent=rule_intent, extracted=merged)
        return BookingTurnResult(reply="", intent="continue", extracted=merged)

    if GeminiClient.looks_off_topic(user_message):
        first = missing[0] if missing else "name"
        return BookingTurnResult(
            reply=f"{OFF_TOPIC_REPLY}\n\n{FIELD_QUESTIONS_AR.get(first, FIELD_QUESTIONS_AR['name'])}",
            intent="off_topic",
            extracted=merged,
            off_topic=True,
        )

    if rule_intent in VALID_INTENTS:
        if rule_intent == "inquiry":
            return BookingTurnResult(
                reply="",
                intent="inquiry",
                extracted=merged,
            )
        if rule_intent == "contact":
            return BookingTurnResult(
                reply="📞 تواصلك وصل. يمكنك كتابة رسالتك هنا، وسيتم حفظها في سجل المحادثات للعيادة.",
                intent="contact",
                extracted=merged,
            )
        if rule_intent == "new_booking":
            return BookingTurnResult(
                reply="📅 تمام، خلينا نبدأ حجز جديد. " + FIELD_QUESTIONS_AR["name"],
                intent="new_booking",
                extracted=merged,
            )
        if rule_intent == "cancel":
            return BookingTurnResult(reply="", intent="cancel", extracted=merged)

    if not missing and merged:
        return BookingTurnResult(
            reply="تمام، خلينا نكمل الحجز. 👍",
            intent="continue",
            extracted=merged,
        )

    first = missing[0] if missing else "name"
    return BookingTurnResult(
        reply=FIELD_QUESTIONS_AR.get(first, FIELD_QUESTIONS_AR["name"]),
        intent=rule_intent or "continue",
        extracted=merged,
    )


def detect_confirm_intent(user_message: str) -> str | None:
    """Map Palestinian Arabic at CONFIRM to FSM intents (backup when LLM JSON fails)."""
    norm = normalize(user_message or "").lower()
    if not norm:
        return None

    cancel_tokens = ("الغاء", "إلغاء", "الغي", "إلغي", "كنسل", "cancel", "لا ")
    if any(t in norm for t in cancel_tokens) and not any(
        w in norm for w in ("بدي", "اريد", "حاب", "موعد")
    ):
        return "decline"

    confirm_tokens = (
        "نعم", "ايوه", "آيوه", "تمام", "ماشي", "موافق", "احجز", "تاكيد", "تأكيد",
        "اه", "آه", "ايه", "يلا", "ok", "اوك", "اوكي", "مناسب", "حاضر", "كويس",
        "ممتاز", "اكيد", "أكيد", "صح", "ثبت", "خلص", "طيب", "حسنا", "confirm", "sure",
    )
    if any(t in norm for t in confirm_tokens):
        return "confirm"

    next_slot_phrases = (
        "موعد آخر", "موعد تاني", "موعد ثاني", "وقت تاني", "وقت ثاني",
        "غير الموعد", "بدي غير", "بدي اغير", "بدي أغير", "مش هاد", "مو بدي هاد",
        "🔄",
    )
    if any(p in norm for p in next_slot_phrases):
        return "next_slot"

    if any(p in norm for p in ("اشوف المواعيد", "شو المواعيد", "خيارات", "بدائل", "فرجيني", "ورجيني")):
        return "slot_list"

    edit_phrases = (
        "تعديل", "تعديل الموعد", "غير الوقت", "بدي وقت", "بدي موعد بكرا",
        "بدي موعد اليوم", "بدي موعد", "✏️",
    )
    if any(p in norm for p in edit_phrases):
        return "edit_time"

    return None


def detect_rule_intent(
    user_message: str,
    *,
    rule_hint: str | None = None,
    phase: str = "CHATTING",
) -> str | None:
    """Rule-based intent hints merged with LLM output (operations stay rule-driven)."""
    if rule_hint:
        return rule_hint
    if phase == "CONFIRM":
        confirm_intent = detect_confirm_intent(user_message)
        if confirm_intent:
            return confirm_intent
    norm = normalize(user_message or "").lower()
    if not norm:
        return None
    cancel_tokens = ("الغاء", "إلغاء", "الغي", "إلغي", "كنسل", "cancel")
    if any(t in norm for t in cancel_tokens):
        return "cancel"
    if any(t in norm for t in ("حجز موعد", "موعد جديد", "ابدأ", "من جديد", "restart", "book")):
        return "new_booking"
    if "استعلام" in norm or "موعدي" in norm:
        return "inquiry"
    if "مواعيد" in norm and any(w in norm for w in ("موجود", "مسجل", "محجوز", "ضايل", "متبق", "باقي")):
        return "inquiry"
    if "موعد" in norm and any(w in norm for w in ("مسجل", "محجوز", "حجزي", "اخر", "آخر", "عندي", "وين")):
        return "inquiry"
    if "تواصل" in norm or "اتصل" in norm:
        return "contact"
    return None


async def run_booking_turn(
    user_message: str,
    phase: str,
    collected: dict,
    chat_history: list[dict],
    slot_context: dict | None = None,
    operation_context: dict | None = None,
    *,
    required_fields: list[str] | None = None,
    score_from_label=None,
) -> BookingTurnResult:
    """
    Run one LLM booking conversation turn.
    Returns reply + structured extraction for the orchestrator.
    """
    required = required_fields or ["name", "complaint", "urgency_score", "time_pref"]
    text = (user_message or "").strip()
    op_ctx = dict(operation_context or {})
    rule_intent = detect_rule_intent(text, rule_hint=op_ctx.get("rule_hint"), phase=phase)
    if rule_intent and "rule_hint" not in op_ctx:
        op_ctx["rule_hint"] = rule_intent

    if not gemini.is_ready:
        return fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )

    raw_json = await gemini.booking_turn(
        user_message=text,
        phase=phase,
        collected=collected,
        chat_history=chat_history,
        slot_context=slot_context,
        operation_context=op_ctx,
    )
    if not raw_json:
        return fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )

    try:
        parsed = _parse_json_response(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("booking_turn JSON parse failed: %s (raw=%s)", exc, raw_json[:200])
        return fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )

    reply = (parsed.get("reply") or "").strip()
    intent = (parsed.get("intent") or "continue").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "continue"
    # Prefer rule intent over a vague LLM "continue" at CONFIRM.
    if phase == "CONFIRM" and rule_intent in VALID_INTENTS:
        intent = rule_intent
    elif rule_intent in VALID_INTENTS and intent == "continue":
        intent = rule_intent

    # Modern/weaker models invent "تم الحجز" without intent=confirm — coerce or strip.
    if reply_claims_booking_done(reply):
        if phase == "CONFIRM":
            intent = "confirm"
        else:
            reply = ""

    off_topic = bool(parsed.get("off_topic")) or intent == "off_topic"
    # Never drop medical/urgency extraction on a false off_topic flag.
    force_extract = (
        _message_has_urgency_signal(text)
        or _message_looks_like_complaint(text)
        or bool(_validate_name(text))
    )
    if force_extract:
        off_topic = False

    extracted_raw = parsed.get("extracted") or {}
    if not isinstance(extracted_raw, dict):
        extracted_raw = {}

    extracted: dict[str, Any] = {}
    if not off_topic:
        if extracted_raw.get("name"):
            extracted["name"] = extracted_raw["name"]
        if extracted_raw.get("complaint"):
            extracted["complaint"] = extracted_raw["complaint"]
        if extracted_raw.get("urgency"):
            extracted["urgency"] = extracted_raw["urgency"]
        if extracted_raw.get("time_pref"):
            extracted["time_pref"] = extracted_raw["time_pref"]
        rule_merge = merge_rule_extracted(text, collected)
        for k, v in rule_merge.items():
            # Rule urgency/complaint beat weak LLM nulls; prefer rules when both exist for urgency labels.
            if k == "urgency_score" and _message_has_urgency_signal(text):
                extracted[k] = v
            else:
                extracted.setdefault(k, v)

    if off_topic and not reply:
        first = missing_required_fields(collected, required)
        q = FIELD_QUESTIONS_AR.get(first[0] if first else "name", FIELD_QUESTIONS_AR["name"])
        reply = f"{OFF_TOPIC_REPLY}\n\n{q}"

    if not reply:
        fb = fallback_reply(
            collected, required, text, rule_hint=op_ctx.get("rule_hint"), phase=phase
        )
        reply = fb.reply
        if intent == "continue" and fb.intent != "continue":
            intent = fb.intent
        for k, v in fb.extracted.items():
            extracted.setdefault(k, v)

    return BookingTurnResult(
        reply=reply,
        intent=intent,
        extracted=extracted,
        off_topic=off_topic,
    )
