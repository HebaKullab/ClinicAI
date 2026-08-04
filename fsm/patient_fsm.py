"""
fsm/patient_fsm.py
Patient appointment booking FSM:
  collect data → validate checklist → classify clinic → score priority
  → create/update patient file → check DB slots → confirm → reserve slot/book appointment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from nlp.extractor import extract_patient_fields
from nlp.normalizer import normalize
from scheduler.priority import score_and_classify
from scheduler.classifier import (
    auto_resolve_specialty,
    classify_specialty,
    classify_with_gemini_fallback,
    detect_unsupported_specialty,
    is_supported_specialty,
    SPECIALTY_NAMES_AR,
)
from nlp.gemini_client import gemini, OFF_TOPIC_REPLY, GeminiClient
from fsm.ui_actions import UIAction
from fsm.services import BookingServices

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(normalize(text).split())


def _matches_any_token(norm: str, words: set[str]) -> bool:
    """Match whole tokens or multi-word phrases — avoids 'ok' inside longer words."""
    tokens = _tokenize(norm)
    for word in words:
        w_norm = normalize(word)
        if " " in w_norm or len(w_norm) > 4:
            if w_norm in norm:
                return True
        elif w_norm in tokens:
            return True
    return False



class State(Enum):
    GREETING = auto()
    COLLECT_NAME = auto()
    COLLECT_COMPLAINT = auto()
    COLLECT_URGENCY = auto()
    COLLECT_TIME = auto()
    VALIDATE = auto()          # requirements-loop checkpoint
    COLLECT_SPECIALTY = auto() # used when classifier confidence is low
    OFFER_GP_FALLBACK = auto() # unsupported clinic → offer general practice
    CLASSIFY = auto()          # specialty + priority
    FIND_SLOT = auto()
    CONFIRM = auto()
    FINALIZED = auto()
    WAITLISTED = auto()
    CANCELLED = auto()


# Required fields — the checklist before touching scheduling.
REQUIRED_FIELDS: list[str] = ["name", "complaint", "urgency_score", "time_pref"]

FIELD_QUESTIONS_AR: dict[str, str] = {
    "name": "ما اسمك الكريم؟",
    "complaint": "شو الشكوى أو سبب الزيارة؟",
    "urgency_score": "هل الموضوع عاجل، متوسط، أو روتيني؟",
    "time_pref": "متى تحب الموعد؟ (اليوم، بكرا، الأسبوع الجاي...)",
}

CONFIRM_WORDS = {
    "نعم", "ايوه", "آيوه", "تمام", "ماشي", "اوك", "yes", "يلا", "احجز",
    "تاكيد", "تأكيد", "تاكيد الحجز", "تأكيد الحجز", "موافق", "✅",
    "اه", "آه", "ايه", "اي", "ايوا", "يب", "ok",
}
CANCEL_WORDS = {"لا", "الغي", "إلغي", "الغاء", "إلغاء", "بدي الغي", "مش حابب", "❌"}
EDIT_WORDS = {"تعديل", "تعديل الموعد", "✏️", "✏️ تعديل الموعد"}
NEXT_SLOT_WORDS = {"موعد آخر", "🔄", "🔄 موعد آخر"}


def _looks_like_slot_list_request(norm: str) -> bool:
    """Patient wants to browse remaining slot options at CONFIRM."""
    if not norm:
        return False
    browse = ("اشوف", "فرج", "فرجي", "فرجيم", "عرض", "ورج", "اعرض", "شوف", "شي", "بين")
    slot_words = ("موعد", "مواعيد", "خيار", "خيارات", "بديل", "ضايل", "ضايله", "ضايلة", "متبق", "موجود", "باقي")
    if any(w in norm for w in browse) and any(w in norm for w in slot_words):
        return True
    if "مواعيد" in norm and any(w in norm for w in ("ضايل", "ضايله", "ضايلة", "متبق", "موجود", "باق", "ثان")):
        return True
    if any(w in norm for w in ("خيارات", "بدائل")) and any(w in norm for w in browse + ("بدي", "بده", "اريد")):
        return True
    return False


def _looks_like_decline(norm: str) -> bool:
    """Patient wants to back out — «ما بدي اشي»، «لا شكراً»."""
    if not norm:
        return False
    if _matches_any_token(norm, CANCEL_WORDS):
        return True
    decline = (
        "ما بدي", "مش بدي", "ما بده", "لا بدي", "لا شكر", "مش حاب", "ما بدي اشي",
        "ما بدي شي", "ما بدي اي", "مو بدي", "مش عايز", "بطل", "خلص", "سكر",
        "مش interested", "no thanks",
    )
    return any(p in norm for p in decline)


def _looks_like_soft_confirm(norm: str) -> bool:
    """Informal Arabic affirmatives — «اه», «بدi موعد اه», emoji confirm buttons."""
    if not norm or _matches_any_token(norm, CANCEL_WORDS):
        return False
    if _looks_like_slot_list_request(norm) or _looks_like_decline(norm):
        return False
    if any(w in norm for w in ("اشوف", "فرج", "عرض", "شوف", "ورج", "ليش", "لماذا", "كيف")):
        return False
    if _matches_any_token(norm, CONFIRM_WORDS):
        return True
    tokens = _tokenize(norm)
    if tokens & {"اه", "آه", "ايه", "اي", "ايوا", "يب", "تمام", "ماشي", "اوك", "ok"}:
        return True
    if any(w in norm for w in ("بدي", "بده", "اريد", "حاب", "حابب")):
        if any(w in norm for w in ("احجز", "حجز", "نعم", "اه", "آه", "موافق", "تمام", "اكد", "أكد")):
            return True
        # «بدi موعد» alone is confirm intent; «بدi اشوف مواعيد» excluded above
        if "موعد" in norm and "اشوف" not in norm and "فرج" not in norm:
            return True
    return False


def _is_name_dispute(text: str) -> bool:
    norm = normalize(text or "")
    if not norm:
        return False
    markers = (
        "مين قال", "من قال", "مو اسمي", "مش اسمي", "اسمي مش", "اسمي مو",
        "مو منال", "غلط الاسم", "الاسم غلط", "اسم غلط", "not my name",
        "مش هاد اسمي", "مو هاد اسمي",
    )
    if any(m in norm for m in markers):
        return True
    return "اسمي" in norm and any(w in norm for w in ("مش", "مو", "ليس", "wrong", "غلط"))


def _is_low_signal_input(text: str) -> bool:
    """Dots, single letters, or empty pings — not meaningful booking input."""
    raw = (text or "").strip()
    if not raw or raw in {".", "…", "..", "..."}:
        return True
    return len(raw) <= 2 and not raw.isdigit()


def _is_bot_meta_question(text: str) -> bool:
    """«ماذا تريد» / «شو بدك» — not a name or complaint."""
    norm = normalize(text or "").strip()
    phrases = (
        "ماذا تريد", "ماذا تبغ", "شو تريد", "شو بدك", "شو بتحب",
        "what do you want", "what do you need",
    )
    return any(p in norm for p in phrases)


_GREETING_TOKENS = frozenset({
    "مرحبا", "مرحب", "هلا", "اهلا", "اهلين", "هلين", "هاي",
    "السلام", "سلام", "عليكم", "عليك", "وسهلا",
    "صباح", "مساء", "الخير", "start", "hi", "hello", "hey",
})
_GREETING_PHRASES = (
    "السلام عليكم", "صباح الخير", "مساء الخير",
    "اهلا وسهلا", "مرحبا بك", "hi there",
)

SPECIALTY_LABEL_TO_KEY = {
    # "قلب": "cardiology",
    # "اوعيه": "cardiology",
    # "أوعية": "cardiology",
    "اعصاب": "neurology",
    "أعصاب": "neurology",
    "عظام": "orthopedics",
    "مفاصل": "orthopedics",
    "نساء": "gynecology",
    "توليد": "gynecology",
    # "اطفال": "pediatrics",
    # "أطفال": "pediatrics",
    # "اسنان": "dentistry",
    # "أسنان": "dentistry",
    # "عيون": "ophthalmology",
    "جلدية": "dermatology",
    "جلديه": "dermatology",
    "هضمي": "gastroenterology",
    "مزمن": "chronic_diseases",
    "كبار": "elderly",
    "طب عام": "general_practice",
    "عام": "general_practice",
}


@dataclass
class PatientFSM:
    user_id: int
    services: BookingServices = field(default_factory=BookingServices.default)
    state: State = State.GREETING
    data: dict = field(default_factory=dict)
    slot: Optional[dict] = None              # Plain dict, not detached ORM object
    slot_options: list = field(default_factory=list)
    slot_index: int = 0
    priority: Optional[object] = None        # PriorityResult
    finalized_appointment_id: Optional[str] = None
    waitlist_position: Optional[int] = None

    # ── Public entry ──────────────────────────────────────────────────────────

    async def handle(self, text: str) -> tuple[str, UIAction, dict]:
        """Process one message, advance state, return Arabic reply + UI action."""
        text = text or ""
        norm = normalize(text)

        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED):
            if self._is_new_booking_request(text):
                self._reset()
            else:
                return self._reply("تم إنهاء الطلب السابق. إذا بدك حجز جديد اكتب: حجز موعد جديد 📅", UIAction.SHOW_MAIN_MENU)

        if _is_name_dispute(text):
            self.data.pop("name", None)
            self.state = State.COLLECT_NAME
            return self._reply("عذراً على اللبس! 🙂 شو اسمك الصحيح؟", None)

        if self._is_clarification_request(text):
            return await self._reply_with_context(text)

        if self.state != State.CONFIRM:
            question_reply = await self._maybe_answer_question(text)
            if question_reply is not None:
                return question_reply

        if self.state not in (State.COLLECT_URGENCY, State.CONFIRM, State.OFFER_GP_FALLBACK):
            await self._absorb(text)

        if self.state == State.VALIDATE:
            return await self._run_validate()

        if self.state == State.CLASSIFY:
            return await self._classify_and_schedule()

        if self.state == State.GREETING:
            self._preload_patient_name()
            if self.data.get("name") and not self._is_greeting_only(text) and not self._looks_like_name(text):
                if _is_bot_meta_question(text) or GeminiClient.looks_like_question(text):
                    stored = self.data.pop("name", None)
                    self.state = State.COLLECT_NAME
                    return self._reply(
                        "أنا مساعد حجز المواعيد في العيادة. "
                        + (f"سجلت اسمك سابقاً «{stored}» — إذا غلط اكتب اسمك الصحيح.\n" if stored else "")
                        + FIELD_QUESTIONS_AR["name"],
                        None,
                    )
            if self.data.get("name"):
                self.state = State.COLLECT_COMPLAINT
                return self._reply(
                    f"أهلاً {self.data['name']}! 👋\n" + FIELD_QUESTIONS_AR["complaint"],
                    None,
                )
            if self._looks_like_name(text):
                self.data["name"] = text.strip()
                self.state = State.COLLECT_COMPLAINT
                return self._reply(
                    f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"],
                    None,
                )
            self.state = State.COLLECT_NAME
            if self._is_greeting_only(text):
                return self._reply(self._greeting_and_ask_name(), None)
            return self._reply(FIELD_QUESTIONS_AR["name"], None)

        if self.state == State.COLLECT_NAME:
            if self._is_greeting_only(text):
                return self._reply(self._greeting_and_ask_name(), None)

            if self.data.get("name") and self._is_greeting_only(self.data["name"]):
                self.data.pop("name", None)

            if self.data.get("name"):
                self.state = State.COLLECT_COMPLAINT
                return self._reply(f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"], None)

            if self._looks_like_name(text):
                self.data["name"] = text.strip()
                self.state = State.COLLECT_COMPLAINT
                return self._reply(f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"], None)

            raw = text.strip()
            if raw and any(ch.isdigit() for ch in raw):
                return self._reply(
                    "الاسم ما بكون أرقام 🙂 " + FIELD_QUESTIONS_AR["name"],
                    None,
                )

            return self._reply(FIELD_QUESTIONS_AR["name"], None)

        if self.state == State.COLLECT_COMPLAINT:
            unsupported = self.services.detect_unsupported(text)
            if unsupported:
                self._stash_complaint_for_unsupported(text)
                self.data["unsupported_clinic_label"] = unsupported
                self.state = State.OFFER_GP_FALLBACK
                return self._reply(self._gp_fallback_message(unsupported), UIAction.SHOW_CONFIRM)

            if self.data.get("complaint"):
                unsupported = detect_unsupported_specialty(self.data["complaint"].get("raw", ""))
                if unsupported:
                    self._stash_complaint_for_unsupported(text)
                    self.data["unsupported_clinic_label"] = unsupported
                    self.state = State.OFFER_GP_FALLBACK
                    return self._reply(self._gp_fallback_message(unsupported), UIAction.SHOW_CONFIRM)

                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], UIAction.SHOW_URGENCY)

            complaint = await self._extract_complaint_from_text(text)
            if complaint:
                self.data["complaint"] = complaint
                unsupported = self.services.detect_unsupported(complaint.get("raw", ""))
                if unsupported:
                    self._stash_complaint_for_unsupported(text)
                    self.data["unsupported_clinic_label"] = unsupported
                    self.state = State.OFFER_GP_FALLBACK
                    return self._reply(self._gp_fallback_message(unsupported), UIAction.SHOW_CONFIRM)

                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], UIAction.SHOW_URGENCY)

            if text.strip():
                self.data["complaint"] = {
                    "raw": text.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }
                unsupported = self.services.detect_unsupported(text.strip())
                if unsupported:
                    self._stash_complaint_for_unsupported(text)
                    self.data["unsupported_clinic_label"] = unsupported
                    self.state = State.OFFER_GP_FALLBACK
                    return self._reply(self._gp_fallback_message(unsupported), UIAction.SHOW_CONFIRM)

                self.state = State.COLLECT_URGENCY
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], UIAction.SHOW_URGENCY)

            return self._reply("ممكن تخبرني أكثر عن سبب زيارتك؟")

        if self.state == State.COLLECT_URGENCY:
            if not text.strip():
                return self._reply(FIELD_QUESTIONS_AR["urgency_score"], UIAction.SHOW_URGENCY)
            # Clear incidental NLP defaults; accept only explicit urgency in this message.
            self.data.pop("urgency_score", None)
            if self._absorb_urgency(norm, text):
                self.state = State.COLLECT_TIME
                return self._reply(FIELD_QUESTIONS_AR["time_pref"], UIAction.SHOW_TIME)
            return self._reply(
                "ما فهمت مستوى الأولوية. اكتب عاجل/روتيني/متوسط أو اختار من الأزرار.",
                UIAction.SHOW_URGENCY,
            )

        if self.state == State.OFFER_GP_FALLBACK:
            if _looks_like_soft_confirm(norm):
                self.data["specialty_hint"] = "general_practice"
                self.data["specialty_ar"] = SPECIALTY_NAMES_AR["general_practice"]
                self.data["specialty_method"] = "gp_fallback"
                self.data.pop("unsupported_clinic_label", None)
                if self._missing_fields():
                    return await self._run_validate()
                return await self._score_and_find_slot()
            if _matches_any_token(norm, CANCEL_WORDS):
                self.state = State.CANCELLED
                return self._reply("تم الإلغاء. إذا احتجت أي شيء، أنا هون. 👋", UIAction.NONE)
            label = self.data.get("unsupported_clinic_label", "هذا التخصص")
            return self._reply(self._gp_fallback_message(label), UIAction.SHOW_CONFIRM)

        if self.state == State.COLLECT_TIME:
            if not text.strip():
                return self._reply(FIELD_QUESTIONS_AR["time_pref"], UIAction.SHOW_TIME)
            mapped = self._parse_time_label(text)
            if not mapped and gemini.is_ready:
                extracted = await gemini.extract_missing_field(text, "time_pref")
                if extracted:
                    mapped = self._parse_time_label(extracted) or {"date": None, "phrase": extracted.strip()}
            if mapped:
                self.data["time_pref"] = mapped
                return await self._run_validate()
            return self._reply(
                "ما فهمت متى بدك الموعد. اكتب مثلاً: اليوم، بكرا، الأسبوع الجاي — أو اختار من الأزرار.",
                UIAction.SHOW_TIME,
            )

        if self.state == State.COLLECT_SPECIALTY:
            unsupported = detect_unsupported_specialty(text)
            if unsupported:
                self._stash_complaint_for_unsupported(text)
                self.data["unsupported_clinic_label"] = unsupported
                self.state = State.OFFER_GP_FALLBACK
                return self._reply(self._gp_fallback_message(unsupported), UIAction.SHOW_CONFIRM)

            specialty_key = self._parse_specialty_label(text)
            if not specialty_key and gemini.is_ready:
                extracted = await gemini.extract_missing_field(text, "specialty")
                if extracted and extracted in SPECIALTY_NAMES_AR:
                    specialty_key = extracted
            if not specialty_key:
                return self._reply(
                    "ما فهمت التخصص. اكتب اسم التخصص أو اختار من الأزرار.",
                    UIAction.SHOW_SPECIALTY,
                )
            self.data["specialty_hint"] = specialty_key
            self.data["specialty_ar"] = SPECIALTY_NAMES_AR.get(specialty_key, specialty_key)
            self.data["specialty_confirmed_by_patient"] = True
            return await self._score_and_find_slot()

        if self.state == State.CONFIRM:
            return await self._handle_confirm(text)

        if self.state == State.FIND_SLOT:
            return await self._find_slot()

        return self._reply("عفواً، ما فهمت. ممكن تعيد؟")

    async def handle_callback(self, data: str) -> tuple[str, UIAction, dict]:
        # Kept for compatibility if inline keyboards are added later.
        if data.startswith("urgency:"):
            level = data.split(":", 1)[1]
            self._set_urgency_from_label(level)
            self.state = State.COLLECT_TIME
            return self._reply(FIELD_QUESTIONS_AR["time_pref"], UIAction.SHOW_TIME)

        if data.startswith("time:"):
            self.data["time_pref"] = self._map_time_selection(data.split(":", 1)[1])
            return await self._run_validate()

        if data.startswith("spec:"):
            specialty_key = data.split(":", 1)[1]
            if specialty_key in SPECIALTY_NAMES_AR:
                self.data["specialty_hint"] = specialty_key
                self.data["specialty_ar"] = SPECIALTY_NAMES_AR[specialty_key]
                self.data["specialty_confirmed_by_patient"] = True
                return await self._score_and_find_slot()

        if data.startswith("confirm:"):
            return await self._handle_confirm("نعم" if data.endswith("yes") else "لا")

        return self._reply("عفواً، لم أفهم اختيارك. حاول مرة أخرى.")

    def _reply(self, text: str, action: UIAction = UIAction.NONE, payload: dict | None = None) -> tuple[str, UIAction, dict]:
        return text, action, payload or {}

    # ── Extraction / validation ───────────────────────────────────────────────

    async def _absorb(self, text: str):
        """Merge newly extracted fields and optionally enrich with AI."""
        if self._is_greeting_only(text):
            return
        if GeminiClient.looks_like_question(text):
            return

        extracted = extract_patient_fields(text)
        for k, v in extracted.items():
            if v is None or self.data.get(k):
                continue
            if k == "name" and self.state not in (State.GREETING, State.COLLECT_NAME):
                continue
            if k == "urgency_score" and self.state != State.COLLECT_URGENCY:
                continue
            if k == "time_pref" and self.state != State.COLLECT_TIME:
                continue
            if k == "time_pref" and isinstance(v, dict) and not (v.get("date") or v.get("phrase")):
                continue
            if k == "complaint" and self.state not in (State.GREETING, State.COLLECT_NAME, State.COLLECT_COMPLAINT):
                continue
            self.data[k] = v

        if text.strip() and self._should_try_ai_extraction():
            await self._try_ai_extraction(text)

    def _should_try_ai_extraction(self) -> bool:
        if not gemini.is_ready:
            return False
        if self.state == State.COLLECT_NAME and not self.data.get("name"):
            return True
        if self.state == State.COLLECT_COMPLAINT and not self.data.get("complaint"):
            return True
        if self.state == State.COLLECT_URGENCY and self.data.get("urgency_score") is None:
            return True
        if self.state == State.COLLECT_TIME and not self.data.get("time_pref"):
            return True
        return False

    async def _extract_complaint_from_text(self, original_text: str) -> dict | None:
        complaint = extract_patient_fields(original_text).get("complaint")
        if complaint:
            return complaint

        if gemini.is_ready:
            complaint_text = await gemini.extract_missing_field(original_text, "complaint")
            if complaint_text:
                return {
                    "raw": complaint_text.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }
        return None

    def _missing_fields(self) -> list[str]:
        missing = []
        for field_name in REQUIRED_FIELDS:
            val = self.data.get(field_name)
            if val is None:
                missing.append(field_name)
            elif field_name == "time_pref" and isinstance(val, dict) and not (val.get("date") or val.get("phrase")):
                missing.append(field_name)
            elif field_name == "complaint" and not val:
                missing.append(field_name)
        return missing

    async def _run_validate(self) -> tuple[str, object | None]:
        """Checklist loop: never schedule until all required data is available."""
        self.state = State.VALIDATE
        missing = self._missing_fields()
        if missing:
            first_missing = missing[0]
            if first_missing == "name" and self.data.get("name"):
                missing = [f for f in missing if f != "name"]
                first_missing = missing[0] if missing else None
            if not first_missing:
                return await self._classify_and_schedule()
            self.state = {
                "name": State.COLLECT_NAME,
                "complaint": State.COLLECT_COMPLAINT,
                "urgency_score": State.COLLECT_URGENCY,
                "time_pref": State.COLLECT_TIME,
            }[first_missing]
            action = (
                UIAction.SHOW_URGENCY
                if first_missing == "urgency_score"
                else UIAction.SHOW_TIME
                if first_missing == "time_pref"
                else UIAction.NONE
            )
            return self._reply(
                f"بعدنا محتاجين معلومة واحدة 📋\n{FIELD_QUESTIONS_AR[first_missing]}",
                action,
            )

        return await self._classify_and_schedule()

    # ── Classify + priority + schedule ────────────────────────────────────────

    async def _classify_and_schedule(self) -> tuple[str, UIAction, dict]:
        self.state = State.CLASSIFY

        if self.data.get("specialty_method") == "gp_fallback":
            self.data["specialty_hint"] = "general_practice"
            self.data["specialty_ar"] = SPECIALTY_NAMES_AR["general_practice"]
            return await self._score_and_find_slot()

        norm_complaint = normalize(self.data.get("complaint", {}).get("raw", ""))
        unsupported = self.services.detect_unsupported(norm_complaint)
        if unsupported:
            self.data["unsupported_clinic_label"] = unsupported
            self.state = State.OFFER_GP_FALLBACK
            return self._reply(self._gp_fallback_message(unsupported), UIAction.SHOW_CONFIRM)

        gemini_client = self.services.gemini
        if gemini_client is not None and getattr(gemini_client, "is_ready", False):
            spec_result = await self.services.classify_with_fallback(norm_complaint, gemini_client)
        else:
            spec_result = auto_resolve_specialty(self.services.classify(norm_complaint))

        if not is_supported_specialty(spec_result.get("specialty")):
            label = spec_result.get("specialty_ar") or spec_result.get("specialty") or "هذا التخصص"
            self.data["unsupported_clinic_label"] = label
            self.state = State.OFFER_GP_FALLBACK
            return self._reply(self._gp_fallback_message(label), UIAction.SHOW_CONFIRM)

        self.data["specialty_hint"] = spec_result["specialty"]
        self.data["specialty_ar"] = spec_result["specialty_ar"]
        self.data["specialty_method"] = spec_result.get("method")
        self.data["specialty_confidence"] = spec_result.get("confidence")

        return await self._score_and_find_slot()

    async def _score_and_find_slot(self) -> tuple[str, UIAction, dict]:
        self.priority = self.services.score(self.data)
        self.data["priority_class"] = self.priority.priority_class
        self.data["priority_score"] = self.priority.score
        self.data["priority_breakdown"] = self.priority.breakdown

        self.state = State.FIND_SLOT
        return await self._find_slot()

    async def _find_slot(self) -> tuple[str, UIAction, dict]:
        from database.db import get_db

        self.slot_options = []
        self.slot_index = 0
        self.slot = None

        with get_db() as db:
            slots = self.services.find_slots(
                db,
                specialty=self.data.get("specialty_hint", "general_practice"),
                priority_class=self.priority.priority_class,
                preferred_date=self.data.get("time_pref", {}).get("date"),
                telegram_id=self.user_id,
                limit=3,
            )
            if slots:
                self.slot_options = [self._slot_to_dict(slot) for slot in slots]
                self.slot_index = 0
                self.slot = self.slot_options[0]

        if not self.slot:
            await self._save_waitlist()
            self.state = State.WAITLISTED
            return self._reply(
                "عفواً، ما في مواعيد متاحة حالياً في هذا الاختصاص. 😔\n"
                "تم حفظ ملفك وإضافتك لقائمة الانتظار، وسنتواصل معك بأقرب وقت.",
                UIAction.NONE,
            )

        self.state = State.CONFIRM
        return self._format_confirm_message()

    async def _handle_confirm(self, text: str) -> tuple[str, UIAction, dict]:
        norm = normalize(text)
        await self._reload_slot_options_if_needed()

        if _looks_like_decline(norm):
            self.state = State.CANCELLED
            return self._reply("تمام، ما في مشكلة — ألغيت الحجز. إذا احتجت شي لاحقاً أنا هون. 👋", UIAction.NONE)

        if _looks_like_soft_confirm(norm):
            result = await self._finalize()
            if result.get("slot_conflict"):
                self.slot = None
                self.state = State.FIND_SLOT
                prefix = "للأسف الموعد انحجز قبل التأكيد بثواني. رح أبحث لك عن أقرب موعد بديل الآن.\n\n"
                reply, action, payload = await self._find_slot()
                return self._reply(prefix + reply, action, payload)

            if result.get("booking_conflict"):
                conflict = result["booking_conflict"]
                existing = conflict.get("appointment")
                if isinstance(existing, dict):
                    existing_dt = existing.get("appt_datetime")
                    when = existing_dt.strftime("%A، %d/%m/%Y — %H:%M") if existing_dt else "موعد سابق"
                    specialty = existing.get("specialty_ar") or existing.get("specialty") or "نفس التخصص"
                else:
                    when = existing.appt_datetime.strftime("%A، %d/%m/%Y — %H:%M") if existing and existing.appt_datetime else "موعد سابق"
                    specialty = (existing.specialty_ar or existing.specialty or "نفس التخصص") if existing else "نفس التخصص"
                self.state = State.FINALIZED
                if conflict.get("type") == "time_overlap":
                    return self._reply(
                        "ما بقدر أثبت هذا الموعد لأن عندك موعد آخر بنفس الوقت أو وقت متداخل. ⏰\n"
                        f"موعدك الحالي: {when} — {specialty}.\n"
                        "ممكن تحجز موعدًا بتخصص مختلف في نفس اليوم بشرط يكون بوقت آخر غير متداخل.",
                        UIAction.NONE,
                    )
                return self._reply(
                    "عندك موعد فعال مسبقًا لنفس التخصص في نفس اليوم، لذلك ما حجزت موعدًا ثانيًا. ✅\n"
                    f"موعدك الحالي: {when} — {specialty}.\n"
                    "لو بدك تشوف تخصصًا مختلفًا، ممكن تحجز موعدًا آخر بوقت غير متداخل.",
                    UIAction.NONE,
                )

            if not result.get("appointment"):
                self.state = State.WAITLISTED
                return self._reply(
                    "تم حفظ ملفك، لكن لم أستطع تثبيت الموعد حالياً. أضفتك لقائمة الانتظار وسنتواصل معك. 🌿",
                    UIAction.NONE,
                )

            appt = result["appointment"]
            self.finalized_appointment_id = appt.appt_id
            self.state = State.FINALIZED
            return self._reply(
                "✅ تم تأكيد حجزك وحفظ ملفك في النظام!\n"
                f"رقم الحجز: {appt.appt_id}\n"
                f"📆 {appt.appt_datetime.strftime('%A، %d/%m/%Y — %H:%M')}\n"
                f"👨‍⚕️ الطبيب: {self.slot.get('doctor_name') or '—'}\n"
                f"🏢 العيادة: {self.slot.get('clinic_name') or '—'}\n"
                "سيظهر الموعد تلقائياً في لوحة التحكم كموعد محجوز. نتمنى لك الشفاء 🌿",
                UIAction.NONE,
            )

        if _is_low_signal_input(text):
            return self._reply(self._confirm_nudge(), UIAction.SHOW_CONFIRM)

        if GeminiClient.looks_like_question(norm):
            return self._reply(self._confirm_explain_question(text), UIAction.SHOW_CONFIRM)

        if _matches_any_token(norm, EDIT_WORDS):
            self.state = State.COLLECT_TIME
            return self._reply(
                "تمام، متى تحب الموعد؟\n" + FIELD_QUESTIONS_AR["time_pref"],
                UIAction.SHOW_TIME,
            )

        if _matches_any_token(norm, NEXT_SLOT_WORDS) or _looks_like_slot_list_request(norm):
            if _looks_like_slot_list_request(norm):
                return self._reply(self._format_slot_options_list(), UIAction.SHOW_CONFIRM)
            if len(self.slot_options) > 1:
                self.slot_index = (self.slot_index + 1) % len(self.slot_options)
                self.slot = self.slot_options[self.slot_index]
                extra = f"\n\n(خيار {self.slot_index + 1} من {len(self.slot_options)})"
                reply, action, payload = self._format_confirm_message()
                return self._reply(reply + extra, action, payload)
            return self._reply("هذا هو أقرب موعد متاح حالياً.", UIAction.SHOW_CONFIRM)

        if self.slot_options and norm.strip().isdigit():
            pick = int(norm.strip()) - 1
            if 0 <= pick < len(self.slot_options):
                self.slot_index = pick
                self.slot = self.slot_options[pick]
                extra = f"\n\n(اخترت خيار {pick + 1} من {len(self.slot_options)})"
                reply, action, payload = self._format_confirm_message()
                return self._reply(reply + extra, action, payload)


        if _matches_any_token(norm, CANCEL_WORDS):
            self.state = State.CANCELLED
            return self._reply("تم الإلغاء. إذا احتجت أي شيء، أنا هون. 👋", UIAction.NONE)

        return await self._confirm_natural_reply(text)

    def _confirm_nudge(self) -> str:
        return "لسا معك بالحجز 🙂 بدك تأكيد الموعد، تشوف وقت ثاني، أو نلغي؟"

    def _confirm_explain_question(self, text: str) -> str:
        """Rule-based natural answers at CONFIRM — no emoji instruction spam."""
        norm = normalize(text or "")
        if any(w in norm for w in ("ليش", "لماذا", "ليه", "why")):
            when = "—"
            if self.slot and self.slot.get("slot_datetime"):
                when = self.slot["slot_datetime"].strftime("%A %d/%m — %H:%M")
            return (
                f"اقترحنا موعد {when}. قبل ما نثبته بالنظام بدنا موافقتك — "
                "إذا مناسب قولي «تمام» أو اضغط ✅، وإذا لا في مش مانع."
            )
        if any(w in norm for w in ("كيف", "شو", "ماذا", "ايش")):
            return (
                "هاي آخر خطوة: إما تأكيد الموعد المعروض، أو تختار وقت ثاني، أو إلغاء. "
                "استخدم الأزرار تحت إذا أسهل عليك."
            )
        return self._confirm_nudge()

    async def _confirm_natural_reply(self, text: str) -> tuple[str, UIAction, dict]:
        """Gemini fallback only when rules did not already answer."""
        if gemini.is_ready:
            answer = await gemini.answer_in_booking_context(
                text,
                self.state.name,
                {
                    **self.data,
                    "slot": self.slot,
                    "slot_options": self.slot_options,
                    "slot_count": len(self.slot_options),
                },
                self._confirm_gemini_hint(),
            )
            if answer and "✅ لتأكيد" not in answer:
                return self._reply(answer.strip(), UIAction.SHOW_CONFIRM)
        return self._reply(self._confirm_nudge(), UIAction.SHOW_CONFIRM)

    def _confirm_gemini_hint(self) -> str:
        when = "—"
        if self.slot and self.slot.get("slot_datetime"):
            when = self.slot["slot_datetime"].strftime("%A %d/%m — %H:%M")
        return (
            f"المريض بمرحلة تأكيد موعد مقترح ({when}). "
            "ردّي بلهجة فلسطينية طبيعية بجملة أو جملتين. "
            "لا تذكري قائمة أزرار ولا تكرري «اكتب ✅»."
        )

    async def _reload_slot_options_if_needed(self) -> None:
        """Restore slot options after DB reload — common when FSM snapshot was incomplete."""
        if self.slot_options:
            return
        if self.slot:
            self.slot_options = [self.slot]
            return
        if not self.priority:
            return
        from database.db import get_db

        with get_db() as db:
            slots = self.services.find_slots(
                db,
                specialty=self.data.get("specialty_hint", "general_practice"),
                priority_class=self.priority.priority_class,
                preferred_date=self.data.get("time_pref", {}).get("date"),
                telegram_id=self.user_id,
                limit=3,
            )
            if slots:
                self.slot_options = [self._slot_to_dict(s) for s in slots]
                self.slot_index = min(self.slot_index, len(self.slot_options) - 1)
                self.slot = self.slot_options[self.slot_index]

    async def _finalize(self) -> dict:
        from database.db import get_db

        with get_db() as db:
            return self.services.book(
                db=db,
                telegram_id=self.user_id,
                data=self.data,
                slot_id=self.slot["slot_id"] if self.slot else None,
            )

    async def _save_waitlist(self) -> None:
        from database.db import get_db
        from database import crud

        with get_db() as db:
            entry = self.services.enqueue_waitlist(
                db,
                specialty=self.data.get("specialty_hint", "general_practice"),
                priority_class=self.data.get("priority_class", "P3"),
                priority_score=float(self.data.get("priority_score", 0.3)),
                urgency_score=float(self.data.get("urgency_score", 0.3)),
                telegram_id=self.user_id,
            )
            self.waitlist_position = entry.position
            crud.create_waitlist_appointment(db, self.user_id, self.data)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _stash_complaint_for_unsupported(self, text: str) -> None:
        """Keep the patient's full message as complaint when routing to GP fallback."""
        raw = (text or "").strip()
        if not raw:
            return
        existing = self.data.get("complaint") or {}
        self.data["complaint"] = {
            "raw": raw,
            "category": existing.get("category", "general"),
            "urgency_score": existing.get("urgency_score", 0.3),
            "specialty": "general_practice",
        }

    def _absorb_urgency(self, norm: str, raw_text: str = "") -> bool:
        combined = f"{norm} {normalize(raw_text)}"
        score = self._score_from_label(combined)
        if score is not None:
            self.data["urgency_score"] = score
            return True

        if any(w in combined for w in ["عاجل", "طارئ", "فوري", "خطير", "🔴"]):
            self.data["urgency_score"] = max(float(self.data.get("urgency_score", 0)), 0.85)
            return True
        if any(w in combined for w in ["روتيني", "مش عاجل", "🟢"]):
            self.data["urgency_score"] = min(float(self.data.get("urgency_score", 0.5)), 0.25)
            return True
        if any(w in combined for w in ["متوسط", "خلال أسبوع", "خلال اسبوع", "🟡", "عادي"]):
            self.data["urgency_score"] = 0.5
            return True
        if any(w in combined for w in ["اي وقت", "أي وقت"]) and self.state == State.COLLECT_URGENCY:
            self.data["urgency_score"] = 0.25
            return True

        return False

    def _score_from_label(self, label: str | None) -> float | None:
        if not label:
            return None
        label = label.lower()
        if any(w in label for w in ["عاجل", "طارئ", "فوري", "خطر"]):
            return 0.9
        if any(w in label for w in ["متوسط", "خلال أسبوع", "خلال اسبوع", "عادي"]):
            return 0.5
        if any(w in label for w in ["روتيني", "مش عاجل", "أي وقت", "اي وقت"]):
            return 0.2
        return None

    def _set_urgency_from_label(self, label: str):
        if label == "P1":
            score = 0.9
        elif label == "P2":
            score = 0.5
        elif label == "P3":
            score = 0.2
        else:
            score = self._score_from_label(label)
        if score is not None:
            self.data["urgency_score"] = score

    def _map_time_selection(self, selection: str) -> dict:
        from datetime import date, timedelta

        today = date.today()
        if selection == "today":
            return {"date": str(today), "phrase": "اليوم"}
        if selection == "tomorrow":
            return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
        if selection == "day_after":
            return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
        if selection == "next_week":
            return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
        return {"date": None, "phrase": "أي وقت متاح"}

    def _parse_time_label(self, text: str) -> dict | None:
        if not text:
            return None
        t = text.lower()
        from datetime import date, timedelta

        today = date.today()
        if "بعد بكرا" in t or "بعد غد" in t:
            return {"date": str(today + timedelta(days=2)), "phrase": "بعد بكرا"}
        if "اليوم" in t:
            return {"date": str(today), "phrase": "اليوم"}
        if "بكرا" in t or "غدا" in t or "غداً" in t:
            return {"date": str(today + timedelta(days=1)), "phrase": "بكرا"}
        if "أسبوع" in t or "اسبوع" in t or "الأسبوع" in t:
            return {"date": str(today + timedelta(days=7)), "phrase": "الأسبوع الجاي"}
        if "أي وقت" in t or "اي وقت" in t or "لا يهم" in t or "أي وقت متاح" in t:
            return {"date": None, "phrase": "أي وقت متاح"}
        return None

    def _parse_specialty_label(self, text: str) -> str | None:
        norm = normalize(text or "")
        for label_part, key in SPECIALTY_LABEL_TO_KEY.items():
            if normalize(label_part) in norm:
                return key
        for key, label_ar in SPECIALTY_NAMES_AR.items():
            if normalize(label_ar) in norm or norm == normalize(key):
                return key
        return None

    async def _try_ai_extraction(self, text: str):
        if not gemini.is_ready or self._is_greeting_only(text):
            return

        if not self.data.get("name") and self.state in (State.GREETING, State.COLLECT_NAME):
            name = await gemini.extract_missing_field(text, "name")
            if name and not self._is_greeting_only(name):
                self.data["name"] = name.strip()

        if not self.data.get("complaint") and self.state in (State.GREETING, State.COLLECT_NAME, State.COLLECT_COMPLAINT):
            complaint = await gemini.extract_missing_field(text, "complaint")
            if complaint:
                self.data["complaint"] = {
                    "raw": complaint.strip(),
                    "category": "general",
                    "urgency_score": 0.3,
                    "specialty": "general_practice",
                }

        if not self.data.get("urgency_score") and self.state == State.COLLECT_URGENCY:
            urgency = await gemini.extract_missing_field(text, "urgency")
            score = self._score_from_label(urgency)
            if score is not None:
                self.data["urgency_score"] = score

        if not self.data.get("time_pref") and self.state == State.COLLECT_TIME:
            time_pref = await gemini.extract_missing_field(text, "time_pref")
            if time_pref:
                self.data["time_pref"] = {"date": None, "phrase": time_pref.strip()}

    async def _reply_with_context(self, text: str) -> tuple[str, UIAction, dict]:
        if self.data and gemini.is_ready:
            prompt = (
                f"المستخدم قال: {text}\n"
                f"الحالة الحالية: {self.state.name}\n"
                "اكتب رداً عربيًا فلسطينيًا مختصرًا (جملة أو جملتين) يوضح أنك فهمته وتكمّل الحجز."
            )
            try:
                reply = await gemini.ask(prompt, max_tokens=80)
                if reply:
                    return self._reply(reply)
            except Exception as exc:
                logger.warning("Gemini clarification reply failed: %s", exc)

        if self.data.get("name"):
            return self._reply(f"تمام {self.data['name']}، خلينا نكمّل الحجز.")
        return self._reply("تمام، خلينا نكمّل الحجز خطوة بخطوة.")

    _UNCLEAR_REPLY_MARKERS = (
        "عفواً، ما فهمت",
        "ما فهمت مستوى الأولوية",
        "ما فهمت متى بدك الموعد",
        "ما فهمت التخصص",
        "ممكن تخبرني أكثر",
    )

    def rule_reply_seems_inadequate(self, text: str, reply: str) -> bool:
        """True when the rule-based reply likely missed user intent — prefer Gemini."""
        if self.state in (State.CONFIRM, State.OFFER_GP_FALLBACK):
            return False

        raw = (text or "").strip()
        if not raw:
            return False

        if self.state in (State.GREETING, State.COLLECT_NAME) and self._looks_like_name(raw):
            return False
        if (
            self.state == State.COLLECT_COMPLAINT
            and self.data.get("name")
            and len(raw) > 2
            and not self._is_greeting_only(raw)
            and not self._is_greeting_only(str(self.data.get("name", "")))
        ):
            if not any(marker in reply for marker in self._UNCLEAR_REPLY_MARKERS):
                return False

        if any(marker in reply for marker in self._UNCLEAR_REPLY_MARKERS):
            return True

        stored_name = self.data.get("name")
        if isinstance(stored_name, str) and self._is_greeting_only(stored_name):
            return True
        if self._is_greeting_only(raw):
            norm_raw = normalize(raw)
            if stored_name and normalize(str(stored_name)) == norm_raw:
                return True
            if norm_raw and norm_raw in normalize(reply):
                return True

        field_q = self._current_field_question().strip()
        reply_stripped = reply.strip()
        if GeminiClient.looks_like_question(raw) or GeminiClient.looks_off_topic(raw):
            if reply_stripped == field_q or reply_stripped.endswith(field_q):
                return True

        if (
            len(raw) > 10
            and self.state
            in (State.COLLECT_NAME, State.COLLECT_COMPLAINT, State.COLLECT_TIME, State.COLLECT_SPECIALTY)
            and (reply_stripped == field_q or reply_stripped.endswith(field_q))
        ):
            return True

        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED):
            if "تم إنهاء الطلب السابق" in reply and not self._is_new_booking_request(raw):
                if len(raw) > 4 and not self._is_greeting_only(raw):
                    return True

        return False

    def _undo_obvious_rule_mistakes(self, text: str) -> None:
        """Revert state/data when rules clearly mis-read the last message."""
        raw = (text or "").strip()
        name = self.data.get("name")
        if isinstance(name, str) and self._is_greeting_only(name):
            self.data.pop("name", None)
            if self.state == State.COLLECT_COMPLAINT:
                self.state = State.COLLECT_NAME
        if self._is_greeting_only(raw) and isinstance(name, str) and normalize(name) == normalize(raw):
            self.data.pop("name", None)
            if self.state == State.COLLECT_COMPLAINT:
                self.state = State.COLLECT_NAME

    async def maybe_gemini_fallback(self, text: str) -> str | None:
        if not gemini.is_ready:
            return None

        if self.state in (State.GREETING, State.COLLECT_NAME) and self._looks_like_name(text):
            return None

        self._undo_obvious_rule_mistakes(text)

        if self.state not in (
            State.GREETING,
            State.COLLECT_NAME,
            State.COLLECT_COMPLAINT,
            State.COLLECT_URGENCY,
            State.COLLECT_TIME,
            State.COLLECT_SPECIALTY,
            State.FINALIZED,
            State.CANCELLED,
            State.WAITLISTED,
        ):
            return None

        current_q = self._current_field_question()
        if GeminiClient.looks_off_topic(text):
            return f"{OFF_TOPIC_REPLY}\n\n{current_q}"
        if GeminiClient.looks_like_question(text):
            return await gemini.answer_in_booking_context(
                text,
                self.state.name,
                self.data,
                current_q,
            )
        try:
            return await gemini.build_response(
                self.state.name,
                {**self.data, "last_user_message": text, "current_question": current_q},
            )
        except Exception as exc:
            logger.warning("Gemini fallback reply failed: %s", exc)
            return None

    def _current_field_question(self) -> str:
        mapping = {
            State.GREETING: FIELD_QUESTIONS_AR["name"],
            State.COLLECT_NAME: FIELD_QUESTIONS_AR["name"],
            State.COLLECT_COMPLAINT: FIELD_QUESTIONS_AR["complaint"],
            State.COLLECT_URGENCY: FIELD_QUESTIONS_AR["urgency_score"],
            State.COLLECT_TIME: FIELD_QUESTIONS_AR["time_pref"],
            State.COLLECT_SPECIALTY: "اختار أقرب تخصص للشكوى.",
            State.OFFER_GP_FALLBACK: "موافق/لا على الطب العام؟",
            State.CONFIRM: "هل بتحب تأكيد الموعد المعروض؟",
            State.FINALIZED: "إذا بدك حجز جديد اكتب: حجز موعد جديد 📅",
            State.CANCELLED: "إذا بدك حجز جديد اكتب: حجز موعد جديد 📅",
            State.WAITLISTED: "إذا بدك حجز جديد اكتب: حجز موعد جديد 📅",
        }
        return mapping.get(self.state, "كيف بقدر أساعدك بالحجز؟")

    def _current_ui_action(self) -> UIAction | None:
        mapping = {
            State.COLLECT_URGENCY: UIAction.SHOW_URGENCY,
            State.COLLECT_TIME: UIAction.SHOW_TIME,
            State.COLLECT_SPECIALTY: UIAction.SHOW_SPECIALTY,
            State.OFFER_GP_FALLBACK: UIAction.SHOW_CONFIRM,
            State.CONFIRM: UIAction.SHOW_CONFIRM,
        }
        return mapping.get(self.state)

    async def _maybe_answer_question(self, text: str) -> tuple[str, UIAction, dict] | None:
        if not text.strip():
            return None
        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED, State.CLASSIFY, State.FIND_SLOT):
            return None
        if not GeminiClient.looks_like_question(text) and not GeminiClient.looks_off_topic(text):
            return None

        current_q = self._current_field_question()
        if GeminiClient.looks_off_topic(text):
            return self._reply(f"{OFF_TOPIC_REPLY}\n\n{current_q}", self._current_ui_action())

        if gemini.is_ready:
            ctx_data = dict(self.data)
            if self.state == State.CONFIRM and self.slot:
                ctx_data["slot"] = self.slot
            answer = await gemini.answer_in_booking_context(
                text,
                self.state.name,
                ctx_data,
                current_q,
            )
            if answer:
                return self._reply(answer.strip(), self._current_ui_action())

        if GeminiClient.looks_off_topic(text):
            return self._reply(f"{OFF_TOPIC_REPLY}\n\n{current_q}", self._current_ui_action())
        return None

    def _looks_like_name(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw or raw.startswith("/"):
            return False
        if self._is_greeting_only(raw):
            return False
        if _is_bot_meta_question(raw):
            return False
        if GeminiClient.looks_like_question(raw):
            return False
        norm = normalize(raw)
        if any(w in norm for w in ["موعد", "حجز", "عاجل", "روتيني", "اليوم", "بكرا", "شكوى", "الم", "وجع", "مرض"]):
            return False
        words = raw.split()
        if len(words) > 4 or len(raw) > 40:
            return False
        if any(ch.isdigit() for ch in raw):
            return False
        return True

    @staticmethod
    def _is_greeting_only(text: str) -> bool:
        norm = normalize(text or "").strip()
        if not norm:
            return False
        for phrase in _GREETING_PHRASES:
            p = normalize(phrase)
            if norm == p or norm.startswith(f"{p} ") or norm.endswith(f" {p}"):
                return True
        tokens = set(norm.split())
        if not tokens:
            return False
        return tokens.issubset(_GREETING_TOKENS) or norm in _GREETING_TOKENS

    @staticmethod
    def _greeting_and_ask_name() -> str:
        return f"أهلاً وسهلاً 👋 {FIELD_QUESTIONS_AR['name']}"

    def _preload_patient_name(self) -> None:
        if self.data.get("name"):
            return
        from database.db import get_db
        from database import crud

        with get_db() as db:
            patient = crud.get_patient_by_telegram_id(db, self.user_id)
            if patient is not None and isinstance(getattr(patient, "name", None), str):
                name = patient.name.strip()
                if name:
                    self.data["name"] = name

    def _gp_fallback_message(self, label: str) -> str:
        return (
            f"عذراً، {label} غير متوفرة لدينا حالياً.\n"
            f"نقدر نحجزك في {SPECIALTY_NAMES_AR['general_practice']}.\n"
            "موافق؟ (✅ تأكيد / ❌ إلغاء)"
        )

    def _slot_to_dict(self, slot) -> dict:
        return {
            "slot_id": slot.slot_id,
            "slot_datetime": slot.slot_datetime,
            "specialty": slot.doctor.specialty if slot.doctor else slot.specialty,
            "priority_class": slot.priority_class,
            "doctor_id": slot.doctor.doctor_id if slot.doctor else None,
            "doctor_name": slot.doctor.name if slot.doctor else None,
            "clinic_code": slot.doctor.clinic_code if slot.doctor else None,
            "clinic_name": slot.doctor.clinic_name if slot.doctor else None,
        }

    def _format_slot_options_list(self) -> str:
        lines = ["📋 المواعيد المتاحة حالياً:"]
        for idx, slot in enumerate(self.slot_options, start=1):
            dt = slot["slot_datetime"].strftime("%A، %d/%m/%Y — %H:%M")
            marker = " ← المقترح" if idx - 1 == self.slot_index else ""
            lines.append(
                f"{idx}. {dt} — {slot.get('doctor_name') or '—'} "
                f"({slot.get('clinic_name') or '—'}){marker}"
            )
        lines.append("\nاكتب رقم الخيار إذا بدك تغيّر.")
        return "\n".join(lines)

    def _format_confirm_message(self) -> tuple[str, UIAction, dict]:
        dt = self.slot["slot_datetime"].strftime("%A، %d/%m/%Y — %H:%M")
        alt_hint = ""
        if len(self.slot_options) > 1:
            alt_hint = f"\n\n🔄 في {len(self.slot_options) - 1} مواعيد ثانية — قولي «موعد آخر» أو اضغط 🔄."
        return self._reply(
            f"وجدت موعد مناسب! 📅\n\n"
            f"📆 {dt}\n"
            f"🏥 التخصص: {self.data.get('specialty_ar', '')}\n"
            f"👨‍⚕️ الطبيب: {self.slot.get('doctor_name') or '—'}\n"
            f"🏢 العيادة: {self.slot.get('clinic_name') or '—'} ({self.slot.get('clinic_code') or '—'})\n"
            f"{alt_hint}\n"
            f"مناسبلك؟",
            UIAction.SHOW_CONFIRM,
        )

    def _is_clarification_request(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        return bool(lowered) and any(word in lowered for word in [
            "أنت فهمت", "انت فهمت", "ماذا فهمت", "إيه اللي فهمته", "ايش فهمت",
            "أشرح", "اشرح", "what did you understand", "what do you know",
        ])

    def _is_new_booking_request(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        return any(token in lowered for token in ["حجز موعد", "موعد جديد", "ابدأ", "من جديد", "restart", "book"])

    def _reset(self) -> None:
        self.state = State.GREETING
        self.data.clear()
        self.slot = None
        self.slot_options = []
        self.slot_index = 0
        self.priority = None
        self.finalized_appointment_id = None
        self.waitlist_position = None

    def to_snapshot(self) -> dict:
        priority_json = None
        if self.priority is not None:
            priority_json = {
                "priority_class": getattr(self.priority, "priority_class", None),
                "score": getattr(self.priority, "score", None),
                "label_ar": getattr(self.priority, "label_ar", None),
                "breakdown": getattr(self.priority, "breakdown", None),
            }
        slot_options = []
        for item in self.slot_options:
            copy = dict(item)
            dt = copy.get("slot_datetime")
            if hasattr(dt, "isoformat"):
                copy["slot_datetime"] = dt.isoformat()
            slot_options.append(copy)
        slot_json = None
        if self.slot:
            slot_json = dict(self.slot)
            dt = slot_json.get("slot_datetime")
            if hasattr(dt, "isoformat"):
                slot_json["slot_datetime"] = dt.isoformat()
        return {
            "state": self.state.name,
            "data_json": self.data,
            "slot_options_json": slot_options,
            "slot_index": self.slot_index,
            "slot_json": slot_json,
            "priority_json": priority_json,
            "finalized_appointment_id": self.finalized_appointment_id,
        }

    @classmethod
    def from_snapshot(cls, user_id: int, row) -> PatientFSM:
        from datetime import datetime

        fsm = cls(user_id=user_id)
        fsm.state = State[row.state]
        fsm.data = row.data_json or {}
        fsm.slot_index = row.slot_index or 0
        fsm.finalized_appointment_id = row.finalized_appointment_id
        fsm.waitlist_position = (row.data_json or {}).get("waitlist_position")

        slot_options = []
        for item in row.slot_options_json or []:
            copy = dict(item)
            raw_dt = copy.get("slot_datetime")
            if isinstance(raw_dt, str):
                copy["slot_datetime"] = datetime.fromisoformat(raw_dt)
            slot_options.append(copy)
        fsm.slot_options = slot_options

        if row.slot_json:
            slot = dict(row.slot_json)
            raw_dt = slot.get("slot_datetime")
            if isinstance(raw_dt, str):
                slot["slot_datetime"] = datetime.fromisoformat(raw_dt)
            fsm.slot = slot
        elif slot_options and 0 <= fsm.slot_index < len(slot_options):
            fsm.slot = slot_options[fsm.slot_index]

        if row.priority_json:
            from scheduler.priority import PriorityResult

            fsm.priority = PriorityResult(
                score=float(row.priority_json.get("score", 0.3)),
                priority_class=row.priority_json.get("priority_class", "P3"),
                label_ar=row.priority_json.get("label_ar", ""),
                label_color="",
                breakdown=row.priority_json.get("breakdown") or {},
            )
        return fsm

    @property
    def is_done(self) -> bool:
        return self.state in (State.FINALIZED, State.WAITLISTED, State.CANCELLED)
