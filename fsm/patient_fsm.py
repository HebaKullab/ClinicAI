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
    detect_unsupported_specialty,
    is_supported_specialty,
    SPECIALTY_NAMES_AR,
)
from nlp.booking_agent import (
    BookingTurnResult,
    append_history,
    apply_extracted_to_data,
    fallback_reply,
    merge_rule_extracted,
    missing_required_fields,
    run_booking_turn,
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
    CHATTING = auto()          # LLM-driven free conversation (collects all fields)
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
    UNRECOGNIZED = auto()


_CHAT_LEGACY_STATES = frozenset({
    State.GREETING,
    State.COLLECT_NAME,
    State.COLLECT_COMPLAINT,
    State.COLLECT_URGENCY,
    State.COLLECT_TIME,
    State.UNRECOGNIZED,
})


def _normalize_loaded_state(state: State) -> State:
    if state in _CHAT_LEGACY_STATES:
        return State.CHATTING
    return state


def _is_chatting_state(state: State) -> bool:
    return state in _CHAT_LEGACY_STATES or state == State.CHATTING


# Required fields — the checklist before touching scheduling.
REQUIRED_FIELDS: list[str] = ["name", "complaint", "urgency_score", "time_pref"]

FIELD_QUESTIONS_AR: dict[str, str] = {
    "name": "ما اسمك الكريم؟ 😊",
    "complaint": "سلامتك! شو الأعراض أو سبب الزيارة؟ 🩺",
    "urgency_score": "كيف شايف حالة المريض؟ (عاجل 🔴 / متوسط 🟡 / روتيني 🟢)",
    "time_pref": "متى يناسبك الموعد؟ (اليوم، بكرا، الأسبوع الجاي...)",
}

CONFIRM_WORDS = {
    "نعم", "ايوه", "آيوه", "تمام", "ماشي", "اوك", "اوكي", "yes", "يلا", "احجز",
    "تاكيد", "تأكيد", "تاكيد الحجز", "تأكيد الحجز", "موافق", "✅",
    "اه", "آه", "ايه", "اي", "ايوا", "يب", "ok", "okay",
    "مناسب", "مناسبلي", "مناسب لك", "حاضر", "كويس", "ممتاز", "اكيد", "أكيد",
    "صح", "ثبت", "ثبته", "ثبتلي", "خلص", "طيب", "حسنا", "حسناً", "حسنًا",
    "اوكيه", "okay", "sure", "confirm",
}
CANCEL_WORDS = {"لا", "الغي", "إلغي", "الغاء", "إلغاء", "بدي الغي", "مش حابب", "❌"}
EDIT_WORDS = {
    "تعديل", "تعديل الموعد", "✏️", "✏️ تعديل الموعد",
    "اغير", "أغير", "بدي اغير", "بدي أغير", "غير الوقت", "غير الموعد",
}
NEXT_SLOT_WORDS = {
    "موعد آخر", "🔄", "🔄 موعد آخر",
    "وقت تاني", "وقت ثاني", "موعد تاني", "موعد ثاني", "بدي اغير", "بدي أغير",
}


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
    """Patient wants to back out — «ما بدي اشي»، «لا شكراً» (not explicit cancel buttons)."""
    if not norm or _matches_any_token(norm, CANCEL_WORDS):
        return False
    decline = (
        "ما بدي", "ما اريد", "مش بدي", "مش اريد", "ما بده", "لا بدي", "لا اريد",
        "لا شكر", "مش حاب", "ما بدي اشي", "ما بدي شي", "ما بدي اي", "ما اريد شي",
        "مو بدي", "مش عايز", "بطل", "خلص", "سكر",
        "مش interested", "no thanks",
    )
    return any(p in norm for p in decline)


def _looks_like_soft_confirm(norm: str) -> bool:
    """Informal Arabic affirmatives — «اه», «بدي موعد اه», emoji confirm buttons."""
    if not norm or _matches_any_token(norm, CANCEL_WORDS):
        return False
    if _looks_like_slot_list_request(norm) or _looks_like_decline(norm):
        return False
    if any(w in norm for w in ("اشوف", "فرج", "عرض", "شوف", "ورج", "ليش", "لماذا", "كيف", "متى", "وين")):
        return False
    # Time-change requests are not confirms.
    if any(w in norm for w in ("ساعه", "ساعة", "الصبح", "العصر", "المساء", "بعد بكرا", "بكرا", "اليوم")):
        if any(w in norm for w in ("بدي", "غير", "بدل", "عدل", "مو", "مش")):
            return False
    if _matches_any_token(norm, CONFIRM_WORDS):
        return True
    tokens = _tokenize(norm)
    if tokens & {
        "اه", "آه", "ايه", "اي", "ايوا", "يب", "تمام", "ماشي", "اوك", "اوكي", "ok",
        "مناسب", "حاضر", "كويس", "ممتاز", "اكيد", "صح", "طيب", "خلص", "ثبت",
    }:
        return True
    if any(w in norm for w in ("بدي", "بده", "اريد", "حاب", "حابب")):
        if any(w in norm for w in ("احجز", "حجز", "نعم", "اه", "آه", "موافق", "تمام", "اكد", "أكد", "ثبت")):
            return True
        # «بدي موعد» alone is confirm intent; «بدي اشوف مواعيد» excluded above
        if "موعد" in norm and "اشوف" not in norm and "فرج" not in norm:
            return True
    return False


def _looks_like_confirm_phase_question(norm: str, raw: str = "") -> bool:
    """Patient is asking about the offer — do not auto-finalize."""
    if not norm:
        return False
    if (raw or "").strip().endswith(("?", "؟")):
        return True
    return any(w in norm for w in (
        "ليش", "لماذا", "كيف", "وين", "متى", "شو يعني", "ايش يعني", "وضح", "اشرح",
        "قديش", "كم السعر", "وين العياده", "وين العيادة", "شو صار", "ماذا عن",
        "هل يمكن", "ممكن اغير", "ممكن أغير",
    ))

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
    """«ماذا تريد» / «شو بدك» / «شو وظيفتك» — not a name or complaint."""
    norm = normalize(text or "").strip()
    phrases = (
        "ماذا تريد", "ماذا تبغ", "شو تريد", "شو بدك", "شو بتحب",
        "شو وظيف", "شو شغل", "شو بتعمل", "شو دورك", "مين انت", "من انت",
        "what do you want", "what do you need", "what are you", "who are you",
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


def _message_has_urgency_signal(text: str) -> bool:
    norm = normalize(text or "").lower()
    tokens = (
        "عاجل", "طارئ", "فوري", "خطير", "روتيني", "متوسط", "عادي",
        "p1", "p2", "p3", "🔴", "🟡", "🟢",
    )
    return any(t in norm for t in tokens)


def _looks_like_appointment_inquiry(norm: str) -> bool:
    """Natural-language ask about the patient's booked/waitlisted appointment."""
    if not norm:
        return False
    if "استعلام" in norm or "موعدي" in norm:
        return True
    if "مواعيد" in norm and any(w in norm for w in ("موجود", "مسجل", "محجوز", "ضايل", "متبق", "باقي")):
        return True
    if "موعد" in norm and any(w in norm for w in ("مسجل", "محجوز", "حجزي", "اخر", "آخر", "عندي", "وين")):
        return True
    return False


def _looks_like_frustration_question(norm: str, raw: str = "") -> bool:
    if not norm and not raw:
        return False
    phrases = (
        "ليش هيك", "ليش هيك", "شو هاد", "شو هيك", "ما فهمت", "مش فاهم",
        "لييه", "ليش", "لماذا هكذا", "لماذا", "why", "what is this",
    )
    for p in phrases:
        if p in (norm or "") or p in normalize(raw or ""):
            return True
    return False


def _is_cancel_intent(text: str) -> bool:
    norm = normalize(text or "").lower().strip()
    if not norm:
        return False
    cancel_tokens = ("الغاء", "إلغاء", "الغي", "إلغي", "الغيلي", "كنسل", "cancel")
    if any(token in norm for token in cancel_tokens):
        return True
    return any(phrase in norm for phrase in ("بديش احجز", "مش حاب احجز", "لا اريد الحجز", "بدي الغي"))


def _clean_extracted_name(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw or raw.startswith("/"):
        return None
    if PatientFSM._is_greeting_only(raw) or GeminiClient.looks_like_question(raw) or GeminiClient.looks_off_topic(raw):
        return None
    norm = normalize(raw)
    if any(w in norm for w in ["موعد", "حجز", "عاجل", "روتيني", "اليوم", "بكرا", "شكوى", "الم", "وجع", "مرض"]):
        return None
    cleaned = raw
    prefixes = [
        "أنا اسمي", "انا اسمي", "اسمي", "أنا", "انا", "معك", "معاكم",
        "مرحبا أنا", "مرحبا انا", "أهلاً أنا", "اهلا انا", "حجز باسم", "اسم"
    ]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix + " ") or cleaned.startswith(prefix + " "):
            cleaned = cleaned[len(prefix):].strip()
    words = cleaned.split()
    if 1 <= len(words) <= 4 and len(cleaned) <= 40 and not any(ch.isdigit() for ch in cleaned):
        return cleaned
    return None


@dataclass
class PatientFSM:
    user_id: int
    services: BookingServices = field(default_factory=BookingServices.default)
    state: State = State.CHATTING
    data: dict = field(default_factory=dict)
    chat_history: list = field(default_factory=list)
    slot: Optional[dict] = None              # Plain dict, not detached ORM object
    slot_options: list = field(default_factory=list)
    slot_index: int = 0
    priority: Optional[object] = None        # PriorityResult
    finalized_appointment_id: Optional[str] = None
    waitlist_position: Optional[int] = None

    # ── Public entry ──────────────────────────────────────────────────────────

    def _phase_for_state(self) -> str:
        if self.state == State.CONFIRM:
            return "CONFIRM"
        if self.state == State.OFFER_GP_FALLBACK:
            return "GP_FALLBACK"
        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED):
            return "TERMINAL"
        return "CHATTING"

    def _slot_context(self) -> dict | None:
        if not self.slot:
            return None
        when = "—"
        if self.slot.get("slot_datetime"):
            when = self.slot["slot_datetime"].strftime("%A، %d/%m/%Y — %H:%M")
        return {"when": when}

    def _stored_appointment_summary(self) -> str | None:
        from database.db import get_db
        from database import crud

        with get_db() as db:
            appt = crud.get_latest_patient_appointment(db, self.user_id)
        if not appt:
            return None
        date_text = (
            appt.appt_datetime.strftime("%A، %d/%m/%Y — %H:%M")
            if appt.appt_datetime
            else "قائمة الانتظار"
        )
        specialty = appt.specialty_ar or appt.specialty or "—"
        return f"{date_text} — {specialty} — {appt.status}"

    def _build_operation_context(self, text: str) -> dict:
        ctx: dict = {
            "missing_fields": self._missing_fields(),
        }
        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED):
            ctx["terminal_state"] = self.state.name
        if self.data.get("unsupported_clinic_label"):
            ctx["unsupported_specialty"] = self.data["unsupported_clinic_label"]
        if self.state == State.CONFIRM and self.slot:
            when = self.slot.get("slot_datetime")
            ctx["proposed_slot"] = (
                when.strftime("%A، %d/%m/%Y — %H:%M") if when else "—"
            )
            ctx["slot_options_count"] = len(self.slot_options)
        appt = self._stored_appointment_summary()
        if appt:
            ctx["stored_appointment"] = appt

        norm = normalize(text or "")
        if _looks_like_appointment_inquiry(norm):
            ctx["rule_hint"] = "inquiry"
        elif _is_cancel_intent(text):
            ctx["rule_hint"] = "cancel"
        elif self._is_new_booking_request(text):
            ctx["rule_hint"] = "new_booking"
        elif any(p in norm for p in ("تواصل", "اتصل", "رقم", "هاتف", "موقع", "عنوان")):
            ctx["rule_hint"] = "contact"
        return ctx

    async def _ai_turn(
        self,
        text: str,
        *,
        phase: str | None = None,
        append_user: bool = True,
    ) -> BookingTurnResult:
        """Single LLM call per user message — reply + intent + extraction."""
        raw_text = (text or "").strip()
        phase_name = phase or self._phase_for_state()
        if append_user and raw_text:
            self.chat_history = append_history(self.chat_history, "user", raw_text)
        return await run_booking_turn(
            raw_text or "مرحبا",
            phase_name,
            self.data,
            self.chat_history,
            self._slot_context(),
            self._build_operation_context(text),
            required_fields=REQUIRED_FIELDS,
            score_from_label=self._score_from_label,
        )

    async def _execute_turn_intent(
        self, turn: BookingTurnResult, text: str
    ) -> tuple[str, UIAction, dict] | None:
        """Rule-driven operations triggered by LLM/rule intent — no extra AI calls."""
        intent = turn.intent

        if intent == "cancel":
            cancel_db = self.state in (
                State.GREETING,
                State.FINALIZED,
                State.WAITLISTED,
                State.CANCELLED,
            )
            return await self._execute_cancellation(cancel_db=cancel_db)

        if intent == "new_booking":
            self._reset()
            return await self.welcome_message()

        if intent == "inquiry":
            factual = self._format_stored_appointment_reply()
            reply = turn.reply.strip() if turn.reply else factual
            if turn.reply and self._stored_appointment_summary() and "موعد" not in normalize(turn.reply):
                reply = f"{turn.reply}\n\n{factual}"
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.SHOW_MAIN_MENU)

        if intent == "contact":
            reply = (
                turn.reply
                or "📞 تواصلك وصل. يمكنك كتابة رسالتك هنا، وسيتم حفظها في سجل المحادثات للعيادة."
            )
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.SHOW_MAIN_MENU)

        if intent == "accept_gp" and self.state == State.OFFER_GP_FALLBACK:
            return await self._accept_gp_fallback()

        if intent == "reject_gp" and self.state == State.OFFER_GP_FALLBACK:
            return await self._execute_cancellation(cancel_db=False)

        if intent == "confirm" and self.state == State.CONFIRM:
            return await self._finalize_confirm()

        if intent == "decline" and self.state == State.CONFIRM:
            self.state = State.CANCELLED
            reply = turn.reply or "تمام، ما في مشكلة — ألغيت الحجز. إذا احتجت شي لاحقاً أنا هون. 👋"
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        if intent == "edit_time" and self.state == State.CONFIRM:
            self.data.pop("time_pref", None)
            self.state = State.CHATTING
            reply = turn.reply or ("تمام، متى تحب الموعد؟\n" + FIELD_QUESTIONS_AR["time_pref"])
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        if intent == "next_slot" and self.state == State.CONFIRM:
            return await self._cycle_slot_option(turn.reply)

        if intent == "slot_list" and self.state == State.CONFIRM:
            reply = turn.reply or self._format_slot_options_list()
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        return None

    async def handle(self, text: str) -> tuple[str, UIAction, dict]:
        """Process one message — one LLM turn for reply; rules for operations."""
        text = text or ""
        norm = normalize(text)

        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED):
            if self._is_new_booking_request(text):
                self._reset()
                return await self.welcome_message()
            norm_terminal = normalize(text)
            if _looks_like_frustration_question(norm_terminal, text):
                meta = self._terminal_meta_reply(text, norm_terminal)
                if meta:
                    self.chat_history = append_history(self.chat_history, "user", text.strip())
                    self.chat_history = append_history(self.chat_history, "assistant", meta)
                    return self._reply(meta, UIAction.SHOW_MAIN_MENU)
            if _looks_like_appointment_inquiry(norm_terminal):
                reply = self._format_stored_appointment_reply()
                self.chat_history = append_history(self.chat_history, "user", text.strip())
                self.chat_history = append_history(self.chat_history, "assistant", reply)
                return self._reply(reply, UIAction.SHOW_MAIN_MENU)
            meta_early = self._terminal_meta_reply(text, norm_terminal)
            if meta_early and (_is_bot_meta_question(text) or any(
                p in norm_terminal for p in ("اداره", "ادارة", "لوحه", "لوحة", "dashboard", "admin", "الادارة", "تواصل", "اتصل")
            )):
                self.chat_history = append_history(self.chat_history, "user", text.strip())
                self.chat_history = append_history(self.chat_history, "assistant", meta_early)
                return self._reply(meta_early, UIAction.SHOW_MAIN_MENU)

        if _is_name_dispute(text):
            self.data.pop("name", None)
            self.state = State.CHATTING
            turn = await self._ai_turn(text)
            reply = turn.reply or "عذراً على اللبس! 🙂 شو اسمك الصحيح؟"
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        if _is_bot_meta_question(text):
            self.data.pop("name", None)
            self.state = State.CHATTING
            turn = await self._ai_turn(text)
            reply = turn.reply or (
                "أنا مساعد حجز المواعيد في العيادة 🏥. بساعدك تحجز موعد. "
                + FIELD_QUESTIONS_AR["name"]
            )
            if "مساعد" not in reply:
                reply = "أنا مساعد حجز المواعيد في العيادة 🏥. " + reply
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        if self._is_clarification_request(text):
            turn = await self._ai_turn(text)
            reply = turn.reply or "تمام، خلينا نكمّل الحجز خطوة بخطوة."
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        if self.state == State.VALIDATE:
            return await self._run_validate()

        if self.state == State.CLASSIFY:
            return await self._classify_and_schedule()

        if self.state == State.FIND_SLOT:
            return await self._find_slot()

        if self.state == State.COLLECT_SPECIALTY:
            return await self._handle_collect_specialty(text)

        if self.state == State.CONFIRM:
            rule_result = await self._handle_confirm_rules(text)
            if rule_result is not None:
                return rule_result

        if self.state == State.OFFER_GP_FALLBACK:
            rule_result = await self._handle_gp_fallback_rules(text)
            if rule_result is not None:
                return rule_result

        turn = await self._ai_turn(text)
        intent_result = await self._execute_turn_intent(turn, text)
        if intent_result is not None:
            return intent_result

        if self.state in (State.FINALIZED, State.CANCELLED, State.WAITLISTED):
            return await self._apply_terminal_turn(turn, text)

        if self.state == State.CONFIRM:
            reply = turn.reply or self._confirm_nudge()
            from nlp.booking_agent import reply_claims_booking_done

            if reply_claims_booking_done(reply):
                return await self._finalize_confirm()
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        if self.state == State.OFFER_GP_FALLBACK:
            label = self.data.get("unsupported_clinic_label", "هذا التخصص")
            reply = turn.reply or self._gp_fallback_message(label)
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.NONE)

        return await self._apply_chatting_turn(turn, text)

    def _sanitize_chatting_reply(self, reply: str, user_text: str) -> str:
        """Replace hallucinated booking confirmations and re-asks for known fields."""
        from nlp.booking_agent import FIELD_QUESTIONS_AR as AGENT_Q
        from nlp.booking_agent import reply_claims_booking_done

        text = (reply or "").strip()
        if reply_claims_booking_done(text):
            missing = self._missing_fields()
            if missing:
                return FIELD_QUESTIONS_AR[missing[0]]
            return "تمام، خلينا نكمّل. إذا بدك تأكيد موعد موجود قولي نعم بعد ما أعرضه عليك."

        missing = self._missing_fields()
        if not missing:
            return text

        next_q = FIELD_QUESTIONS_AR[missing[0]]
        # If the model re-asks a field we already have, steer to the real gap.
        already_have_markers = []
        if self.data.get("name"):
            already_have_markers.extend([FIELD_QUESTIONS_AR["name"], AGENT_Q["name"], "ما اسمك"])
        if self.data.get("complaint"):
            already_have_markers.extend([FIELD_QUESTIONS_AR["complaint"], AGENT_Q["complaint"], "شو الأعراض", "سبب الزيارة"])
        if self.data.get("urgency_score") is not None:
            already_have_markers.extend([
                FIELD_QUESTIONS_AR["urgency_score"], AGENT_Q["urgency_score"],
                "عاجل / متوسط", "عاجل/متوسط", "مستوى الأولوية",
            ])
        if self.data.get("time_pref"):
            already_have_markers.extend([FIELD_QUESTIONS_AR["time_pref"], AGENT_Q["time_pref"], "متى يناسبك"])

        if any(m in text for m in already_have_markers if m):
            name = self.data.get("name")
            if missing[0] == "complaint" and name:
                return f"أهلاً {name}! 😊\n" + next_q
            return next_q

        if not text:
            return next_q
        return text

    async def _handle_collect_specialty(self, text: str) -> tuple[str, UIAction, dict]:
        unsupported = detect_unsupported_specialty(text)
        if unsupported:
            self._stash_complaint_for_unsupported(text)
            self.data["unsupported_clinic_label"] = unsupported
            self.state = State.OFFER_GP_FALLBACK
            return self._reply(self._gp_fallback_message(unsupported), UIAction.NONE)

        specialty_key = self._parse_specialty_label(text)
        if not specialty_key:
            turn = await self._ai_turn(text)
            intent_result = await self._execute_turn_intent(turn, text)
            if intent_result is not None:
                return intent_result
            return self._reply(
                turn.reply or "ما فهمت التخصص. اكتب اسم التخصص الأقرب لحالتك.",
                UIAction.NONE,
            )
        self.data["specialty_hint"] = specialty_key
        self.data["specialty_ar"] = SPECIALTY_NAMES_AR.get(specialty_key, specialty_key)
        self.data["specialty_confirmed_by_patient"] = True
        return await self._score_and_find_slot()

    async def _handle_gp_fallback_rules(self, text: str) -> tuple[str, UIAction, dict] | None:
        norm = normalize(text)
        if _looks_like_soft_confirm(norm):
            return await self._accept_gp_fallback()
        if _matches_any_token(norm, CANCEL_WORDS) or _looks_like_decline(norm):
            return await self._execute_cancellation(cancel_db=False)
        return None

    async def _accept_gp_fallback(self) -> tuple[str, UIAction, dict]:
        self.data["specialty_hint"] = "general_practice"
        self.data["specialty_ar"] = SPECIALTY_NAMES_AR["general_practice"]
        self.data["specialty_method"] = "gp_fallback"
        self.data.pop("unsupported_clinic_label", None)
        if self._missing_fields():
            return await self._run_validate()
        return await self._score_and_find_slot()

    async def _apply_chatting_turn(
        self, turn: BookingTurnResult, text: str
    ) -> tuple[str, UIAction, dict]:
        """Apply LLM turn in CHATTING — rules merge fields; scheduling stays rule-based."""
        self.state = State.CHATTING
        self._preload_patient_name()
        raw_text = (text or "").strip()

        if raw_text and any(ch.isdigit() for ch in raw_text) and not self.data.get("name") and len(raw_text) <= 20:
            if not self.data.get("complaint"):
                reply = turn.reply or ("الاسم ما بكون أرقام 🙂 " + FIELD_QUESTIONS_AR["name"])
                self.chat_history = append_history(self.chat_history, "assistant", reply)
                return self._reply(reply, UIAction.NONE)

        if not turn.off_topic:
            apply_extracted_to_data(self.data, turn.extracted, score_from_label=self._score_from_label)
            self._merge_rules_from_message(raw_text)
        else:
            # Still absorb clear urgency/complaint even if LLM flagged off_topic.
            self._merge_rules_from_message(raw_text)

        unsupported_msg = self.services.detect_unsupported(raw_text) if raw_text else None
        if unsupported_msg and self.data.get("specialty_method") != "gp_fallback":
            self._stash_complaint_for_unsupported(raw_text)
            self.data["unsupported_clinic_label"] = unsupported_msg
            self.state = State.OFFER_GP_FALLBACK
            gp_reply = self._gp_fallback_message(unsupported_msg)
            self.chat_history = append_history(self.chat_history, "assistant", gp_reply)
            return self._reply(gp_reply, UIAction.NONE)

        reply = turn.reply or fallback_reply(self.data, REQUIRED_FIELDS, raw_text).reply
        reply = self._sanitize_chatting_reply(reply, raw_text)
        if (
            self.data.get("name")
            and not self.data.get("complaint")
            and self.data["name"] not in reply
            and (
                FIELD_QUESTIONS_AR["complaint"] in reply
                or FIELD_QUESTIONS_AR["name"] in reply
                or self.data["name"] in raw_text
            )
        ):
            reply = f"أهلاً {self.data['name']}! 😊\n" + FIELD_QUESTIONS_AR["complaint"]
        self.chat_history = append_history(self.chat_history, "assistant", reply)

        complaint_raw = (self.data.get("complaint") or {}).get("raw", "")
        if complaint_raw and self.data.get("specialty_method") != "gp_fallback":
            unsupported = self.services.detect_unsupported(complaint_raw)
            if unsupported:
                self._stash_complaint_for_unsupported(complaint_raw)
                self.data["unsupported_clinic_label"] = unsupported
                self.state = State.OFFER_GP_FALLBACK
                gp_reply = self._gp_fallback_message(unsupported)
                self.chat_history = append_history(self.chat_history, "assistant", gp_reply)
                return self._reply(gp_reply, UIAction.NONE)

        if not self._missing_fields():
            return await self._run_validate()

        return self._reply(reply, UIAction.NONE)

    async def _apply_terminal_turn(
        self, turn: BookingTurnResult, text: str
    ) -> tuple[str, UIAction, dict]:
        """TERMINAL phase — AI reply with rule-based factual fallback."""
        raw = (text or "").strip()
        norm = normalize(raw)

        if self.state == State.CANCELLED and (
            _looks_like_decline(norm) or _matches_any_token(norm, CANCEL_WORDS)
        ):
            reply = turn.reply or "تمام، ما في مشكلة. إذا احتجت أي شي لاحقاً أنا هون. 👋"
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.SHOW_MAIN_MENU)

        if _looks_like_frustration_question(norm, raw):
            meta = self._terminal_meta_reply(raw, norm)
            if meta:
                self.chat_history = append_history(self.chat_history, "assistant", meta)
                return self._reply(meta, UIAction.SHOW_MAIN_MENU)

        meta = self._terminal_meta_reply(raw, norm)
        if meta and not turn.reply:
            self.chat_history = append_history(self.chat_history, "assistant", meta)
            return self._reply(meta, UIAction.SHOW_MAIN_MENU)

        if _looks_like_appointment_inquiry(norm) and not turn.reply:
            reply = self._format_stored_appointment_reply()
            self.chat_history = append_history(self.chat_history, "assistant", reply)
            return self._reply(reply, UIAction.SHOW_MAIN_MENU)

        reply = turn.reply or meta or (
            "تم إنهاء الطلب السابق. إذا بدك حجز جديد اكتب: حجز موعد جديد 📅"
        )
        self.chat_history = append_history(self.chat_history, "assistant", reply)
        return self._reply(reply, UIAction.SHOW_MAIN_MENU)

    async def _handle_chatting(self, text: str) -> tuple[str, UIAction, dict]:
        """Backward-compatible entry — delegates to single AI turn."""
        turn = await self._ai_turn(text)
        intent_result = await self._execute_turn_intent(turn, text)
        if intent_result is not None:
            return intent_result
        return await self._apply_chatting_turn(turn, text)

    def _merge_rules_from_message(self, text: str) -> None:
        """Rule-based extraction layered on LLM output (works offline in tests)."""
        if not (text or "").strip():
            return
        rule = merge_rule_extracted(text, self.data)
        apply_extracted_to_data(self.data, rule, score_from_label=self._score_from_label)
        if not self.data.get("name"):
            cleaned = _clean_extracted_name(text) or (text.strip() if self._looks_like_name(text) else None)
            if cleaned:
                self.data["name"] = cleaned
        if self.data.get("urgency_score") is None or _message_has_urgency_signal(text):
            if _message_has_urgency_signal(text):
                self.data.pop("urgency_score", None)
            self._absorb_urgency(normalize(text), text)
        if not self.data.get("time_pref"):
            mapped = self._parse_time_label(text)
            if mapped:
                self.data["time_pref"] = mapped
        if not self.data.get("complaint") and len(text.strip()) >= 2 and not self._is_greeting_only(text):
            complaint = extract_patient_fields(text).get("complaint")
            if complaint:
                self.data["complaint"] = complaint
            elif not GeminiClient.looks_like_question(text) and not _is_bot_meta_question(text):
                from nlp.booking_agent import _message_looks_like_complaint

                if _message_looks_like_complaint(text):
                    self.data["complaint"] = {
                        "raw": text.strip(),
                        "category": "general",
                        "urgency_score": 0.3,
                        "specialty": "general_practice",
                    }

    async def welcome_message(self) -> tuple[str, UIAction, dict]:
        """One-shot welcome — single AI turn."""
        self._preload_patient_name()
        self.state = State.CHATTING
        turn = await self._ai_turn("مرحبا", append_user=True)
        if not turn.off_topic:
            apply_extracted_to_data(self.data, turn.extracted, score_from_label=self._score_from_label)

        if turn.reply:
            reply = turn.reply
        elif self.data.get("name"):
            reply = (
                f"👋 أهلاً {self.data['name']}! أنا مساعد الحجز في العيادة. 🩺\n"
                + FIELD_QUESTIONS_AR["complaint"]
            )
        else:
            reply = "👋 أهلاً وسهلاً بك في العيادة! 🏥\nأنا مساعد الحجز، " + FIELD_QUESTIONS_AR["name"]

        self.chat_history = append_history(self.chat_history, "assistant", reply)
        return self._reply(reply, UIAction.SHOW_MAIN_MENU)

    async def begin_booking_message(self) -> tuple[str, UIAction, dict]:
        """Welcome for 'حجز موعد جديد' — single AI turn."""
        self._preload_patient_name()
        self.state = State.CHATTING
        self.chat_history = append_history(self.chat_history, "user", "حجز موعد جديد")
        turn = await self._ai_turn("حجز موعد جديد", append_user=False)
        if not turn.off_topic:
            apply_extracted_to_data(self.data, turn.extracted, score_from_label=self._score_from_label)
        if turn.reply:
            reply = turn.reply
        elif self.data.get("name"):
            reply = f"📅 تمام! أهلاً {self.data['name']}، خلينا نبدأ حجز جديد.\n" + FIELD_QUESTIONS_AR["complaint"]
        else:
            reply = "📅 تمام، خلينا نبدأ حجز جديد. " + FIELD_QUESTIONS_AR["name"]
        self.chat_history = append_history(self.chat_history, "assistant", reply)
        return self._reply(reply, UIAction.NONE)

    async def _execute_cancellation(self, cancel_db: bool = True) -> tuple[str, UIAction, dict]:
        """Cancel DB appointment if requested, free slot, and set state to CANCELLED."""
        if cancel_db:
            from database.db import get_db
            from database import crud

            with get_db() as db:
                crud.cancel_latest_patient_appointment(db, self.user_id)

            reply = "تم إلغاء آخر موعد لك وإرجاع خيار الحجز كمتاح في النظام. ✅\nإذا أردت حجز موعد جديد في أي وقت، اكتب (حجز موعد جديد) وأنا بخدمتك دائماً. 👋"
        else:
            reply = "تم إلغاء طلب الحجز الحالي. ✅\nإذا أردت حجز موعد جديد في أي وقت، اكتب (حجز موعد جديد) وأنا بخدمتك دائماً. 👋"

        self.state = State.CANCELLED
        return self._reply(reply, UIAction.SHOW_MAIN_MENU)

    async def handle_callback(self, data: str) -> tuple[str, UIAction, dict]:
        # Kept for compatibility if inline keyboards are added later.
        if data.startswith("urgency:"):
            level = data.split(":", 1)[1]
            self._set_urgency_from_label(level)
            self.state = State.CHATTING
            missing = self._missing_fields()
            if not missing:
                return await self._run_validate()
            return self._reply(FIELD_QUESTIONS_AR[missing[0]], UIAction.NONE)

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
        """Outbound gate: never claim a booking unless finalize already set FINALIZED."""
        from nlp.booking_agent import reply_claims_booking_done

        body = text or ""
        if reply_claims_booking_done(body) and self.state != State.FINALIZED:
            if self.state == State.CONFIRM and self.slot:
                body = self._confirm_nudge()
            elif not self._missing_fields():
                body = "تمام، لسا بلّش أثبت الموعد بالنظام. رح أعرضلك أقرب وقت متاح."
            else:
                missing = self._missing_fields()
                body = FIELD_QUESTIONS_AR[missing[0]] if missing else self._confirm_nudge()
        return body, action, payload or {}

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
            self.state = State.CHATTING
            return self._reply(
                f"بعدنا محتاجين معلومة واحدة 📋\n{FIELD_QUESTIONS_AR[first_missing]}",
                UIAction.NONE,
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
            return self._reply(self._gp_fallback_message(unsupported), UIAction.NONE)

        spec_result = auto_resolve_specialty(self.services.classify(norm_complaint))

        if not is_supported_specialty(spec_result.get("specialty")):
            label = spec_result.get("specialty_ar") or spec_result.get("specialty") or "هذا التخصص"
            self.data["unsupported_clinic_label"] = label
            self.state = State.OFFER_GP_FALLBACK
            return self._reply(self._gp_fallback_message(label), UIAction.NONE)

        self.data["specialty_hint"] = spec_result["specialty"]
        self.data["specialty_ar"] = spec_result["specialty_ar"]
        self.data["specialty_method"] = spec_result.get("method")
        self.data["specialty_confidence"] = spec_result.get("confidence")
        if spec_result.get("custom_reply"):
            self.data["custom_reply"] = spec_result["custom_reply"]

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

    async def _handle_confirm_rules(self, text: str) -> tuple[str, UIAction, dict] | None:
        norm = normalize(text)
        await self._reload_slot_options_if_needed()

        if _matches_any_token(norm, CANCEL_WORDS):
            self.state = State.CANCELLED
            return self._reply("تم الإلغاء. إذا احتجت أي شيء، أنا هون. 👋", UIAction.NONE)

        if _looks_like_decline(norm):
            self.state = State.CANCELLED
            return self._reply("تمام، ما في مشكلة — ألغيت الحجز. إذا احتجت شي لاحقاً أنا هون. 👋", UIAction.NONE)

        if _matches_any_token(norm, EDIT_WORDS):
            self.data.pop("time_pref", None)
            self.state = State.CHATTING
            return self._reply(
                "تمام، متى تحب الموعد؟\n" + FIELD_QUESTIONS_AR["time_pref"],
                UIAction.NONE,
            )

        if _matches_any_token(norm, NEXT_SLOT_WORDS) or _looks_like_slot_list_request(norm):
            if _looks_like_slot_list_request(norm):
                return self._reply(self._format_slot_options_list(), UIAction.NONE)
            return await self._cycle_slot_option(None)

        if self.slot_options and norm.strip().isdigit():
            pick = int(norm.strip()) - 1
            if 0 <= pick < len(self.slot_options):
                self.slot_index = pick
                self.slot = self.slot_options[pick]
                extra = f"\n\n(اخترت خيار {pick + 1} من {len(self.slot_options)})"
                reply, action, payload = self._format_confirm_message()
                return self._reply(reply + extra, action, payload)

        # Soft / expanded confirms — modern models often skip returning intent=confirm.
        if _looks_like_soft_confirm(norm):
            return await self._finalize_confirm()

        # Short non-question replies at CONFIRM default to booking (rules > LLM).
        # Skip dots/noise — those get an AI nudge instead.
        if (
            self.slot
            and not _is_low_signal_input(text)
            and not _looks_like_confirm_phase_question(norm, text)
            and len((text or "").strip()) <= 40
            and 1 <= len(norm.split()) <= 8
            and not any(w in norm for w in ("ساعه", "ساعة", "غير", "بدل", "عدل"))
        ):
            return await self._finalize_confirm()

        return None

    async def _handle_confirm(self, text: str) -> tuple[str, UIAction, dict]:
        """Confirm phase — rules first; LLM only for clarifying questions."""
        rule_result = await self._handle_confirm_rules(text)
        if rule_result is not None:
            return rule_result
        turn = await self._ai_turn(text)
        # Prefer rule/LLM confirm intents — never trust a verbal "تم الحجز" alone.
        intent_result = await self._execute_turn_intent(turn, text)
        if intent_result is not None:
            return intent_result
        from nlp.booking_agent import reply_claims_booking_done

        if reply_claims_booking_done(turn.reply or ""):
            return await self._finalize_confirm()
        # Clarifying question only — keep chatting about the offer.
        reply = turn.reply or self._confirm_nudge()
        if reply_claims_booking_done(reply):
            return await self._finalize_confirm()
        self.chat_history = append_history(self.chat_history, "assistant", reply)
        return self._reply(reply, UIAction.NONE)
    async def _cycle_slot_option(self, ai_reply: str | None) -> tuple[str, UIAction, dict]:
        if len(self.slot_options) > 1:
            self.slot_index = (self.slot_index + 1) % len(self.slot_options)
            self.slot = self.slot_options[self.slot_index]
            extra = f"\n\n(خيار {self.slot_index + 1} من {len(self.slot_options)})"
            reply, action, payload = self._format_confirm_message()
            if ai_reply:
                return self._reply(f"{ai_reply}\n\n{reply}{extra}", action, payload)
            return self._reply(reply + extra, action, payload)
        msg = ai_reply or "هذا هو أقرب موعد متاح حالياً."
        return self._reply(msg, UIAction.NONE)

    async def _finalize_confirm(self) -> tuple[str, UIAction, dict]:
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

    def _confirm_nudge(self) -> str:
        return "لسا معك بالحجز 🙂 بدك تأكيد الموعد، تشوف وقت ثاني، أو نلغي؟"

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
        # existing implementation unchanged
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
                return self._reply(answer.strip(), UIAction.NONE)
        return self._reply(self._confirm_nudge(), UIAction.NONE)

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
        if any(w in combined for w in ["اي وقت", "أي وقت"]) and _is_chatting_state(self.state):
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
                self.state = State.CHATTING
        if self._is_greeting_only(raw) and isinstance(name, str) and normalize(name) == normalize(raw):
            self.data.pop("name", None)
            if self.state == State.COLLECT_COMPLAINT:
                self.state = State.CHATTING

    async def maybe_gemini_fallback(self, text: str) -> str | None:
        if not gemini.is_ready:
            return None

        if self.state in (State.GREETING, State.COLLECT_NAME) and self._looks_like_name(text):
            return None

        self._undo_obvious_rule_mistakes(text)

        if self.state not in (
            State.CHATTING,
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
        if _is_chatting_state(self.state):
            missing = self._missing_fields()
            if missing:
                return FIELD_QUESTIONS_AR.get(missing[0], FIELD_QUESTIONS_AR["name"])
            return "كيف بقدر أساعدك بالحجز؟"
        mapping = {
            State.OFFER_GP_FALLBACK: "موافق/لا على الطب العام؟",
            State.CONFIRM: "هل بتحب تأكيد الموعد المعروض؟",
            State.FINALIZED: "إذا بدك حجز جديد اكتب: حجز موعد جديد 📅",
            State.CANCELLED: "إذا بدك حجز جديد اكتب: حجز موعد جديد 📅",
            State.WAITLISTED: "إذا بدك حجز جديد اكتب: حجز موعد جديد 📅",
        }
        return mapping.get(self.state, "كيف بقدر أساعدك بالحجز؟")

    def _current_ui_action(self) -> UIAction | None:
        mapping = {
            State.OFFER_GP_FALLBACK: UIAction.NONE,
            State.CONFIRM: UIAction.NONE,
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
            "موافق؟"
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

        header = self.data.pop("custom_reply", None)
        prefix = f"{header}\n\n" if header else "وجدت موعداً مناسباً! 📅\n\n"

        return self._reply(
            f"{prefix}"
            f"📆 {dt}\n"
            f"🏥 التخصص: {self.data.get('specialty_ar', '')}\n"
            f"👨‍⚕️ الطبيب: {self.slot.get('doctor_name') or '—'}\n"
            f"🏢 العيادة: {self.slot.get('clinic_name') or '—'} ({self.slot.get('clinic_code') or '—'})\n"
            f"{alt_hint}\n"
            f"مناسبلك؟",
            UIAction.NONE,
        )

    def _is_clarification_request(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        return bool(lowered) and any(word in lowered for word in [
            "أنت فهمت", "انت فهمت", "ماذا فهمت", "إيه اللي فهمته", "ايش فهمت",
            "أشرح", "اشرح", "what did you understand", "what do you know",
        ])

    def _is_new_booking_request(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        return any(
            token in lowered
            for token in [
                "حجز موعد",
                "موعد جديد",
                "ابدأ",
                "من جديد",
                "restart",
                "book",
                "كرر",
                "تكرار",
                "من اول",
                "ابدأ من جديد",
                "بدء جديد",
            ]
        )

    def _terminal_meta_reply(self, raw: str, norm: str) -> str | None:
        """Rule-based answers after booking flow ends — works without LLM."""
        from config import CLINIC_NAME, DASHBOARD_HOST, DASHBOARD_PORT

        if not raw.strip():
            return None

        if _is_bot_meta_question(raw) or any(p in norm for p in ("وظيف", "شغل", "دورك", "بتعمل", "مساعد")):
            return (
                f"أنا مساعد حجز المواعيد في {CLINIC_NAME} 🏥 — "
                "بساعدك تحجز موعد، تستعلم عن حجزك، أو تلغيه. "
                "لحجز جديد اكتب: حجز موعد جديد 📅"
            )

        if any(p in norm for p in ("اداره", "ادارة", "لوحه", "لوحة", "dashboard", "admin", "الادارة")):
            dashboard_url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
            return (
                f"لوحة إدارة العيادة: {dashboard_url}\n"
                "منها بتقدر تشوف المواعيد والمرضى والأطباء.\n"
                "وللحجز من هون اكتب: حجز موعد جديد 📅"
            )

        if any(p in norm for p in ("تواصل", "اتصل", "رقم", "هاتف", "موقع", "عنوان")):
            return (
                "📞 للتواصل مع العيادة اكتب رسالتك هنا مباشرة، "
                "وسيتم حفظها في سجل المحادثات."
            )

        if _looks_like_frustration_question(norm, raw):
            if self.state == State.FINALIZED:
                return (
                    "آسف على اللبس! 🙏 تم تأكيد حجزك سابقاً — "
                    "للاستعلام عن موعدك اكتب «شو موعدي» أو «استعلام عن موعد»، "
                    "ولحجز جديد: حجز موعد جديد 📅"
                )
            if self.state == State.WAITLISTED:
                return (
                    "آسف على اللبس! 🙏 أنت بقائمة الانتظار حالياً — "
                    "رح نتواصل معك لما يتوفر موعد. "
                    "للاستعلام اكتب «شو موعدي»."
                )
            if self.state == State.CANCELLED:
                return (
                    "آسف على اللبس! 🙏 انتهى طلب الحجز السابق (ملغي أو ما اكتمل). "
                    "إذا بدك تحجز من جديد اكتب: حجز موعد جديد 📅"
                )

        return None

    def _format_stored_appointment_reply(self) -> str:
        from database.db import get_db
        from database import crud

        with get_db() as db:
            appt = crud.get_latest_patient_appointment(db, self.user_id)

        if not appt:
            return (
                "ما في موعد مسجل باسمك حالياً. 📋\n"
                "المواعيد المتاحة بالعيادة بتظهر لما تبدأ حجز جديد — اكتب: حجز موعد جديد 📅"
            )

        date_text = (
            appt.appt_datetime.strftime("%A، %d/%m/%Y — %H:%M")
            if appt.appt_datetime
            else "قائمة الانتظار"
        )
        status_ar = {
            "confirmed": "مؤكد",
            "waitlisted": "قائمة انتظار",
            "completed": "مكتمل",
            "no_show": "غياب",
            "cancelled": "ملغي",
        }.get(appt.status, appt.status)
        specialty = appt.specialty_ar or appt.specialty or "—"
        doctor = appt.slot.doctor if appt.slot and appt.slot.doctor else None
        return (
            "📌 موعدك المسجل:\n"
            f"رقم الحجز: {appt.appt_id}\n"
            f"الوقت: {date_text}\n"
            f"التخصص: {specialty}\n"
            f"الطبيب: {doctor.name if doctor else '—'}\n"
            f"الحالة: {status_ar}\n\n"
            "لحجز موعد جديد اكتب: حجز موعد جديد 📅"
        )

    def _handle_terminal_state(self, text: str) -> tuple[str, UIAction, dict] | None:
        """Reply when FSM is done (FINALIZED/CANCELLED/WAITLISTED). None = fall through after reset."""
        raw = (text or "").strip()
        norm = normalize(raw)

        if self.state == State.CANCELLED and (
            _looks_like_decline(norm) or _matches_any_token(norm, CANCEL_WORDS)
        ):
            return self._reply(
                "تمام، ما في مشكلة. إذا احتجت أي شي لاحقاً أنا هون. 👋",
                UIAction.SHOW_MAIN_MENU,
            )

        meta = self._terminal_meta_reply(raw, norm)
        if meta:
            return self._reply(meta, UIAction.SHOW_MAIN_MENU)

        if _looks_like_appointment_inquiry(norm):
            return self._reply(self._format_stored_appointment_reply(), UIAction.SHOW_MAIN_MENU)

        if GeminiClient.looks_like_question(raw):
            if self.state == State.FINALIZED:
                hint = "تم تأكيد حجزك. "
            elif self.state == State.WAITLISTED:
                hint = "أنت بقائمة الانتظار. "
            else:
                hint = "انتهى طلب الحجز السابق. "
            return self._reply(
                f"{hint}للاستعلام عن موعدك اكتب «شو موعدي»، "
                "ولحجز جديد: حجز موعد جديد 📅",
                UIAction.SHOW_MAIN_MENU,
            )

        return self._reply(
            "تم إنهاء الطلب السابق. إذا بدك حجز جديد اكتب: حجز موعد جديد 📅",
            UIAction.SHOW_MAIN_MENU,
        )

    def _reset(self) -> None:
        self.state = State.CHATTING
        self.data.clear()
        self.chat_history = []
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
            "data_json": {**self.data, "_chat_history": self.chat_history},
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
        fsm.state = _normalize_loaded_state(State[row.state])
        fsm.data = dict(row.data_json or {})
        fsm.chat_history = fsm.data.pop("_chat_history", []) or []
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
