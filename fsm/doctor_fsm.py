"""
fsm/doctor_fsm.py — Doctor session data-entry finite state machine.
Doctor speaks → Whisper → extract structured fields → review → save.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from nlp.normalizer import normalize
import re


class DoctorState(Enum):
    IDLE      = auto()
    LISTENING = auto()   # waiting for voice/text session note
    REVIEW    = auto()   # showing extracted summary for confirmation
    EDITING   = auto()   # doctor correcting a specific field
    SAVED     = auto()


CONFIRM_WORDS = {"تاكيد", "تأكيد", "صح", "تمام", "نعم", "ايوه", "احفظ"}

EDITABLE_FIELDS = {
    "اسم المريض":    "patient_name",
    "الشكوى":        "chief_complaint",
    "التشخيص":       "diagnosis",
    "الدواء":        "medications",
    "الفحوصات":      "investigations",
    "المتابعة":      "followup_days",
}


@dataclass
class DoctorFSM:
    doctor_id:  int
    telegram_id: int
    state:      DoctorState = DoctorState.IDLE
    session:    dict        = field(default_factory=dict)

    # ── Public entry ──────────────────────────────────────────────────────────

    async def handle(self, text: str, is_voice: bool = False) -> str:
        norm = normalize(text)
        command = (text or "").strip().lower()

        if command in {"/session", "session"} and self.state in (DoctorState.IDLE, DoctorState.SAVED):
            self.state = DoctorState.LISTENING
            self.session = {}
            return (
                "🎙️ أرسل ملاحظات الجلسة بصوتك أو نصاً.\n"
                "مثال: 'المريض اسمه أحمد، شاكي من ألم ركبة، إيبوبروفين 400 مرتين، متابعة بعد أسبوعين'"
            )

        if self.state == DoctorState.IDLE:
            self.state = DoctorState.LISTENING
            return (
                "🎙️ أرسل ملاحظات الجلسة بصوتك أو نصاً.\n"
                "مثال: 'المريض اسمه أحمد، شاكي من ألم ركبة، إيبوبروفين 400 مرتين، متابعة بعد أسبوعين'"
            )

        if self.state == DoctorState.LISTENING:
            return await self._process_session_note(text)

        if self.state == DoctorState.REVIEW:
            return await self._handle_review_input(norm, text)

        if self.state == DoctorState.EDITING:
            return self._apply_edit(text)

        return "أرسل /session لبدء تسجيل جلسة جديدة."

    # ── Session note processing ───────────────────────────────────────────────

    async def _process_session_note(self, text: str) -> str:
        from nlp.doctor_extractor import extract_session_fields
        self.session = extract_session_fields(text)
        self.session["raw_transcription"] = text
        self.state = DoctorState.REVIEW
        return self._format_summary()

    def _format_summary(self) -> str:
        s = self.session
        meds = "\n".join(
            f"  • {m['name']} {m.get('dose','')} — {m.get('frequency','')} — {m.get('duration','')}"
            for m in s.get("medications", [])
        ) or "  لا يوجد"

        invs = "\n".join(
            f"  • {i.get('name_ar', i)}"
            for i in s.get("investigations", [])
        ) or "  لا يوجد"

        return (
            "📋 *ملخص الجلسة — يرجى المراجعة:*\n\n"
            f"👤 المريض: {s.get('patient_name') or '❓'}\n"
            f"🩺 الشكوى: {s.get('chief_complaint') or '❓'}\n"
            f"⏱ المدة: {s.get('symptom_duration') or '—'}\n"
            f"🔬 التشخيص: {s.get('diagnosis') or '—'}\n"
            f"💊 الأدوية:\n{meds}\n"
            f"🔭 الفحوصات:\n{invs}\n"
            f"📅 متابعة: بعد {s.get('followup_days') or '?'} يوم\n\n"
            "✅ اكتب *تأكيد* للحفظ\n"
            "✏️ أو اسم الحقل لتعديله (مثال: `الشكوى: وجع ظهر`)"
        )

    # ── Review / edit ─────────────────────────────────────────────────────────

    async def _handle_review_input(self, norm: str, raw: str) -> str:
        if any(w in norm for w in CONFIRM_WORDS):
            return await self._save_session()

        for label, field_key in EDITABLE_FIELDS.items():
            if label in raw and ":" in raw:
                return self._apply_edit(raw)

        return "اكتب *تأكيد* للحفظ، أو الحقل الذي تريد تعديله."

    def _apply_edit(self, text: str) -> str:
        for label, field_key in EDITABLE_FIELDS.items():
            if label in text and ":" in text:
                value = text.split(":", 1)[-1].strip()
                self.session[field_key] = self._coerce_field_value(field_key, value)
                self.state = DoctorState.REVIEW
                return f"✏️ تم تحديث *{label}*.\n\n" + self._format_summary()
        self.state = DoctorState.REVIEW
        return "ما عرفت أي حقل تقصد. " + self._format_summary()

    def _coerce_field_value(self, field_key: str, value: str):
        if field_key == "followup_days":
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if field_key == "medications":
            items = [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
            return [{"name": item} for item in items] or value
        if field_key == "investigations":
            items = [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
            return [{"name_ar": item} for item in items] or value
        return value

    async def _save_session(self) -> str:
        from database.db import get_db
        from database import crud
        with get_db() as db:
            saved = crud.create_session(db, self.session, doctor_id=self.doctor_id)
            linked_patient = bool(saved.patient_id)
            linked_appointment = bool(saved.appointment_id)

        self.state = DoctorState.SAVED
        self.session = {}

        link_note = []
        if linked_patient:
            link_note.append("تم ربطها بملف المريض")
        if linked_appointment:
            link_note.append("تم ربطها بالموعد وتحديثه كمكتمل")

        suffix = "\n🔗 " + "، و".join(link_note) if link_note else "\n⚠️ حُفظت الجلسة بدون ربط تلقائي؛ تأكدي من اسم المريض أو appointment_id."
        return "✅ تم حفظ الجلسة بنجاح!" + suffix + "\nأرسل /session لتسجيل جلسة جديدة."

    def to_snapshot(self) -> dict:
        return {
            "state": self.state.name,
            "data_json": {"session": self.session},
        }

    @classmethod
    def from_snapshot(cls, doctor_id: int, telegram_id: int, row) -> DoctorFSM:
        fsm = cls(doctor_id=doctor_id, telegram_id=telegram_id)
        fsm.state = DoctorState[row.state]
        data = row.data_json or {}
        fsm.session = data.get("session") or {}
        return fsm

    def discard(self) -> None:
        self.state = DoctorState.IDLE
        self.session = {}
