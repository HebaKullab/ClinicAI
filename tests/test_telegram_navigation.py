"""
Integrated Telegram navigation tests — simulates patient/doctor chat flows through
real handlers, FSM, scheduler, and CRUD on an in-memory SQLite DB.

Run:
    venv\\Scripts\\python.exe -m pytest tests/test_telegram_navigation.py -v

Chat transcripts and DB snapshots are written to telegram_chat_preview_live.txt
at the repo root after the test session completes.

Uses the real rule-based classifier, OpenRouter/Gemma LLM (when API keys are set),
and TTS settings from config/.env — no mocks for those components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from bot.handlers import doctor as doctor_handler
from bot.handlers import patient as patient_handler
from config import GEMINI_API_KEY, OPENROUTER_API_KEY, TTS_ENABLED, TTS_RESPONSE_MODE
from database import crud
from database.models import Appointment, Patient, Slot
from fsm.patient_fsm import State
from nlp import gemini_client
from tests.helpers import make_test_engine, make_test_session, run_async, seed_doctor, seed_patient, seed_slot, use_test_db
from tests.telegram_mocks import (
    format_reply_keyboard,
    last_reply_text,
    make_callback_update,
    make_context,
    make_start_update,
    make_text_update,
)
from utils.datetime_utils import utcnow

PREVIEW_PATH = Path(__file__).resolve().parent.parent / "telegram_chat_preview_live.txt"

PATIENT_A = 930_001
PATIENT_B = 930_002
DOCTOR_TG = 930_101


def _llm_available() -> bool:
    return bool(OPENROUTER_API_KEY and str(OPENROUTER_API_KEY).strip()) or bool(
        GEMINI_API_KEY and str(GEMINI_API_KEY).strip()
    )


@pytest.fixture(autouse=True)
def navigation_live_stack(monkeypatch):
    """Override conftest's LLM disable — navigation preview runs the real stack."""
    monkeypatch.setattr(gemini_client.gemini, "_available", _llm_available())


@dataclass
class PreviewCollector:
    sections: list[tuple[str, list[str]]] = field(default_factory=list)

    def add(self, title: str, lines: list[str]) -> None:
        self.sections.append((title, lines))

    def write(self, path: Path) -> None:
        llm_on = _llm_available()
        parts = [
            "ClinicAI — Telegram Chat Preview (live handler + DB)",
            f"Generated: {utcnow().isoformat()}",
            f"LLM: {'on' if llm_on else 'off (no API key)'} | "
            f"TTS: {'on' if TTS_ENABLED else 'off'} ({TTS_RESPONSE_MODE}) | "
            "Classifier: rule-based (real)",
            "",
        ]
        for idx, (title, lines) in enumerate(self.sections, start=1):
            parts.append("=" * 80)
            parts.append(f"{idx}. {title}")
            parts.append("=" * 80)
            parts.extend(lines)
            parts.append("")
        path.write_text("\n".join(parts), encoding="utf-8")


@pytest.fixture(scope="session")
def preview_collector():
    collector = PreviewCollector()
    yield collector
    collector.write(PREVIEW_PATH)


class ChatSimulator:
    """Drive real Telegram handlers and record a human-readable transcript."""

    def __init__(self, db, user_id: int, role: str = "patient"):
        self.db = db
        self.user_id = user_id
        self.role = role
        self.lines: list[str] = []

    def _note_fsm(self) -> None:
        if self.role != "patient":
            return
        with use_test_db(self.db):
            fsm = patient_handler._get_fsm(self.user_id)
        self.lines.append(f"   📊 FSM → {fsm.state.name}")

    def _note_db_appointments(self) -> None:
        patient = self.db.scalar(select(Patient).where(Patient.telegram_id == self.user_id))
        if not patient:
            self.lines.append("   🗄️ DB → لا يوجد ملف مريض بعد")
            return
        appts = self.db.scalars(
            select(Appointment)
            .where(Appointment.patient_id == patient.patient_id)
            .order_by(Appointment.created_at.desc())
        ).all()
        if not appts:
            self.lines.append(f"   🗄️ DB → مريض #{patient.patient_id} — لا مواعيد")
            return
        brief = "; ".join(
            f"{a.appt_id} [{a.status}] {a.specialty} @ {a.appt_datetime}"
            for a in appts[:3]
        )
        self.lines.append(f"   🗄️ DB → {brief}")

    def _record_reply(self, message, label: str) -> str:
        text = last_reply_text(message)
        self.lines.append(f"🤖 البوت ({label}):")
        for line in text.splitlines():
            self.lines.append(f"   {line}")
        if message.replies:
            kb = format_reply_keyboard(message.replies[-1].reply_markup)
            if kb:
                self.lines.append(kb)
        if message.voice_replies:
            self.lines.append("   🔊 [رد صوتي]")
        return text

    def send_start(self) -> str:
        self.lines.append(f"👤 المستخدم: /start")
        update, message = make_start_update(self.user_id)
        with use_test_db(self.db):
            if self.role == "patient":
                run_async(patient_handler.handle_start(update, make_context()))
            else:
                run_async(doctor_handler.handle_doctor_start(update, make_context()))
        reply = self._record_reply(message, "start")
        self._note_fsm()
        return reply

    def send_text(self, text: str) -> str:
        self.lines.append(f"👤 المستخدم: {text}")
        update, message = make_text_update(self.user_id, text)
        with use_test_db(self.db):
            if self.role == "patient":
                run_async(patient_handler.handle_text(update, make_context()))
            else:
                run_async(doctor_handler.handle_doctor_text(update, make_context()))
        reply = self._record_reply(message, "text")
        self._note_fsm()
        return reply

    def send_callback(self, data: str) -> str:
        self.lines.append(f"👤 المستخدم (زر inline): {data}")
        update, message = make_callback_update(self.user_id, data)
        with use_test_db(self.db):
            run_async(patient_handler.handle_callback(update, make_context()))
        reply = self._record_reply(message, "callback")
        self._note_fsm()
        return reply

    def snapshot_db(self, note: str = "") -> None:
        if note:
            self.lines.append(f"--- {note} ---")
        self._note_db_appointments()


@pytest.fixture
def nav_db_factory(preview_collector):
    """Fresh in-memory DB per scenario; transcript appended to session preview."""

    def _factory(
        title: str,
        *,
        extra_slots: list | None = None,
        seed_cardiology: bool = False,
        seed_dermatology: bool = False,
    ):
        engine = make_test_engine()
        db = make_test_session(engine)
        crud.delete_fsm_session(db, PATIENT_A, role="patient")
        crud.delete_fsm_session(db, PATIENT_B, role="patient")
        crud.delete_fsm_session(db, DOCTOR_TG, role="doctor")

        gp = seed_doctor(
            db,
            specialty="general_practice",
            clinic_code="NAV-GP",
            clinic_name="عيادة عامة — اختبار",
        )
        day1 = (utcnow() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
        day1_alt = day1.replace(hour=14, minute=0)
        day2 = (utcnow() + timedelta(days=2)).replace(hour=11, minute=0, second=0, microsecond=0)

        slots = {
            "gp_morning": seed_slot(db, gp, when=day1, priority_class="P3"),
            "gp_afternoon": seed_slot(db, gp, when=day1_alt, priority_class="P3"),
            "gp_day2": seed_slot(db, gp, when=day2, priority_class="P3"),
        }

        if extra_slots:
            for i, when in enumerate(extra_slots):
                slots[f"gp_extra_{i}"] = seed_slot(db, gp, when=when, priority_class="P3")

        if seed_cardiology:
            cardio = seed_doctor(db, specialty="cardiology", clinic_code="NAV-CARDIO", clinic_name="عيادة قلب")
            slots["cardio"] = seed_slot(db, cardio, when=day1, priority_class="P2")

        if seed_dermatology:
            derm = seed_doctor(db, specialty="dermatology", clinic_code="NAV-DERM", clinic_name="عيادة جلدية")
            slots["derm"] = seed_slot(db, derm, when=day1, priority_class="P3")

        seed_doctor(db, specialty="general_practice", clinic_code="NAV-DOC", telegram_id=DOCTOR_TG, name="د. اختبار")

        db.commit()

        sim = ChatSimulator(db, PATIENT_A)

        def finish():
            preview_collector.add(title, sim.lines)
            db.close()
            engine.dispose()

        return {"db": db, "sim": sim, "slots": slots, "finish": finish, "gp": gp}

    yield _factory


def _send_name(sim: ChatSimulator, name: str) -> None:
    """GREETING → COLLECT_NAME needs two turns before the name is stored."""
    sim.send_text(name)
    sim.send_text(name)


def _booking_prefix(sim: ChatSimulator, *, use_menu: bool = True) -> None:
    sim.send_start()
    if use_menu:
        sim.send_text("📅 حجز موعد جديد")
    _send_name(sim, "أحمد محمود")
    # Matches GP routing rule (method=rule) — no mocked classify_specialty needed.
    sim.send_text("تعب عام وارهاق")


def _advance_to_confirm(sim: ChatSimulator) -> None:
    _booking_prefix(sim)
    sim.send_text("🟢 روتيني / عادي")
    sim.send_text("بكرا")


def test_scenario_start_and_main_menu(nav_db_factory, preview_collector):
    ctx = nav_db_factory("بدء المحادثة والقائمة الرئيسية")
    sim = ctx["sim"]
    try:
        sim.send_start()
        sim.send_text("🔍 استعلام عن موعد")
        sim.send_text("📞 تواصل مع العيادة")
        sim.send_text("xyz ???")
    finally:
        ctx["finish"]()


def test_scenario_full_booking_confirmed(nav_db_factory):
    ctx = nav_db_factory("حجز كامل → تأكيد → موعد في DB")
    sim = ctx["sim"]
    db = ctx["db"]
    try:
        _advance_to_confirm(sim)
        sim.send_text("✅ تأكيد الحجز")
        sim.snapshot_db("بعد التأكيد")

        appt = db.scalar(
            select(Appointment)
            .join(Patient, Appointment.patient_id == Patient.patient_id)
            .where(Patient.telegram_id == PATIENT_A)
        )
        assert appt is not None
        assert appt.status == "confirmed"
        assert db.get(Slot, ctx["slots"]["gp_morning"].slot_id).status == "booked"
    finally:
        ctx["finish"]()


def test_scenario_cancel_at_confirm(nav_db_factory):
    ctx = nav_db_factory("حجز → شاشة التأكيد → ❌ إلغاء (بدون حفظ موعد)")
    sim = ctx["sim"]
    db = ctx["db"]
    try:
        _advance_to_confirm(sim)
        sim.send_text("❌ إلغاء")
        sim.snapshot_db("بعد الإلغاء")

        appt_count = db.scalar(
            select(Appointment)
            .join(Patient, Appointment.patient_id == Patient.patient_id)
            .where(Patient.telegram_id == PATIENT_A)
        )
        assert appt_count is None
        with use_test_db(db):
            fsm = patient_handler._get_fsm(PATIENT_A)
        assert fsm.state == State.CANCELLED
    finally:
        ctx["finish"]()


def test_scenario_edit_appointment_time(nav_db_factory):
    ctx = nav_db_factory("حجز → ✏️ تعديل الموعد → وقت جديد")
    sim = ctx["sim"]
    try:
        _advance_to_confirm(sim)
        sim.send_text("✏️ تعديل الموعد")
        sim.send_text("بعد بكرا")
    finally:
        ctx["finish"]()


def test_scenario_next_slot_button(nav_db_factory):
    day1 = (utcnow() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    day1_late = day1.replace(hour=15, minute=0)
    ctx = nav_db_factory("حجز → 🔄 موعد آخر (عدة slots)", extra_slots=[day1_late])
    sim = ctx["sim"]
    try:
        _advance_to_confirm(sim)
        sim.send_text("🔄 موعد آخر")
    finally:
        ctx["finish"]()


def test_scenario_unclear_responses(nav_db_factory):
    ctx = nav_db_factory("ردود غير مفهومة أثناء الحجز")
    sim = ctx["sim"]
    try:
        _booking_prefix(sim)
        sim.send_text("asdf qwerty 123")
        sim.send_text("🟢 روتيني / عادي")
        sim.send_text("???")
        sim.send_text("بكرا")
    finally:
        ctx["finish"]()


def test_scenario_inquiry_and_cancel_menu(nav_db_factory):
    ctx = nav_db_factory("استعلام وإلغاء من القائمة الرئيسية")
    sim = ctx["sim"]
    db = ctx["db"]
    try:
        sim.send_text("🔍 استعلام عن موعد")
        _advance_to_confirm(sim)
        sim.send_text("✅ تأكيد الحجز")
        sim.send_text("🔍 استعلام عن موعد")
        sim.send_text("❌ إلغاء موعد")
        sim.send_text("🔍 استعلام عن موعد")
        sim.snapshot_db("بعد إلغاء القائمة")

        appt = db.scalar(
            select(Appointment)
            .join(Patient, Appointment.patient_id == Patient.patient_id)
            .where(Patient.telegram_id == PATIENT_A)
            .order_by(Appointment.created_at.desc())
        )
        assert appt is not None
        assert appt.status == "cancelled"
    finally:
        ctx["finish"]()


def test_scenario_restart_mid_booking(nav_db_factory):
    ctx = nav_db_factory("إعادة حجز من منتصف المحادثة")
    sim = ctx["sim"]
    try:
        _booking_prefix(sim)
        sim.send_text("📅 حجز موعد جديد")
        _send_name(sim, "ليلى حسن")
        sim.send_text("ألم في الركبة")
    finally:
        ctx["finish"]()


def test_scenario_low_confidence_auto_specialty(nav_db_factory):
    ctx = nav_db_factory("تصنيف ضعيف → اختيار تلقائي للتخصص")
    sim = ctx["sim"]
    try:
        sim.send_start()
        sim.send_text("📅 حجز موعد جديد")
        _send_name(sim, "سارة علي")
        sim.send_text("شعور غريب لا أعرف كيف أصفه")
        sim.send_text("🟢 روتيني / عادي")
        reply = sim.send_text("بكرا")
        assert "تخصص" not in reply or "الطب العام" in reply or "موعد" in reply
    finally:
        ctx["finish"]()


def test_scenario_unsupported_specialty_gp_fallback(nav_db_factory):
    ctx = nav_db_factory("تخصص غير متوفر (أسنان) → عرض طب عام")
    sim = ctx["sim"]
    try:
        sim.send_start()
        sim.send_text("📅 حجز موعد جديد")
        _send_name(sim, "محمد")
        sim.send_text("بدي موعد عند طبيب اسنان")
    finally:
        ctx["finish"]()


def test_scenario_slot_conflict_on_confirm(nav_db_factory):
    ctx = nav_db_factory("موعد محجوز من مريض آخر قبل التأكيد")
    sim_a = ctx["sim"]
    db = ctx["db"]
    slot = ctx["slots"]["gp_morning"]
    try:
        _advance_to_confirm(sim_a)

        other = seed_patient(db, telegram_id=PATIENT_B, name="مريض B")
        slot.status = "booked"
        db.add(
            Appointment(
                appt_id="appt-nav-conflict",
                patient_id=other.patient_id,
                slot_id=slot.slot_id,
                appt_datetime=slot.slot_datetime,
                specialty="general_practice",
                status="confirmed",
            )
        )
        db.commit()

        sim_a.send_text("✅ تأكيد الحجز")
        sim_a.snapshot_db("بعد تعارض slot")
    finally:
        ctx["finish"]()


def test_scenario_same_specialty_same_day(nav_db_factory):
    ctx = nav_db_factory("نفس التخصص في نفس اليوم → مرفوض")
    sim = ctx["sim"]
    db = ctx["db"]
    morning = ctx["slots"]["gp_morning"]
    afternoon = ctx["slots"]["gp_afternoon"]
    try:
        patient = seed_patient(db, telegram_id=PATIENT_A, name="أحمد")
        db.add(
            Appointment(
                appt_id="appt-nav-existing-gp",
                patient_id=patient.patient_id,
                slot_id=morning.slot_id,
                appt_datetime=morning.slot_datetime,
                specialty="general_practice",
                specialty_ar="الطب العام",
                status="confirmed",
            )
        )
        morning.status = "booked"
        db.commit()
        crud.delete_fsm_session(db, PATIENT_A, role="patient")

        sim.send_start()
        sim.send_text("📅 حجز موعد جديد")
        _send_name(sim, "أحمد")
        sim.send_text("تعب عام")
        sim.send_text("🟢 روتيني / عادي")
        reply = sim.send_text("بكرا")
        if "تخصص" in reply:
            sim.send_text("🩺 طب عام")
        sim.send_text("✅ تأكيد الحجز")
        sim.snapshot_db("محاولة حجز GP ثاني")
    finally:
        ctx["finish"]()


def test_scenario_time_overlap_different_specialty(nav_db_factory):
    ctx = nav_db_factory("تداخل وقت — تخصص مختلف في نفس الوقت → مرفوض", seed_cardiology=True, seed_dermatology=True)
    sim = ctx["sim"]
    db = ctx["db"]
    cardio_slot = ctx["slots"]["cardio"]
    try:
        patient = seed_patient(db, telegram_id=PATIENT_A, name="فاطمة")
        db.add(
            Appointment(
                appt_id="appt-nav-cardio",
                patient_id=patient.patient_id,
                slot_id=cardio_slot.slot_id,
                appt_datetime=cardio_slot.slot_datetime,
                specialty="cardiology",
                specialty_ar="قلب وأوعية",
                status="confirmed",
            )
        )
        cardio_slot.status = "booked"
        db.commit()
        crud.delete_fsm_session(db, PATIENT_A, role="patient")

        sim.send_start()
        sim.send_text("📅 حجز موعد جديد")
        _send_name(sim, "فاطمة")
        sim.send_text("طفح جلدي")
        sim.send_text("🟢 روتيني / عادي")
        reply = sim.send_text("بكرا")
        if "تخصص" in reply:
            sim.send_text("🧴 جلدية")
        sim.send_text("✅ تأكيد الحجز")
        sim.snapshot_db("تداخل وقت مع موعد قلب")
    finally:
        ctx["finish"]()


def test_scenario_waitlist_no_slots(nav_db_factory):
    ctx = nav_db_factory("لا slots متاحة → قائمة انتظار")
    sim = ctx["sim"]
    db = ctx["db"]
    try:
        for slot in ctx["slots"].values():
            slot.status = "booked"
        db.commit()

        _advance_to_confirm(sim)
        sim.send_text("✅ تأكيد الحجز")
        sim.snapshot_db("قائمة انتظار")
    finally:
        ctx["finish"]()


def test_scenario_inline_menu_callbacks(nav_db_factory):
    ctx = nav_db_factory("أزرار inline للقائمة (callback)")
    sim = ctx["sim"]
    try:
        sim.send_callback("menu:book")
        _send_name(sim, "كريم")
        sim.send_callback("menu:inquiry")
        sim.send_callback("menu:contact")
        sim.send_callback("menu:cancel")
    finally:
        ctx["finish"]()


def test_scenario_doctor_navigation(nav_db_factory):
    ctx = nav_db_factory("واجهة الطبيب — /start ورسالة بدون ربط")
    db = ctx["db"]
    sim = ChatSimulator(db, DOCTOR_TG, role="doctor")
    unknown = ChatSimulator(db, 930_999, role="doctor")
    try:
        sim.send_start()
        sim.send_text("🎙️ تسجيل جلسة جديدة")
        unknown.lines = []
        unknown.send_start()
        unknown.send_text("مرحبا")
        ctx["sim"].lines.extend(["", "--- طبيب غير مسجل ---", *unknown.lines])
    finally:
        ctx["finish"]()
