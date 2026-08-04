"""
End-to-end tests through Telegram handler mocks.

Exercises the real handler → FSM → CRUD stack on an in-memory SQLite DB.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import select

from bot.handlers import doctor as doctor_handler
from bot.handlers import patient as patient_handler
from database import crud
from database.models import Appointment, Patient, Session as DoctorSession, Slot
from fsm.doctor_fsm import DoctorState
from fsm.patient_fsm import PatientFSM, State
from nlp.gemini_client import gemini
from scheduler.priority import PriorityResult
from tests.helpers import make_test_engine, make_test_session, run_async, seed_doctor, seed_patient, seed_slot, use_test_db
from tests.telegram_mocks import last_reply_text, make_context, make_start_update, make_text_update
from utils.datetime_utils import utcnow


PATIENT_ID = 910_001
DOCTOR_TG_ID = 920_001


@pytest.fixture
def patient_e2e_db():
    engine = make_test_engine()
    db = make_test_session(engine)
    crud.delete_fsm_session(db, PATIENT_ID, role="patient")
    doctor = seed_doctor(
        db,
        specialty="general_practice",
        clinic_code="CLINIC-E2E-GP",
        clinic_name="عيادة E2E",
    )
    slot_when = (utcnow() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    slot = seed_slot(db, doctor, when=slot_when, priority_class="P3")
    db.commit()
    yield {"engine": engine, "db": db, "doctor": doctor, "slot": slot, "slot_when": slot_when}
    db.close()
    engine.dispose()


def _send_patient(db, text: str, user_id: int = PATIENT_ID):
    update, message = make_text_update(user_id, text)
    with use_test_db(db):
        run_async(patient_handler.handle_text(update, make_context()))
    return message


@patch("scheduler.classifier.classify_specialty", side_effect=lambda *a, **k: {
    "specialty": "general_practice",
    "specialty_ar": "الطب العام",
    "method": "rule",
    "confidence": 0.95,
})
@patch.object(gemini, "_available", False)
@patch("bot.handlers.patient.TTS_ENABLED", False)
def test_full_booking_flow_creates_confirmed_appointment(_classify, patient_e2e_db):
    db = patient_e2e_db["db"]
    slot = patient_e2e_db["slot"]

    update, _ = make_start_update(PATIENT_ID)
    with use_test_db(db):
        run_async(patient_handler.handle_start(update, make_context()))

    _send_patient(db, "أحمد محمود")
    _send_patient(db, "أحمد محمود")
    _send_patient(db, "صداع خفيف من يومين")
    _send_patient(db, "🟢 روتيني / عادي")
    confirm_msg = _send_patient(db, "بكرا")
    assert "وجدت موعد" in last_reply_text(confirm_msg)

    with use_test_db(db):
        fsm = patient_handler._get_fsm(PATIENT_ID)
    assert fsm.state == State.CONFIRM

    final_msg = _send_patient(db, "✅ تأكيد الحجز")
    assert "تم تأكيد حجزك" in last_reply_text(final_msg)
    with use_test_db(db):
        fsm = patient_handler._get_fsm(PATIENT_ID)
    assert fsm.state == State.FINALIZED

    appt = db.scalar(
        select(Appointment)
        .join(Patient, Appointment.patient_id == Patient.patient_id)
        .where(Patient.telegram_id == PATIENT_ID)
        .order_by(Appointment.created_at.desc())
    )
    assert appt is not None
    assert appt.status == "confirmed"
    assert appt.slot_id == slot.slot_id
    assert db.get(Slot, slot.slot_id).status == "booked"
    assert crud.get_profile(db, PATIENT_ID).get("name") == "أحمد محمود"


@patch.object(gemini, "_available", False)
@patch("bot.handlers.patient.TTS_ENABLED", False)
def test_confirm_cancel_does_not_cancel_db_appointment(patient_e2e_db, clear_patient_fsm):
    """❌ إلغاء in CONFIRM must not call cancel_latest_patient_appointment."""
    db = patient_e2e_db["db"]
    slot = patient_e2e_db["slot"]
    slot_when = patient_e2e_db["slot_when"]

    patient = seed_patient(db, telegram_id=PATIENT_ID, name="ليلى")
    existing_slot = seed_slot(
        db,
        patient_e2e_db["doctor"],
        when=(utcnow() + timedelta(days=5)).replace(hour=12, minute=0, second=0, microsecond=0),
        priority_class="P3",
        status="booked",
    )
    db.add(
        Appointment(
            appt_id="appt-existing-e2e",
            patient_id=patient.patient_id,
            slot_id=existing_slot.slot_id,
            appt_datetime=existing_slot.slot_datetime,
            specialty="general_practice",
            status="confirmed",
        )
    )
    db.commit()
    clear_patient_fsm(PATIENT_ID)

    fsm = PatientFSM(user_id=PATIENT_ID)
    fsm.state = State.CONFIRM
    fsm.data = {
        "name": "ليلى",
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.3,
        "time_pref": {"date": str(slot_when.date()), "phrase": "بكرا"},
        "specialty_hint": "general_practice",
        "specialty_ar": "الطب العام",
        "priority_class": "P3",
    }
    fsm.slot = {
        "slot_id": slot.slot_id,
        "slot_datetime": slot_when,
        "doctor_name": "د. اختبار",
        "clinic_name": "عيادة E2E",
    }
    fsm.priority = PriorityResult(score=0.3, priority_class="P3", label_ar="روتيني", label_color="green", breakdown={})
    with use_test_db(db):
        crud.upsert_fsm_session(
            db,
            PATIENT_ID,
            role="patient",
            state=fsm.state.name,
            data_json=fsm.data,
            slot_json={**fsm.slot, "slot_datetime": fsm.slot["slot_datetime"].isoformat()},
            priority_json={"priority_class": "P3", "score": 0.3, "label_ar": "روتيني", "breakdown": {}},
        )

    with patch.object(crud, "cancel_latest_patient_appointment") as mock_cancel:
        msg = _send_patient(db, "❌ إلغاء")
        mock_cancel.assert_not_called()

    assert "تم الإلغاء" in last_reply_text(msg)
    with use_test_db(db):
        reloaded = patient_handler._get_fsm(PATIENT_ID)
    assert reloaded.state == State.CANCELLED
    assert db.scalar(select(Appointment).where(Appointment.appt_id == "appt-existing-e2e")).status == "confirmed"


@patch("scheduler.classifier.classify_specialty", side_effect=lambda *a, **k: {
    "specialty": "general_practice",
    "specialty_ar": "الطب العام",
    "method": "rule",
    "confidence": 0.95,
})
@patch.object(gemini, "_available", False)
@patch("bot.handlers.patient.TTS_ENABLED", False)
def test_natural_language_cancel_after_finalize(_classify, patient_e2e_db):
    """«أريد الغاء الموعد» must cancel a confirmed DB appointment when FSM is FINALIZED."""
    db = patient_e2e_db["db"]
    slot = patient_e2e_db["slot"]

    update, _ = make_start_update(PATIENT_ID)
    with use_test_db(db):
        run_async(patient_handler.handle_start(update, make_context()))

    _send_patient(db, "ليلى")
    _send_patient(db, "صداع")
    _send_patient(db, "🟢 روتيني / عادي")
    _send_patient(db, "بكرا")
    _send_patient(db, "✅ تأكيد الحجز")

    with use_test_db(db):
        fsm = patient_handler._get_fsm(PATIENT_ID)
    assert fsm.state == State.FINALIZED

    msg = _send_patient(db, "اريد الغاء الموعد")
    assert "تم إلغاء آخر موعد" in last_reply_text(msg)

    appt = db.scalar(
        select(Appointment)
        .join(Patient, Appointment.patient_id == Patient.patient_id)
        .where(Patient.telegram_id == PATIENT_ID)
        .order_by(Appointment.created_at.desc())
    )
    assert appt is not None
    assert appt.status == "cancelled"
    assert db.get(Slot, slot.slot_id).status == "available"


@patch.object(gemini, "_available", False)
@patch("bot.handlers.patient.TTS_ENABLED", False)
def test_session_persists_across_handler_reload(patient_e2e_db):
    db = patient_e2e_db["db"]

    update, _ = make_start_update(PATIENT_ID)
    with use_test_db(db):
        run_async(patient_handler.handle_start(update, make_context()))
    _send_patient(db, "سارة محمود")
    _send_patient(db, "عندي صداع")

    with use_test_db(db):
        row = crud.get_fsm_session(db, PATIENT_ID, role="patient")
    assert row is not None
    assert row.state == State.COLLECT_URGENCY.name

    with use_test_db(db):
        reloaded = patient_handler._get_fsm(PATIENT_ID)
    assert reloaded.state == State.COLLECT_URGENCY
    assert reloaded.data.get("name") == "سارة محمود"


@pytest.fixture
def doctor_e2e_db():
    engine = make_test_engine()
    db = make_test_session(engine)
    crud.delete_fsm_session(db, DOCTOR_TG_ID, role="doctor")
    doctor = seed_doctor(
        db,
        specialty="general_practice",
        clinic_code="CLINIC-E2E-DOC",
        telegram_id=DOCTOR_TG_ID,
        name="د. E2E",
    )
    seed_patient(db, telegram_id=880_001, name="ليلى أبو علي")
    db.commit()
    yield {"engine": engine, "db": db, "doctor": doctor}
    db.close()
    engine.dispose()


def _send_doctor(db, text: str, user_id: int = DOCTOR_TG_ID):
    update, message = make_text_update(user_id, text)
    with use_test_db(db):
        run_async(doctor_handler.handle_doctor_text(update, make_context()))
    return message


@patch("bot.handlers.patient.TTS_ENABLED", False)
def test_doctor_session_note_saved_and_linked_to_patient(doctor_e2e_db):
    db = doctor_e2e_db["db"]
    doctor = doctor_e2e_db["doctor"]
    note = (
        "المريض اسمه ليلى أبو علي، شاكي من سعال، "
        "التشخيص التهاب bronchi، ايبوبروفين 400 مرتين، متابعة بعد اسبوع"
    )
    _send_doctor(db, "/session")
    review_msg = _send_doctor(db, note)
    assert "ملخص الجلسة" in last_reply_text(review_msg)

    with use_test_db(db):
        fsm = doctor_handler._get_fsm(doctor)
    assert fsm.state == DoctorState.REVIEW

    save_msg = _send_doctor(db, "تأكيد")
    assert "تم حفظ الجلسة" in last_reply_text(save_msg)
    with use_test_db(db):
        row = crud.get_fsm_session(db, DOCTOR_TG_ID, role="doctor")
    assert row.state == DoctorState.SAVED.name

    session = db.scalar(
        select(DoctorSession)
        .where(DoctorSession.doctor_id == doctor.doctor_id)
        .order_by(DoctorSession.session_id.desc())
    )
    assert session is not None
    assert session.patient_id is not None
    assert session.patient_name == "ليلى أبو علي"
    assert session.chief_complaint is not None


@pytest.fixture
def router_e2e_db(clear_patient_fsm):
    engine = make_test_engine()
    db = make_test_session(engine)
    clear_patient_fsm(PATIENT_ID)
    crud.delete_fsm_session(db, DOCTOR_TG_ID, role="doctor")
    seed_doctor(db, clinic_code="CLINIC-RT-DOC", telegram_id=DOCTOR_TG_ID)
    db.commit()
    yield db
    db.close()
    engine.dispose()


@patch.object(gemini, "_available", False)
@patch("bot.handlers.patient.TTS_ENABLED", False)
def test_route_start_patient_gets_booking_greeting(router_e2e_db):
    from bot.router import route_start

    update, message = make_start_update(PATIENT_ID)
    with use_test_db(router_e2e_db):
        run_async(route_start(update, make_context()))
    assert "مساعد الحجز" in last_reply_text(message)


@patch("bot.handlers.patient.TTS_ENABLED", False)
def test_route_start_doctor_gets_doctor_greeting(router_e2e_db):
    from bot.router import route_start

    update, message = make_start_update(DOCTOR_TG_ID)
    with use_test_db(router_e2e_db):
        run_async(route_start(update, make_context()))
    assert "دكتور" in last_reply_text(message)
