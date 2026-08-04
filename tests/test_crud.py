"""Unit tests for database/crud.py — patients, slots, booking, sessions."""
from datetime import timedelta

import pytest
from sqlalchemy import select

from database import crud
from database.models import Appointment, PatientProfile, Slot
from tests.helpers import seed_doctor, seed_patient, seed_slot
from utils.datetime_utils import utcnow


def _booking_data(**overrides):
    base = {
        "name": "أحمد محمود",
        "complaint": {"raw": "صداع", "category": "general", "urgency_score": 0.3},
        "urgency_score": 0.4,
        "time_pref": {"date": None, "phrase": "أي وقت"},
        "specialty_hint": "general_practice",
        "specialty_ar": "الطب العام",
        "priority_class": "P3",
        "priority_score": 0.35,
    }
    base.update(overrides)
    return base


def test_get_or_create_patient_creates_then_updates_name(db_session):
    p1 = crud.get_or_create_patient(db_session, telegram_id=1001, name="أحمد")
    db_session.commit()
    p2 = crud.get_or_create_patient(db_session, telegram_id=1001, name="أحمد محمود")
    db_session.commit()

    assert p1.patient_id == p2.patient_id
    assert p2.name == "أحمد محمود"


def test_search_patient_by_partial_name(db_session):
    crud.get_or_create_patient(db_session, telegram_id=1002, name="سارة خالد")
    db_session.commit()
    results = crud.search_patient(db_session, "سارة")
    assert len(results) == 1
    assert "سارة" in results[0].name


def test_find_next_available_slot_returns_future_slot(db_session):
    doctor = seed_doctor(db_session, specialty="general_practice", clinic_code="CLINIC-GP-CRUD")
    future = utcnow() + timedelta(days=2)
    slot = seed_slot(db_session, doctor, when=future.replace(hour=9, minute=0), priority_class="P3")

    found = crud.find_next_available_slot(
        db_session,
        specialty="general_practice",
        priority_class="P3",
        telegram_id=9999,
    )
    assert found is not None
    assert found.slot_id == slot.slot_id


def test_find_slot_skips_booked_slots(db_session):
    doctor = seed_doctor(db_session, specialty="general_practice", clinic_code="CLINIC-GP-CRUD2")
    future = utcnow() + timedelta(days=2)
    slot = seed_slot(db_session, doctor, when=future.replace(hour=9, minute=0), priority_class="P3")
    slot.status = "booked"
    db_session.commit()

    found = crud.find_next_available_slot(
        db_session,
        specialty="general_practice",
        priority_class="P3",
    )
    assert found is None


def test_create_patient_file_and_book_success(db_session):
    doctor = seed_doctor(db_session, specialty="general_practice", clinic_code="CLINIC-GP-BOOK")
    when = (utcnow() + timedelta(days=3)).replace(hour=11, minute=0, second=0, microsecond=0)
    slot = seed_slot(db_session, doctor, when=when, priority_class="P3")

    result = crud.create_patient_file_and_book(
        db_session,
        telegram_id=2001,
        data=_booking_data(),
        slot_id=slot.slot_id,
    )
    assert result["patient"] is not None
    assert result["appointment"] is not None
    assert result["slot_conflict"] is False
    assert result["booking_conflict"] is None

    appt = result["appointment"]
    assert appt.status == "confirmed"
    assert appt.slot_id == slot.slot_id
    assert db_session.get(Slot, slot.slot_id).status == "booked"

    profile = db_session.scalar(select(PatientProfile).where(PatientProfile.telegram_id == 2001))
    assert profile.data.get("name") == "أحمد محمود"


def test_booking_detects_slot_conflict(db_session):
    doctor = seed_doctor(db_session, specialty="general_practice", clinic_code="CLINIC-GP-BOOK2")
    when = (utcnow() + timedelta(days=3)).replace(hour=11, minute=0, second=0, microsecond=0)
    slot = seed_slot(db_session, doctor, when=when, priority_class="P3")
    slot.status = "booked"
    db_session.commit()

    result = crud.create_patient_file_and_book(
        db_session,
        telegram_id=2002,
        data=_booking_data(),
        slot_id=slot.slot_id,
    )
    assert result["slot_conflict"] is True
    assert result["appointment"] is None


def test_waitlist_creates_appointment_without_slot(db_session):
    result = crud.create_waitlist_appointment(
        db_session,
        telegram_id=2003,
        data=_booking_data(priority_class="P2"),
    )
    assert result["appointment"] is not None
    assert result["appointment"].status == "waitlisted"
    assert result["appointment"].slot_id is None


def test_time_overlap_conflict_blocks_second_booking(db_session):
    doctor = seed_doctor(db_session, specialty="cardiology", clinic_code="CLINIC-CARD-BOOK")
    patient = seed_patient(db_session, telegram_id=3001)
    when = (utcnow() + timedelta(days=4)).replace(hour=10, minute=0, second=0, microsecond=0)
    slot1 = seed_slot(db_session, doctor, when=when)
    slot2 = seed_slot(db_session, doctor, when=when + timedelta(minutes=15))

    first = crud.reserve_slot_and_create_appointment(
        db_session,
        _booking_data(specialty_hint="cardiology"),
        slot1.slot_id,
        patient,
    )
    db_session.commit()

    conflict = crud.find_patient_booking_conflict(
        db_session,
        patient.patient_id,
        slot2.slot_datetime,
        "cardiology",
    )
    assert conflict is not None
    assert conflict["type"] == "time_overlap"
    assert first["appointment"] is not None


def test_create_session_links_patient_by_name(db_session):
    doctor = seed_doctor(db_session, clinic_code="CLINIC-DOC-SESS")
    seed_patient(db_session, telegram_id=4001, name="ليلى أبو علي")

    session = crud.create_session(
        db_session,
        {
            "patient_name": "ليلى أبو علي",
            "chief_complaint": "سعال",
            "diagnosis": "التهاب bronchi",
            "medications": [{"name": "Paracetamol", "dose": "500mg"}],
            "followup_days": 7,
            "raw_transcription": "ملاحظة صوتية",
        },
        doctor_id=doctor.doctor_id,
    )
    assert session.patient_id is not None
    assert session.patient_name == "ليلى أبو علي"


def test_update_appointment_status_cancels_and_releases_slot(db_session):
    doctor = seed_doctor(db_session, clinic_code="CLINIC-DOC-SESS2")
    patient = seed_patient(db_session, telegram_id=4002)
    when = (utcnow() + timedelta(days=5)).replace(hour=14, minute=0, second=0, microsecond=0)
    slot = seed_slot(db_session, doctor, when=when)
    booking = crud.reserve_slot_and_create_appointment(
        db_session,
        _booking_data(),
        slot.slot_id,
        patient,
    )
    db_session.commit()
    appt_id = booking["appointment"].appt_id

    updated = crud.update_appointment_status(db_session, appt_id, "cancelled")
    assert updated.status == "cancelled"
    assert db_session.get(Slot, slot.slot_id).status == "available"


def test_upsert_profile_merges_data(db_session):
    crud.upsert_profile(db_session, telegram_id=5001, data={"name": "خالد", "last_complaint": "صداع"})
    crud.upsert_profile(db_session, telegram_id=5001, data={"last_specialty": "neurology"})
    db_session.commit()

    profile = crud.get_profile(db_session, 5001)
    assert profile["name"] == "خالد"
    assert profile["last_specialty"] == "neurology"


def test_log_message_creates_conversation(db_session):
    log = crud.log_message(db_session, 5002, "inbound", "text", "مرحبا")
    assert log.log_id is not None
    conversations = crud.get_conversations(db_session, limit=10)
    assert any(c.telegram_id == 5002 for c in conversations)


@pytest.fixture
def scheduling_policy_slots(db_session):
    doctor = seed_doctor(db_session, specialty="general_practice", clinic_code="CLINIC-POLICY")
    when = (utcnow() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    p1_slot = seed_slot(db_session, doctor, when=when, priority_class="P1")
    p3_slot = seed_slot(db_session, doctor, when=when.replace(hour=10), priority_class="P3")
    db_session.commit()
    return doctor, p1_slot, p3_slot


def test_p3_cannot_take_p1_reserved_slot(db_session, scheduling_policy_slots):
    _doctor, p1_slot, p3_slot = scheduling_policy_slots
    found = crud.find_next_available_slot(
        db_session,
        specialty="general_practice",
        priority_class="P3",
    )
    assert found is not None
    assert found.slot_id == p3_slot.slot_id
    assert found.slot_id != p1_slot.slot_id


def test_p1_can_take_p1_reserved_slot(db_session, scheduling_policy_slots):
    _doctor, p1_slot, _p3_slot = scheduling_policy_slots
    found = crud.find_next_available_slot(
        db_session,
        specialty="general_practice",
        priority_class="P1",
    )
    assert found.slot_id == p1_slot.slot_id


def test_same_specialty_same_day_conflict(db_session, scheduling_policy_slots):
    doctor, _p1_slot, p3_slot = scheduling_policy_slots
    patient = seed_patient(db_session, telegram_id=6001)
    first = crud.reserve_slot_and_create_appointment(
        db_session,
        _booking_data(specialty_hint="general_practice"),
        p3_slot.slot_id,
        patient,
    )
    db_session.commit()

    when = p3_slot.slot_datetime
    other_slot = seed_slot(
        db_session,
        doctor,
        when=when.replace(hour=15),
        priority_class="P3",
    )
    conflict = crud.find_patient_booking_conflict(
        db_session,
        patient.patient_id,
        other_slot.slot_datetime,
        "general_practice",
    )
    assert conflict is not None
    assert conflict["type"] == "same_specialty_same_day"
    assert first["appointment"] is not None


@pytest.mark.parametrize(
    "specialty,code",
    [
        ("gastroenterology", "CLINIC-GI-TEST"),
        ("chronic_diseases", "CLINIC-CHR-TEST"),
        ("elderly", "CLINIC-ELD-TEST"),
    ],
)
def test_new_specialty_doctors_have_available_slots(db_session, specialty, code):
    doctor = seed_doctor(db_session, specialty=specialty, clinic_code=code)
    when = (utcnow() + timedelta(days=7)).replace(hour=10, minute=0, second=0, microsecond=0)
    seed_slot(db_session, doctor, when=when, priority_class="P3")
    db_session.commit()

    slot = crud.find_next_available_slot(db_session, specialty=specialty, priority_class="P3")
    assert slot is not None
    assert slot.doctor.specialty == specialty
