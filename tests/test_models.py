"""Unit tests for database/models.py — schema, defaults, and relationships."""
from sqlalchemy import inspect

from database.models import (
    Appointment,
    Conversation,
    Doctor,
    MessageLog,
    Patient,
    PatientProfile,
    Session as DoctorSession,
    Slot,
)
from tests.helpers import seed_doctor, seed_patient, seed_slot


def test_all_expected_tables_exist(db_session):
    names = set(inspect(db_session.bind).get_table_names())
    expected = {
        "patients",
        "doctors",
        "appointments",
        "sessions",
        "slots",
        "conversations",
        "message_logs",
        "patient_profiles",
        "fsm_sessions",
    }
    assert expected.issubset(names)


def test_patient_defaults_and_unique_telegram(db_session):
    p1 = Patient(telegram_id=111, name="أحمد")
    p2 = Patient(telegram_id=222, name="سارة")
    db_session.add_all([p1, p2])
    db_session.commit()

    assert p1.patient_id is not None
    assert p1.created_at is not None
    assert p1.name == "أحمد"


def test_doctor_requires_clinic_identity(db_session):
    doctor = seed_doctor(db_session, specialty="cardiology", clinic_code="CLINIC-CARD-TEST")
    assert doctor.is_active
    assert doctor.specialty == "cardiology"
    assert doctor.telegram_id is None


def test_slot_belongs_to_doctor(db_session):
    doctor = seed_doctor(db_session)
    slot = seed_slot(db_session, doctor, status="available")
    assert slot.doctor_id == doctor.doctor_id
    assert slot.status == "available"
    assert slot.doctor.specialty == doctor.specialty


def test_appointment_links_patient_and_slot(db_session):
    patient = seed_patient(db_session, telegram_id=333)
    doctor = seed_doctor(db_session, specialty="neurology", clinic_code="CLINIC-NEURO-T")
    slot = seed_slot(db_session, doctor)
    appt = Appointment(
        appt_id="appt_test_001",
        patient_id=patient.patient_id,
        slot_id=slot.slot_id,
        appt_datetime=slot.slot_datetime,
        specialty=doctor.specialty,
        status="confirmed",
    )
    db_session.add(appt)
    db_session.commit()

    assert appt.patient.patient_id == patient.patient_id
    assert appt.slot.slot_id == slot.slot_id


def test_doctor_session_optional_patient_and_appointment(db_session):
    doctor = seed_doctor(db_session, clinic_code="CLINIC-GP-T2")
    session = DoctorSession(
        doctor_id=doctor.doctor_id,
        patient_name="مريض بدون ملف",
        chief_complaint="صداع",
        diagnosis="صداع توتري",
    )
    db_session.add(session)
    db_session.commit()

    assert session.patient_id is None
    assert session.appointment_id is None
    assert session.doctor.doctor_id == doctor.doctor_id


def test_patient_profile_json_payload(db_session):
    patient = seed_patient(db_session, telegram_id=444)
    profile = PatientProfile(
        patient_id=patient.patient_id,
        telegram_id=patient.telegram_id,
        data={"name": patient.name, "last_complaint": "ألم ظهر"},
    )
    db_session.add(profile)
    db_session.commit()

    loaded = db_session.get(PatientProfile, profile.profile_id)
    assert loaded.data["last_complaint"] == "ألم ظهر"
    assert loaded.patient.patient_id == patient.patient_id


def test_conversation_and_message_log(db_session):
    conversation = Conversation(telegram_id=555, role="patient")
    db_session.add(conversation)
    db_session.flush()

    log = MessageLog(
        conversation_id=conversation.conversation_id,
        telegram_id=555,
        direction="inbound",
        message_type="text",
        content="مرحبا",
    )
    db_session.add(log)
    db_session.commit()

    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == "مرحبا"
