"""Unit tests for fsm/doctor_fsm.py — doctor session note FSM."""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from fsm.doctor_fsm import EDITABLE_FIELDS, DoctorFSM, DoctorState
from tests.helpers import run_async


SAMPLE_SESSION = {
    "patient_name": "أحمد خالد",
    "chief_complaint": "ألم ركبة",
    "symptom_duration": "3 أيام",
    "diagnosis": "التهاب مفصل",
    "medications": [{"name": "Ibuprofen", "dose": "400mg", "frequency": "مرتين يومياً", "duration": "5 أيام"}],
    "investigations": [{"name_ar": "تحليل دم", "name_en": "CBC"}],
    "followup_days": 14,
}


@pytest.fixture
def doctor_fsm():
    return DoctorFSM(doctor_id=1, telegram_id=70001)


def test_idle_moves_to_listening_with_instructions(doctor_fsm):
    reply = run_async(doctor_fsm.handle("أي نص"))
    assert doctor_fsm.state == DoctorState.LISTENING
    assert "ملاحظات الجلسة" in reply


def test_listening_extracts_and_shows_review(doctor_fsm):
    doctor_fsm.state = DoctorState.LISTENING
    with patch("nlp.doctor_extractor.extract_session_fields", return_value=SAMPLE_SESSION):
        reply = run_async(doctor_fsm.handle("المريض اسمه أحمد خالد، شاكي من ألم ركبة"))

    assert doctor_fsm.state == DoctorState.REVIEW
    assert "أحمد خالد" in reply
    assert "Ibuprofen" in reply
    assert "تأكيد" in reply


def test_review_confirm_saves_session(doctor_fsm):
    doctor_fsm.state = DoctorState.REVIEW
    doctor_fsm.session = dict(SAMPLE_SESSION)
    doctor_fsm.session["raw_transcription"] = "note"

    saved = MagicMock()
    saved.patient_id = 10
    saved.appointment_id = "appt_1"

    @contextmanager
    def fake_get_db():
        yield MagicMock()

    with patch("database.db.get_db", fake_get_db):
        with patch("database.crud.create_session", return_value=saved) as mock_create:
            reply = run_async(doctor_fsm.handle("تأكيد"))

    mock_create.assert_called_once()
    assert doctor_fsm.state == DoctorState.SAVED
    assert doctor_fsm.session == {}
    assert "تم حفظ الجلسة" in reply
    assert "ربطها بملف المريض" in reply


def test_review_edit_field_updates_summary(doctor_fsm):
    doctor_fsm.state = DoctorState.REVIEW
    doctor_fsm.session = dict(SAMPLE_SESSION)

    reply = run_async(doctor_fsm.handle("الشكوى: وجع ظهر"))

    assert doctor_fsm.state == DoctorState.REVIEW
    assert doctor_fsm.session["chief_complaint"] == "وجع ظهر"
    assert "تم تحديث" in reply
    assert "وجع ظهر" in reply


def test_review_unknown_input_shows_save_hint(doctor_fsm):
    doctor_fsm.state = DoctorState.REVIEW
    doctor_fsm.session = dict(SAMPLE_SESSION)

    reply = run_async(doctor_fsm.handle("عدّل شي"))

    assert "تأكيد" in reply
    assert "الحقل" in reply


def test_editing_unknown_field_shows_help(doctor_fsm):
    doctor_fsm.state = DoctorState.REVIEW
    doctor_fsm.session = dict(SAMPLE_SESSION)

    reply = run_async(doctor_fsm.handle("عدّل شي"))

    assert "تأكيد" in reply


def test_saved_state_prompts_new_session(doctor_fsm):
    doctor_fsm.state = DoctorState.SAVED
    reply = run_async(doctor_fsm.handle("مرحبا"))
    assert "/session" in reply


def test_editable_fields_cover_summary_labels():
    labels = set(EDITABLE_FIELDS.keys())
    assert "اسم المريض" in labels
    assert "الشكوى" in labels
    assert "المتابعة" in labels


def test_medication_edit_updates_medications_list():
    fsm = DoctorFSM(doctor_id=1, telegram_id=70002)
    fsm.state = DoctorState.REVIEW
    fsm.session = dict(SAMPLE_SESSION)

    run_async(fsm.handle("الدواء: Paracetamol, Ibuprofen"))

    assert fsm.session["medications"] == [{"name": "Paracetamol"}, {"name": "Ibuprofen"}]


def test_followup_edit_parses_integer():
    fsm = DoctorFSM(doctor_id=1, telegram_id=70003)
    fsm.state = DoctorState.REVIEW
    fsm.session = dict(SAMPLE_SESSION)

    run_async(fsm.handle("المتابعة: 21"))

    assert fsm.session["followup_days"] == 21


def test_session_command_resets_to_listening():
    fsm = DoctorFSM(doctor_id=1, telegram_id=70004)
    fsm.state = DoctorState.SAVED
    fsm.session = {"patient_name": "old"}

    reply = run_async(fsm.handle("/session"))

    assert fsm.state == DoctorState.LISTENING
    assert fsm.session == {}
    assert "ملاحظات الجلسة" in reply
