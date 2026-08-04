"""Tests for scheduler/slot_policy.py unified slot selection."""
from __future__ import annotations

from datetime import timedelta

import pytest

from database.models import Appointment
from scheduler.slot_policy import (
    SlotView,
    filter_by_block_rules,
    filter_by_wave_rules,
    select_slots,
)
from tests.helpers import make_test_engine, make_test_session, seed_doctor, seed_patient, seed_slot, use_test_db
from utils.datetime_utils import utcnow


@pytest.fixture
def slot_policy_ctx():
    engine = make_test_engine()
    db = make_test_session(engine)
    doctor = seed_doctor(db, specialty="general_practice", clinic_code="POL-GP")
    patient = seed_patient(db, telegram_id=901_001, name="مريض سياسة")
    now = utcnow()
    yield {"db": db, "doctor": doctor, "patient": patient, "now": now}
    db.close()
    engine.dispose()


def _view(slot) -> SlotView:
    return SlotView.from_orm(slot)


def test_p3_cannot_take_p1_only_slot(slot_policy_ctx):
    db = slot_policy_ctx["db"]
    doctor = slot_policy_ctx["doctor"]
    now = slot_policy_ctx["now"]

    p1_slot = _view(seed_slot(db, doctor, when=now + timedelta(days=1), priority_class="P1"))
    p3_open = _view(seed_slot(db, doctor, when=now + timedelta(days=1, hours=1), priority_class="P3"))
    db.commit()

    result = filter_by_block_rules([p1_slot, p3_open], "P3")
    assert len(result) == 1
    assert result[0].priority_class == "P3"


def test_p1_wave_blocks_slot_beyond_two_days(slot_policy_ctx):
    db = slot_policy_ctx["db"]
    doctor = slot_policy_ctx["doctor"]
    now = slot_policy_ctx["now"]

    near = _view(seed_slot(db, doctor, when=now + timedelta(days=1), priority_class="P3"))
    far = _view(seed_slot(db, doctor, when=now + timedelta(days=5), priority_class="P3"))
    db.commit()

    result = filter_by_wave_rules([near, far], "P1", now=now)
    assert len(result) == 1
    assert result[0].slot_id == near.slot_id


def test_p3_ranking_prefers_less_loaded_day(slot_policy_ctx):
    db = slot_policy_ctx["db"]
    doctor = slot_policy_ctx["doctor"]
    patient = slot_policy_ctx["patient"]
    now = slot_policy_ctx["now"]

    busy_day = (now + timedelta(days=3)).replace(hour=10, minute=0, second=0, microsecond=0)
    quiet_day = (now + timedelta(days=4)).replace(hour=10, minute=0, second=0, microsecond=0)
    busy_slot = seed_slot(db, doctor, when=busy_day, priority_class="P3")
    quiet_slot = seed_slot(db, doctor, when=quiet_day, priority_class="P3")
    for i in range(3):
        db.add(
            Appointment(
                appt_id=f"load-busy-{i}",
                patient_id=patient.patient_id,
                slot_id=busy_slot.slot_id,
                appt_datetime=busy_day.replace(hour=10 + i),
                specialty="general_practice",
                status="confirmed",
            )
        )
    db.commit()

    with use_test_db(db):
        slots = select_slots(
            db,
            specialty="general_practice",
            priority_class="P3",
            limit=2,
        )
    assert len(slots) >= 2
    assert slots[0].slot_id == quiet_slot.slot_id


def test_select_slots_skips_patient_conflict(slot_policy_ctx):
    db = slot_policy_ctx["db"]
    doctor = slot_policy_ctx["doctor"]
    patient = slot_policy_ctx["patient"]
    now = slot_policy_ctx["now"]

    when = (now + timedelta(days=2)).replace(hour=11, minute=0, second=0, microsecond=0)
    conflict_slot = seed_slot(db, doctor, when=when, priority_class="P3")
    alt_slot = seed_slot(db, doctor, when=when + timedelta(days=1), priority_class="P3")
    db.add(
        Appointment(
            appt_id="appt-conflict-1",
            patient_id=patient.patient_id,
            slot_id=conflict_slot.slot_id,
            appt_datetime=when,
            specialty="general_practice",
            status="confirmed",
        )
    )
    db.commit()

    with use_test_db(db):
        slots = select_slots(
            db,
            specialty="general_practice",
            priority_class="P3",
            patient_id=patient.patient_id,
            limit=3,
        )
    ids = {s.slot_id for s in slots}
    assert conflict_slot.slot_id not in ids
    assert alt_slot.slot_id in ids


def test_gp_fallback_when_specialty_empty(slot_policy_ctx):
    db = slot_policy_ctx["db"]
    doctor = slot_policy_ctx["doctor"]
    now = slot_policy_ctx["now"]

    seed_doctor(db, specialty="orthopedics", clinic_code="POL-ORTHO")
    gp_slot = seed_slot(db, doctor, when=now + timedelta(days=2), priority_class="P3")
    db.commit()

    with use_test_db(db):
        slots = select_slots(
            db,
            specialty="orthopedics",
            priority_class="P3",
            limit=1,
        )
    assert len(slots) == 1
    assert slots[0].slot_id == gp_slot.slot_id
    assert slots[0].doctor.specialty == "general_practice"
