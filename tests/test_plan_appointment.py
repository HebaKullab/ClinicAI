"""Tests for scheduler.plan_appointment orchestrator."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest

from database import crud
from scheduler.scheduler import plan_appointment
from tests.helpers import make_test_engine, make_test_session, run_async, seed_doctor, seed_slot, use_test_db
from utils.datetime_utils import utcnow


def _high_confidence_classify(*args, **kwargs):
    return {
        "specialty": "general_practice",
        "specialty_ar": "الطب العام",
        "method": "rule",
        "confidence": 0.95,
    }


@pytest.fixture
def plan_db():
    engine = make_test_engine()
    db = make_test_session(engine)
    doctor = seed_doctor(db, specialty="general_practice", clinic_code="PLAN-GP")
    when = (utcnow() + timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)
    slot = seed_slot(db, doctor, when=when, priority_class="P3")
    db.commit()
    yield {"db": db, "doctor": doctor, "when": when, "slot": slot}
    db.close()
    engine.dispose()


@patch("scheduler.scheduler.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
def test_plan_appointment_returns_best_slot(_mock_classify, plan_db):
    data = {
        "complaint": {"raw": "صداع خفيف"},
        "urgency_score": 0.25,
        "time_pref": {"date": str(plan_db["when"].date()), "phrase": "بكرا"},
        "telegram_id": 902_001,
    }

    with use_test_db(plan_db["db"]):
        decision = run_async(plan_appointment(data, plan_db["db"], gemini_client=None))

    assert decision.waitlisted is False
    assert decision.slot is not None
    assert decision.slot.slot_id == plan_db["slot"].slot_id
    assert decision.priority_class == "P3"


@patch("scheduler.scheduler.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
def test_plan_appointment_waitlists_when_no_slots(_mock_classify, plan_db):
    plan_db["slot"].status = "booked"
    plan_db["db"].add(plan_db["slot"])
    plan_db["db"].commit()

    data = {
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.25,
        "time_pref": {"date": None, "phrase": "أي وقت"},
        "telegram_id": 902_002,
    }

    with use_test_db(plan_db["db"]):
        decision = run_async(plan_appointment(data, plan_db["db"]))

    assert decision.waitlisted is True
    assert decision.slot is None
    assert decision.waitlist is not None
    assert decision.waitlist.position >= 1


@pytest.fixture
def plan_contract_db():
    engine = make_test_engine()
    db = make_test_session(engine)
    doctor = seed_doctor(db, specialty="general_practice", clinic_code="PLAN-CTR")
    when = (utcnow() + timedelta(days=3)).replace(hour=14, minute=0, second=0, microsecond=0)
    slot = seed_slot(db, doctor, when=when, priority_class="P3")
    db.commit()
    yield {"db": db, "when": when, "slot": slot}
    db.close()
    engine.dispose()


@patch("scheduler.scheduler.classify_with_gemini_fallback", side_effect=_high_confidence_classify)
def test_crud_and_plan_agree_on_best_slot(_mock_classify, plan_contract_db):
    data = {
        "complaint": {"raw": "متابعة دورية"},
        "urgency_score": 0.2,
        "time_pref": {"date": str(plan_contract_db["when"].date()), "phrase": "الأسبوع الجاي"},
        "priority_class": "P3",
    }

    with use_test_db(plan_contract_db["db"]):
        crud_slots = crud.find_available_slots(
            plan_contract_db["db"],
            specialty="general_practice",
            priority_class="P3",
            preferred_date=str(plan_contract_db["when"].date()),
            limit=1,
        )
        decision = run_async(plan_appointment(data, plan_contract_db["db"]))

    assert crud_slots
    assert decision.slot is not None
    assert crud_slots[0].slot_id == decision.slot.slot_id
