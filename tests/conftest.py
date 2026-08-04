"""Shared pytest fixtures for ClinicAI tests."""
from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from database import crud
from tests.helpers import make_test_engine, make_test_session, seed_doctor, seed_slot
from utils.datetime_utils import utcnow


@pytest.fixture(autouse=True)
def disable_live_gemini(monkeypatch):
    """Unit tests must not call the real Gemini API when .env has a key."""
    from nlp import gemini_client

    monkeypatch.setattr(gemini_client.gemini, "_available", False)


@pytest.fixture
def db_engine():
    engine = make_test_engine()
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Session:
    db = make_test_session(db_engine)
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def gp_doctor(db_session):
    return seed_doctor(
        db_session,
        specialty="general_practice",
        clinic_code="CLINIC-GP-FIXTURE",
    )


@pytest.fixture
def gp_slot(db_session, gp_doctor):
    when = (utcnow() + timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
    return seed_slot(db_session, gp_doctor, when=when, priority_class="P3")


@pytest.fixture
def clear_patient_fsm(db_session):
    def _clear(user_id: int = 910_001) -> None:
        crud.delete_fsm_session(db_session, user_id, role="patient")

    return _clear
