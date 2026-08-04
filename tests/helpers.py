"""Shared helpers for ClinicAI unit tests."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager, ExitStack
from datetime import datetime, timedelta
from typing import Iterator
from unittest.mock import patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from database.db import Base
from database import models  # noqa: F401
from database.models import Doctor, Patient, Slot
from utils.datetime_utils import utcnow

def infer_urgency_score(text: str) -> float:
    t = text
    high = ["الآن", "فجأة", "حاد", "شديد", "صعوبة نطق", "ضعف مفاجئ", "وقع", "أزمة", "هبوط سكر"]
    mid = ["دوخة", "تعب", "تنميل", "صفير", "تورم", "رجفة"]
    low = ["متابعة", "دوري", "روتيني", "مراجعة", "بدون أعراض جديدة"]

    if any(k in t for k in high):
        return 0.9
    if any(k in t for k in mid):
        return 0.55
    if any(k in t for k in low):
        return 0.25
    return 0.4


def run_async(coro):
    """Run an async coroutine in a fresh event loop (Python 3.7 compatible)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def unpack_fsm(result):
    """Convert FSM (reply, action, payload) to (reply, keyboard) for tests."""
    from fsm.fsm_result import keyboard_for_action

    reply, action, _payload = result
    return reply, keyboard_for_action(action)


def make_test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return engine


def make_test_session(engine=None) -> Session:
    engine = engine or make_test_engine()
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return factory()


@contextmanager
def use_test_db(db: Session) -> Iterator[None]:
    """Patch every module-level get_db import to use the in-memory session."""

    @contextmanager
    def _get_db():
        yield db

    targets = (
        "database.db.get_db",
        "bot.router.get_db",
        "bot.handlers.patient.get_db",
        "bot.handlers.doctor.get_db",
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, _get_db))
        yield


def seed_doctor(
    db: Session,
    specialty: str = "general_practice",
    *,
    name: str = "د. اختبار",
    clinic_code: str | None = None,
    clinic_name: str | None = None,
    is_active: bool = True,
    telegram_id: int | None = None,
) -> Doctor:
    doctor = Doctor(
        telegram_id=telegram_id,
        name=name,
        specialty=specialty,
        clinic_code=clinic_code or f"CLINIC-{specialty[:4].upper()}",
        clinic_name=clinic_name or f"عيادة {specialty}",
        is_active=is_active,
    )
    db.add(doctor)
    db.flush()
    db.refresh(doctor)
    return doctor


def seed_patient(db: Session, telegram_id: int = 900001, name: str = "مريض اختبار") -> Patient:
    patient = Patient(telegram_id=telegram_id, name=name, updated_at=utcnow())
    db.add(patient)
    db.flush()
    db.refresh(patient)
    return patient


def seed_slot(
    db: Session,
    doctor: Doctor,
    *,
    when: datetime | None = None,
    status: str = "available",
    priority_class: str | None = None,
) -> Slot:
    slot = Slot(
        doctor_id=doctor.doctor_id,
        slot_datetime=when or (utcnow() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0),
        specialty=doctor.specialty,
        priority_class=priority_class,
        status=status,
    )
    db.add(slot)
    db.flush()
    db.refresh(slot)
    return slot
