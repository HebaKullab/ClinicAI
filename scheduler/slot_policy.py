"""
Unified slot selection policy shared by crud, scheduler.plan_appointment, and FSM.

Block rules, wave horizons, load-aware ranking, and patient-conflict filtering
live here so production Telegram flow and benchmark preview use the same logic.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from database.models import Appointment, Doctor, Slot
from utils.datetime_utils import utcnow

# ── Constants ─────────────────────────────────────────────────────────────────

FALLBACK_SPECIALTY = "general_practice"

BLOCK_ACCESS: dict[str, set[str | None]] = {
    "P1": {"P1", "P2", "P3", None},
    "P2": {"P2", "P3", None},
    "P3": {"P3", None},
}

WAVE_HORIZON_DAYS: dict[str, int] = {
    "P1": 2,
    "P2": 7,
    "P3": 30,
}


def allowed_slot_priorities(priority_class: str | None) -> set[str | None]:
    """Reserved-capacity policy: urgent patients may use any slot tier."""
    if priority_class == "P1":
        return {"P1", "P2", "P3", None}
    if priority_class == "P2":
        return {"P2", "P3", None}
    return {"P3", None}


# ── Lightweight slot view ─────────────────────────────────────────────────────

class SlotView:
    """Minimal slot interface for policy functions (ORM Slot or AppointmentSlot)."""

    __slots__ = (
        "slot_id",
        "slot_datetime",
        "specialty",
        "priority_class",
        "doctor_id",
        "_orm",
    )

    def __init__(
        self,
        slot_id: int,
        slot_datetime: datetime,
        specialty: str,
        priority_class: str | None,
        doctor_id: int | None,
        _orm=None,
    ):
        self.slot_id = slot_id
        self.slot_datetime = slot_datetime
        self.specialty = specialty
        self.priority_class = priority_class
        self.doctor_id = doctor_id
        self._orm = _orm

    @classmethod
    def from_orm(cls, row: Slot) -> SlotView:
        doctor = row.doctor
        specialty = doctor.specialty if doctor else row.specialty
        return cls(
            slot_id=row.slot_id,
            slot_datetime=row.slot_datetime,
            specialty=specialty,
            priority_class=row.priority_class,
            doctor_id=row.doctor_id,
            _orm=row,
        )


# ── SQL query builder (raw layer) ─────────────────────────────────────────────

def build_slot_query(
    specialty: str,
    priority_class: str | None,
    preferred_date: str | None = None,
    *,
    allow_general_fallback: bool = False,
    doctor_id: int | None = None,
):
    """Build a slot query owned by an active doctor/clinic."""
    now = utcnow()
    specialties = [specialty]
    if allow_general_fallback and specialty != FALLBACK_SPECIALTY:
        specialties.append(FALLBACK_SPECIALTY)

    allowed = [p for p in allowed_slot_priorities(priority_class) if p is not None]

    stmt = (
        select(Slot)
        .join(Slot.doctor)
        .options(joinedload(Slot.doctor))
        .where(
            Slot.status == "available",
            Slot.slot_datetime >= now,
            Doctor.is_active.is_(True),
            Doctor.specialty.in_(specialties),
        )
    )
    if allowed:
        stmt = stmt.where(
            (Slot.priority_class.is_(None)) | (Slot.priority_class.in_(allowed))
        )

    if doctor_id is not None:
        stmt = stmt.where(Slot.doctor_id == doctor_id)

    if preferred_date:
        try:
            start = datetime.fromisoformat(preferred_date)
            end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
            stmt = stmt.where(Slot.slot_datetime >= start, Slot.slot_datetime <= end)
        except ValueError:
            pass

    return stmt.order_by(
        (Doctor.specialty != specialty),
        Slot.slot_datetime,
        Slot.slot_id,
    )


# ── Policy filters ────────────────────────────────────────────────────────────

def filter_by_block_rules(
    slots: list[SlotView],
    priority_class: str,
) -> list[SlotView]:
    allowed = BLOCK_ACCESS.get(priority_class, {None})
    return [s for s in slots if s.priority_class in allowed]


def filter_by_wave_rules(
    slots: list[SlotView],
    priority_class: str,
    *,
    now: datetime | None = None,
) -> list[SlotView]:
    now = now or utcnow()
    horizon_days = WAVE_HORIZON_DAYS.get(priority_class, 7)
    wave_end = now + timedelta(days=horizon_days)
    return [s for s in slots if now <= s.slot_datetime <= wave_end]


def _parse_preferred_date(preferred_date: str | None) -> date | None:
    if not preferred_date:
        return None
    try:
        return date.fromisoformat(str(preferred_date))
    except (TypeError, ValueError):
        return None


def clinic_load_by_day(db: Session, specialty: str) -> dict[tuple[str, date], int]:
    stmt = (
        select(Appointment.specialty, func.date(Appointment.appt_datetime), func.count())
        .where(
            Appointment.status == "confirmed",
            Appointment.appt_datetime.isnot(None),
            Appointment.specialty == specialty,
        )
        .group_by(Appointment.specialty, func.date(Appointment.appt_datetime))
    )
    out: dict[tuple[str, date], int] = {}
    for spec, day, count in db.execute(stmt).all():
        if isinstance(day, str):
            day = date.fromisoformat(day)
        out[(spec, day)] = count
    return out


def doctor_load_by_day(db: Session) -> dict[tuple[int, date], int]:
    stmt = (
        select(Slot.doctor_id, func.date(Slot.slot_datetime), func.count())
        .where(Slot.status == "booked", Slot.doctor_id.isnot(None))
        .group_by(Slot.doctor_id, func.date(Slot.slot_datetime))
    )
    out: dict[tuple[int, date], int] = {}
    for doc_id, day, count in db.execute(stmt).all():
        if isinstance(day, str):
            day = date.fromisoformat(day)
        out[(doc_id, day)] = count
    return out


def slot_utilization_by_day(db: Session, specialty: str) -> dict[tuple[str, date], float]:
    total_stmt = (
        select(func.date(Slot.slot_datetime), func.count())
        .where(Slot.specialty == specialty)
        .group_by(func.date(Slot.slot_datetime))
    )
    booked_stmt = (
        select(func.date(Slot.slot_datetime), func.count())
        .where(Slot.specialty == specialty, Slot.status == "booked")
        .group_by(func.date(Slot.slot_datetime))
    )
    totals = {day: cnt for day, cnt in db.execute(total_stmt).all()}
    booked = {day: cnt for day, cnt in db.execute(booked_stmt).all()}

    util: dict[tuple[str, date], float] = {}
    for day, total in totals.items():
        b = booked.get(day, 0)
        util[(specialty, day)] = (b / total) if total else 0.0
    return util


def _build_slot_sort_key(
    slot: SlotView,
    *,
    pref_day: date | None,
    priority_class: str,
    clinic_load: dict[tuple[str, date], int],
    doctor_load: dict[tuple[int, date], int],
    utilization: dict[tuple[str, date], float],
) -> tuple:
    day = slot.slot_datetime.date()
    day_match = 0 if (pref_day is None or day == pref_day) else 1
    block_match = 0 if slot.priority_class == priority_class else 1
    c_load = clinic_load.get((slot.specialty, day), 0)
    d_load = doctor_load.get((slot.doctor_id, day), 0) if slot.doctor_id else 0
    util = utilization.get((slot.specialty, day), 0.0)

    if priority_class == "P1":
        return (day_match, block_match, slot.slot_datetime, c_load, d_load, util)
    return (day_match, block_match, c_load, d_load, util, slot.slot_datetime)


def rank_slots(
    db: Session,
    slots: list[SlotView],
    *,
    specialty: str,
    priority_class: str,
    priority_score: float = 0.5,
    preferred_date: str | None = None,
) -> list[SlotView]:
    del priority_score  # reserved for future tie-breakers
    pref_day = _parse_preferred_date(preferred_date)
    clinic_load = clinic_load_by_day(db, specialty)
    doctor_load = doctor_load_by_day(db)
    utilization = slot_utilization_by_day(db, specialty)

    return sorted(
        slots,
        key=lambda s: _build_slot_sort_key(
            s,
            pref_day=pref_day,
            priority_class=priority_class,
            clinic_load=clinic_load,
            doctor_load=doctor_load,
            utilization=utilization,
        ),
    )


def filter_patient_conflicts(
    db: Session,
    slots: list[SlotView],
    patient_id: int | None,
    *,
    conflict_checker,
) -> list[SlotView]:
    """Drop slots that would violate patient booking policy."""
    if not patient_id:
        return slots
    safe: list[SlotView] = []
    for slot in slots:
        orm = slot._orm
        if orm is None:
            safe.append(slot)
            continue
        if conflict_checker(db, patient_id, orm):
            continue
        safe.append(slot)
    return safe


def _collect_candidates(
    db: Session,
    specialty: str,
    priority_class: str,
    preferred_date: str | None,
    *,
    doctor_id: int | None = None,
) -> list[SlotView]:
    seen: set[int] = set()
    views: list[SlotView] = []

    def _add_rows(stmt):
        for row in db.scalars(stmt).all():
            if row.slot_id in seen:
                continue
            seen.add(row.slot_id)
            views.append(SlotView.from_orm(row))

    if priority_class != "P1" and preferred_date:
        _add_rows(build_slot_query(specialty, priority_class, preferred_date, doctor_id=doctor_id))
    _add_rows(build_slot_query(specialty, priority_class, doctor_id=doctor_id))
    if specialty != FALLBACK_SPECIALTY:
        if priority_class != "P1" and preferred_date:
            _add_rows(
                build_slot_query(
                    specialty,
                    priority_class,
                    preferred_date,
                    allow_general_fallback=True,
                    doctor_id=doctor_id,
                )
            )
        _add_rows(
            build_slot_query(
                specialty,
                priority_class,
                allow_general_fallback=True,
                doctor_id=doctor_id,
            )
        )
    return views


def select_slots(
    db: Session,
    *,
    specialty: str,
    priority_class: str,
    preferred_date: str | None = None,
    patient_id: int | None = None,
    doctor_id: int | None = None,
    limit: int = 3,
    apply_wave: bool = True,
    conflict_checker=None,
) -> list[Slot]:
    """
    Unified slot selection: query → block rules → wave rules → rank → patient filter.
    Returns ORM Slot rows in priority order.
    """
    from database import crud

    checker = conflict_checker or crud._slot_conflicts_with_patient
    candidates = _collect_candidates(
        db,
        specialty,
        priority_class,
        preferred_date,
        doctor_id=doctor_id,
    )
    candidates = filter_by_block_rules(candidates, priority_class)
    if apply_wave:
        candidates = filter_by_wave_rules(candidates, priority_class)
    candidates = rank_slots(
        db,
        candidates,
        specialty=specialty,
        priority_class=priority_class,
        preferred_date=preferred_date,
    )
    candidates = filter_patient_conflicts(db, candidates, patient_id, conflict_checker=checker)

    results: list[Slot] = []
    for view in candidates:
        if view._orm is not None:
            results.append(view._orm)
            if len(results) >= limit:
                break
    return results


def select_best_slot(
    db: Session,
    *,
    specialty: str,
    priority_class: str,
    preferred_date: str | None = None,
    patient_id: int | None = None,
    doctor_id: int | None = None,
    conflict_checker=None,
) -> Optional[Slot]:
    slots = select_slots(
        db,
        specialty=specialty,
        priority_class=priority_class,
        preferred_date=preferred_date,
        patient_id=patient_id,
        doctor_id=doctor_id,
        limit=1,
        conflict_checker=conflict_checker,
    )
    return slots[0] if slots else None
