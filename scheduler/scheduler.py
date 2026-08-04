"""
scheduler/scheduler.py — Tier-1 booking orchestration (read-only preview path).

Pipeline:
  request → classify → priority → fetch slots → block rules → wave rules
         → rank/select OR waitlist

NOTE: The actual booking (DB writes) is handled by patient_fsm.py → crud.py.
This module provides classification, ranking, and preview logic only.
GA (Tier 2) and Monte Carlo (Tier 3) stay outside this module.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from database import crud
from database.models import Appointment, Doctor, PatientProfile, Slot
from scheduler.classifier import classify_specialty, classify_with_gemini_fallback
from scheduler.priority import score_and_classify
from scheduler.slot_policy import (
    FALLBACK_SPECIALTY,
    SlotView,
    filter_by_block_rules,
    filter_by_wave_rules,
    rank_slots as policy_rank_slots,
    select_best_slot as policy_select_best_slot,
)
from utils.datetime_utils import utcnow


logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PRIORITY_CLASSES = frozenset({"P1", "P2", "P3"})
APPOINTMENT_STATUSES = frozenset(
    {"confirmed", "waitlisted", "completed", "no_show", "cancelled"}
)

# Re-export unified policy constants for backward compatibility.
from scheduler.slot_policy import BLOCK_ACCESS, WAVE_HORIZON_DAYS  # noqa: E402

def _parse_doctor_id(notes: str | None) -> int | None:
    """Legacy helper: extract doctor_id from notes text. Prefer Slot.doctor_id FK."""
    if not notes:
        return None
    for part in notes.split():
        if part.startswith("doctor_id:"):
            try:
                return int(part.split(":", 1)[1])
            except (TypeError, ValueError):
                return None
    return None


@dataclass(frozen=True)
class AppointmentSlot:
    """Immutable slot view. Actual DB writes go through crud.reserve_slot_and_create_appointment()."""

    slot_id: int
    slot_datetime: datetime
    specialty: str
    priority_class: str | None
    status: str
    doctor_id: int | None = None
    notes: str | None = None

    @classmethod
    def from_orm(cls, row: Slot) -> AppointmentSlot:
        # Prefer the ORM FK column; fall back to parsing notes for legacy data.
        doctor_id = getattr(row, "doctor_id", None) or _parse_doctor_id(row.notes)
        return cls(
            slot_id=row.slot_id,
            slot_datetime=row.slot_datetime,
            specialty=row.specialty,
            priority_class=row.priority_class,
            status=row.status,
            doctor_id=doctor_id,
            notes=row.notes,
        )


@dataclass(frozen=True)
class WaitlistCandidate:
    specialty: str
    priority_class: str
    priority_score: float
    urgency_score: float
    arrival_time: datetime
    telegram_id: int | None = None


@dataclass
class WaitlistEntry:
    specialty: str
    priority_class: str
    priority_score: float
    urgency_score: float
    arrival_time: datetime
    position: int
    estimated_note: str


@dataclass
class ScheduleDecision:
    specialty: str
    specialty_ar: str
    classification_method: str
    classification_confidence: float

    priority_class: str
    priority_score: float
    priority_label_ar: str
    priority_color: str

    slot: Optional[AppointmentSlot]
    waitlisted: bool
    waitlist: Optional[WaitlistEntry] = None


# ── Input helpers ─────────────────────────────────────────────────────────────

def sanitize_input(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data or {})

    complaint = data.get("complaint")
    if not isinstance(complaint, dict):
        complaint = {"raw": str(complaint or "").strip()}
    complaint.setdefault("raw", "")
    data["complaint"] = complaint

    time_pref = data.get("time_pref")
    if not isinstance(time_pref, dict):
        data["time_pref"] = {"date": None, "phrase": "أي وقت متاح"}
    else:
        time_pref.setdefault("date", None)
        time_pref.setdefault("phrase", "أي وقت متاح")

    try:
        if "urgency_score" in data and data["urgency_score"] is not None:
            data["urgency_score"] = float(data["urgency_score"])
    except (TypeError, ValueError):
        data["urgency_score"] = 0.3

    if "arrival_time" not in data:
        data["arrival_time"] = utcnow().isoformat()

    return data


def normalize_priority_class(raw: str | None) -> str:
    val = (raw or "").strip().upper()
    return val if val in PRIORITY_CLASSES else "P2"


# ── Metrics ───────────────────────────────────────────────────────────────────

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
    return {(spec, day): count for spec, day, count in db.execute(stmt).all()}


def doctor_load_by_day(db: Session) -> dict[tuple[int, date], int]:
    """Count booked slots per doctor per day using the FK column directly."""
    stmt = (
        select(Slot.doctor_id, func.date(Slot.slot_datetime), func.count())
        .where(Slot.status == "booked", Slot.doctor_id.isnot(None))
        .group_by(Slot.doctor_id, func.date(Slot.slot_datetime))
    )
    return {(doc_id, day): count for doc_id, day, count in db.execute(stmt).all()}


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


# ── Slots ─────────────────────────────────────────────────────────────────────

def get_available_slots(db: Session, specialty: str) -> list[AppointmentSlot]:
    """Fetch available slots for the requested specialty with GP fallback."""
    from scheduler.slot_policy import build_slot_query

    now = utcnow()
    rows = list(db.scalars(build_slot_query(specialty, "P1")).all())
    if rows:
        return [AppointmentSlot.from_orm(r) for r in rows if r.slot_datetime >= now]

    if specialty != FALLBACK_SPECIALTY:
        rows = list(db.scalars(build_slot_query(FALLBACK_SPECIALTY, "P1")).all())
        if rows:
            return [AppointmentSlot.from_orm(r) for r in rows if r.slot_datetime >= now]

    return []


def apply_block_rules(
    slots: list[AppointmentSlot],
    priority_class: str,
) -> list[AppointmentSlot]:
    views = filter_by_block_rules([_slot_to_view(s) for s in slots], priority_class)
    allowed_ids = {v.slot_id for v in views}
    return [s for s in slots if s.slot_id in allowed_ids]


def apply_wave_rules(
    slots: list[AppointmentSlot],
    priority_class: str,
    *,
    now: datetime | None = None,
) -> list[AppointmentSlot]:
    views = filter_by_wave_rules([_slot_to_view(s) for s in slots], priority_class, now=now or utcnow())
    allowed_ids = {v.slot_id for v in views}
    return [s for s in slots if s.slot_id in allowed_ids]


def _slot_to_view(slot: AppointmentSlot) -> SlotView:
    return SlotView(
        slot_id=slot.slot_id,
        slot_datetime=slot.slot_datetime,
        specialty=slot.specialty,
        priority_class=slot.priority_class,
        doctor_id=slot.doctor_id,
    )


def _build_slot_sort_key(slot: AppointmentSlot, **kwargs):
    """Backward-compatible wrapper for ranking unit tests."""
    from scheduler.slot_policy import _build_slot_sort_key as policy_key

    return policy_key(_slot_to_view(slot), **kwargs)


def book_slot(db: Session, slot: AppointmentSlot) -> None:
    """Mark a slot as booked in the database.

    .. deprecated::
        This only marks the Slot row and does NOT create an Appointment record.
        The correct booking path is ``crud.reserve_slot_and_create_appointment()``
        which atomically checks conflicts, creates the Appointment, and marks
        the slot as booked.  This function is kept for backward compatibility.
    """
    warnings.warn(
        "book_slot() does not create an Appointment record. "
        "Use crud.reserve_slot_and_create_appointment() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    row = db.get(Slot, slot.slot_id)
    if row is None:
        raise ValueError(f"Slot {slot.slot_id} not found")
    if row.status != "available":
        raise ValueError(f"Slot {slot.slot_id} is not available")
    row.status = "booked"
    db.add(row)
    db.commit()


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_slots(
    db: Session,
    slots: list[AppointmentSlot],
    *,
    specialty: str,
    priority_class: str,
    priority_score: float,
    preferred_date: str | None = None,
) -> list[AppointmentSlot]:
    ranked = policy_rank_slots(
        db,
        [_slot_to_view(s) for s in slots],
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        preferred_date=preferred_date,
    )
    order = {view.slot_id: idx for idx, view in enumerate(ranked)}
    return sorted(slots, key=lambda s: order.get(s.slot_id, len(order)))


def select_best_slot(
    db: Session,
    slots: list[AppointmentSlot],
    *,
    specialty: str,
    priority_class: str,
    priority_score: float,
    preferred_date: str | None = None,
) -> Optional[AppointmentSlot]:
    ranked = rank_slots(
        db,
        slots,
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        preferred_date=preferred_date,
    )
    return ranked[0] if ranked else None


def _parse_preferred_date(preferred_date: str | None) -> date | None:
    if not preferred_date:
        return None
    try:
        return date.fromisoformat(str(preferred_date))
    except (TypeError, ValueError):
        return None


# ── Waitlist ──────────────────────────────────────────────────────────────────

def _queue_sort_key(c: WaitlistCandidate) -> tuple:
    return (-c.priority_score, c.arrival_time, -c.urgency_score)


def _candidate_from_profile(specialty: str, profile: PatientProfile) -> WaitlistCandidate | None:
    data = profile.data or {}
    wl = data.get("waitlist")
    if not isinstance(wl, dict) or wl.get("specialty") != specialty:
        return None
    try:
        return WaitlistCandidate(
            specialty=specialty,
            priority_class=str(wl.get("priority_class", "P2")),
            priority_score=float(wl.get("priority_score", 0.0)),
            urgency_score=float(wl.get("urgency_score", 0.0)),
            arrival_time=datetime.fromisoformat(str(wl["arrival_time"])),
            telegram_id=profile.telegram_id,
        )
    except (TypeError, ValueError, KeyError):
        return None


def _load_queue(db: Session, specialty: str) -> list[WaitlistCandidate]:
    queue: list[WaitlistCandidate] = []
    profiles = list(db.scalars(select(PatientProfile)).all())
    profile_by_tg = {p.telegram_id: p for p in profiles}

    stmt = select(Appointment).where(
        Appointment.status == "waitlisted",
        Appointment.specialty == specialty,
    )
    for appt in db.scalars(stmt).all():
        patient = appt.patient
        tg_id = patient.telegram_id if patient else None
        profile = profile_by_tg.get(tg_id) if tg_id else None

        if profile:
            cand = _candidate_from_profile(specialty, profile)
            if cand:
                queue.append(cand)
                continue

        queue.append(
            WaitlistCandidate(
                specialty=specialty,
                priority_class=appt.priority_class or "P3",
                priority_score={"P1": 0.9, "P2": 0.5, "P3": 0.2}.get(
                    appt.priority_class or "P3", 0.2
                ),
                urgency_score=0.3,
                arrival_time=appt.created_at or utcnow(),
                telegram_id=tg_id,
            )
        )

    queue.sort(key=_queue_sort_key)
    return queue


def compute_waitlist_position(db: Session, candidate: WaitlistCandidate) -> int:
    queue = _load_queue(db, candidate.specialty)
    queue.append(candidate)
    queue.sort(key=_queue_sort_key)
    for idx, item in enumerate(queue, start=1):
        if (
            item.telegram_id == candidate.telegram_id
            and item.arrival_time == candidate.arrival_time
            and item.priority_score == candidate.priority_score
        ):
            return idx
    return len(queue)


def persist_waitlist_metadata(
    db: Session,
    *,
    telegram_id: int,
    candidate: WaitlistCandidate,
    position: int,
) -> None:
    crud.upsert_profile(
        db,
        telegram_id,
        {
            "waitlist": {
                "specialty": candidate.specialty,
                "priority_class": candidate.priority_class,
                "priority_score": candidate.priority_score,
                "urgency_score": candidate.urgency_score,
                "arrival_time": candidate.arrival_time.isoformat(),
                "position": position,
            }
        },
    )


def enqueue_waitlist(
    db: Session,
    *,
    specialty: str,
    priority_class: str,
    priority_score: float,
    urgency_score: float,
    arrival_time: datetime | None = None,
    telegram_id: int | None = None,
) -> WaitlistEntry:
    arrival = arrival_time or utcnow()
    candidate = WaitlistCandidate(
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        urgency_score=urgency_score,
        arrival_time=arrival,
        telegram_id=telegram_id,
    )
    position = compute_waitlist_position(db, candidate)

    if telegram_id is not None:
        persist_waitlist_metadata(
            db, telegram_id=telegram_id, candidate=candidate, position=position
        )

    return WaitlistEntry(
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        urgency_score=urgency_score,
        arrival_time=arrival,
        position=position,
        estimated_note=(
            f"أنت رقم {position} في قائمة الانتظار "
            f"({specialty}, أولوية {priority_score:.2f})"
        ),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def _classify(data: dict[str, Any], gemini_client: Optional[object]) -> dict[str, Any]:
    complaint_text = data.get("complaint", {}).get("raw", "") or ""

    if gemini_client is not None:
        try:
            return await classify_with_gemini_fallback(complaint_text, gemini_client)
        except Exception as exc:
            logger.exception("Gemini classification failed; using rule-based classifier: %s", exc)

    return classify_specialty(complaint_text)


async def plan_appointment(
    data: dict[str, Any],
    db: Session,
    gemini_client: Optional[object] = None,
) -> ScheduleDecision:
    """Read-only preview: classify, score, rank slots, but do NOT book.

    The actual booking (DB write) is done by the FSM via
    ``crud.reserve_slot_and_create_appointment()`` which atomically checks
    slot availability, patient booking conflicts (V4 policy), creates the
    Appointment, and marks the slot as booked.

    This function is safe to call multiple times without side effects
    (except waitlist metadata when no slot is found).
    """
    safe_data = sanitize_input(data)

    spec_result = await _classify(safe_data, gemini_client)
    safe_data["specialty_hint"] = spec_result["specialty"]
    safe_data["specialty_ar"] = spec_result["specialty_ar"]

    pr = score_and_classify(safe_data)
    priority_class = normalize_priority_class(pr.priority_class)
    priority_score = pr.score
    safe_data["priority_class"] = priority_class
    safe_data["priority_score"] = priority_score

    specialty = safe_data.get("specialty_hint", "general_practice")
    preferred_date = safe_data.get("time_pref", {}).get("date")
    urgency_score = float(safe_data.get("urgency_score", 0.3))

    try:
        arrival_time = datetime.fromisoformat(str(safe_data.get("arrival_time")))
    except (TypeError, ValueError):
        arrival_time = utcnow()

    telegram_id = safe_data.get("telegram_id") or safe_data.get("user_id")

    # ── Slot search (read-only — no DB writes here) ──────────────────────
    available = get_available_slots(db, specialty)
    available = apply_block_rules(available, priority_class)
    available = apply_wave_rules(available, priority_class)
    slot = select_best_slot(
        db,
        available,
        specialty=specialty,
        priority_class=priority_class,
        priority_score=priority_score,
        preferred_date=preferred_date,
    )

    waitlist = None
    if slot is None:
        waitlist = enqueue_waitlist(
            db,
            specialty=specialty,
            priority_class=priority_class,
            priority_score=priority_score,
            urgency_score=urgency_score,
            arrival_time=arrival_time,
            telegram_id=int(telegram_id) if telegram_id is not None else None,
        )
        logger.info(
            "Waitlisted: specialty=%s score=%.3f position=%s",
            specialty,
            priority_score,
            waitlist.position,
        )

    return ScheduleDecision(
        specialty=specialty,
        specialty_ar=safe_data.get("specialty_ar", "الطب العام"),
        classification_method=str(spec_result.get("method", "unknown")),
        classification_confidence=float(spec_result.get("confidence", 0.0)),
        priority_class=priority_class,
        priority_score=priority_score,
        priority_label_ar=pr.label_ar,
        priority_color=pr.label_color,
        slot=slot,
        waitlisted=(slot is None),
        waitlist=waitlist,
    )
