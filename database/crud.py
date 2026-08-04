from __future__ import annotations

from datetime import datetime, date, timedelta
from uuid import uuid4
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import Session, joinedload
from utils.datetime_utils import utcnow
from utils.datetime_utils import utcnow
from .models import (
    Patient,
    Doctor,
    Appointment,
    Session as DoctorSession,
    MessageLog,
    Slot,
    Conversation,
    PatientProfile,
    FsmSession,
)


ACTIVE_BOOKING_STATUSES = {"confirmed", "arrived", "waitlisted"}
APPOINTMENT_DURATION_MINUTES = 30


# ── IDs / helpers ──────────────────────────────────────────────────────────────

def _ensure_appt_id(appt_data: dict) -> str:
    return appt_data.get("appt_id") or f"appt_{utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:6]}"


def _complaint_raw(data: dict) -> str | None:
    complaint = data.get("complaint")
    if isinstance(complaint, dict):
        return complaint.get("raw") or complaint.get("text")
    if isinstance(complaint, str):
        return complaint
    return None


def _allowed_slot_priorities(priority_class: str | None) -> list[str | None]:
    """Reserved-capacity policy: urgent can use any slot; routine cannot take urgent reserves."""
    from scheduler.slot_policy import allowed_slot_priorities

    return list(allowed_slot_priorities(priority_class))


def _priority_rank_case(priority_class: str | None) -> int:
    if priority_class == "P1":
        return 1
    if priority_class == "P2":
        return 2
    if priority_class == "P3":
        return 3
    return 4


def _build_patient_profile_payload(data: dict) -> dict:
    return {
        "name": data.get("name"),
        "last_complaint": _complaint_raw(data),
        "last_specialty": data.get("specialty_hint") or data.get("specialty"),
        "last_specialty_ar": data.get("specialty_ar"),
        "last_priority_class": data.get("priority_class"),
        "last_priority_score": data.get("priority_score"),
        "last_priority_breakdown": data.get("priority_breakdown"),
        "last_time_preference": data.get("time_pref"),
        "is_followup": bool(data.get("is_followup")),
        "updated_from": "patient_fsm",
        "updated_at": utcnow().isoformat(),
    }


def _day_bounds(value: datetime) -> tuple[datetime, datetime]:
    start = datetime.combine(value.date(), datetime.min.time())
    end = datetime.combine(value.date(), datetime.max.time())
    return start, end


def _appointment_brief(appointment: Appointment | None) -> dict | None:
    if appointment is None:
        return None
    doctor = appointment.slot.doctor if appointment.slot and appointment.slot.doctor else None
    return {
        "appt_id": appointment.appt_id,
        "appt_datetime": appointment.appt_datetime,
        "specialty": appointment.specialty,
        "specialty_ar": appointment.specialty_ar,
        "status": appointment.status,
        "doctor_id": doctor.doctor_id if doctor else None,
        "doctor_name": doctor.name if doctor else None,
        "clinic_code": doctor.clinic_code if doctor else None,
        "clinic_name": doctor.clinic_name if doctor else None,
    }


def find_patient_booking_conflict(
    db: Session,
    patient_id: int,
    slot_datetime: datetime | None,
    specialty: str | None,
    ignore_appt_id: str | None = None,
) -> dict | None:
    """
    Patient booking guard used by the FSM before creating a new appointment.

    Policy:
    - Same patient cannot hold two active appointments that overlap in time, even if
      the specialties/clinics are different.
    - Same patient cannot hold two active appointments for the same specialty on the
      same day. Different specialties are allowed on the same day when their times do
      not overlap.
    """
    if not patient_id or slot_datetime is None:
        return None

    specialty = specialty or "general_practice"
    duration = timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
    overlap_start = slot_datetime - duration
    overlap_end = slot_datetime + duration

    time_stmt = (
        select(Appointment)
        .options(
            joinedload(Appointment.patient),
            joinedload(Appointment.slot).joinedload(Slot.doctor),
        )
        .where(
            Appointment.patient_id == patient_id,
            Appointment.status.in_(ACTIVE_BOOKING_STATUSES),
            Appointment.appt_datetime.is_not(None),
            Appointment.appt_datetime > overlap_start,
            Appointment.appt_datetime < overlap_end,
        )
        .order_by(Appointment.appt_datetime, desc(Appointment.created_at))
    )
    if ignore_appt_id:
        time_stmt = time_stmt.where(Appointment.appt_id != ignore_appt_id)

    overlapping = db.scalar(time_stmt.limit(1))
    if overlapping:
        return {
            "type": "time_overlap",
            "message": "Patient already has an active appointment at an overlapping time.",
            "appointment": overlapping,
        }

    day_start, day_end = _day_bounds(slot_datetime)
    specialty_stmt = (
        select(Appointment)
        .options(
            joinedload(Appointment.patient),
            joinedload(Appointment.slot).joinedload(Slot.doctor),
        )
        .where(
            Appointment.patient_id == patient_id,
            Appointment.status.in_(ACTIVE_BOOKING_STATUSES),
            Appointment.specialty == specialty,
            Appointment.appt_datetime >= day_start,
            Appointment.appt_datetime <= day_end,
        )
        .order_by(Appointment.appt_datetime, desc(Appointment.created_at))
    )
    if ignore_appt_id:
        specialty_stmt = specialty_stmt.where(Appointment.appt_id != ignore_appt_id)

    duplicate_specialty = db.scalar(specialty_stmt.limit(1))
    if duplicate_specialty:
        return {
            "type": "same_specialty_same_day",
            "message": "Patient already has an active appointment for this specialty on the same day.",
            "appointment": duplicate_specialty,
        }

    return None


def _first_slot_without_patient_conflict(db: Session, stmt, patient_id: int | None):
    """Return the earliest slot that does not conflict with the patient's active bookings."""
    for slot in db.scalars(stmt.limit(100)).all():
        if patient_id and _slot_conflicts_with_patient(db, patient_id, slot):
            continue
        return slot
    return None


def _slot_conflicts_with_patient(db: Session, patient_id: int, slot: Slot) -> bool:
    """True when booking this slot would violate patient booking policy."""
    specialty = slot.doctor.specialty if slot.doctor else slot.specialty
    return bool(find_patient_booking_conflict(db, patient_id, slot.slot_datetime, specialty))


# ── Doctors / patients ────────────────────────────────────────────────────────

def get_or_create_patient(db: Session, telegram_id: int, name: str | None = None):
    patient = db.scalar(select(Patient).where(Patient.telegram_id == telegram_id))
    if patient:
        changed = False
        if name and patient.name != name:
            patient.name = name
            changed = True
        if hasattr(patient, "updated_at"):
            patient.updated_at = utcnow()
            changed = True
        if changed:
            db.add(patient)
            db.flush()
            db.refresh(patient)
        return patient

    patient = Patient(telegram_id=telegram_id, name=name, updated_at=utcnow())
    db.add(patient)
    db.flush()
    db.refresh(patient)
    return patient


def get_patient_by_telegram_id(db: Session, telegram_id: int):
    return db.scalar(select(Patient).where(Patient.telegram_id == telegram_id))


def search_patient(db: Session, q: str):
    stmt = select(Patient).options(joinedload(Patient.appointments)).order_by(desc(Patient.created_at))
    if q:
        stmt = stmt.where(Patient.name.ilike(f"%{q}%"))
    return db.scalars(stmt).unique().all()


def get_latest_patient_appointment(db: Session, telegram_id: int):
    stmt = (
        select(Appointment)
        .join(Patient, Appointment.patient_id == Patient.patient_id)
        .options(
            joinedload(Appointment.patient),
            joinedload(Appointment.slot).joinedload(Slot.doctor),
        )
        .where(Patient.telegram_id == telegram_id)
        .order_by(desc(Appointment.created_at))
        .limit(1)
    )
    return db.scalar(stmt)


def cancel_latest_patient_appointment(db: Session, telegram_id: int):
    appt = get_latest_patient_appointment(db, telegram_id)
    if not appt or appt.status not in {"confirmed", "waitlisted"}:
        return None
    return update_appointment_status(db, appt.appt_id, "cancelled")


# ── Dashboard queries ─────────────────────────────────────────────────────────

def get_todays_queue(db: Session, target: date | None = None):
    target = target or date.today()
    start = datetime.combine(target, datetime.min.time())
    end = datetime.combine(target, datetime.max.time())
    stmt = (
        select(Appointment)
        .options(
            joinedload(Appointment.patient),
            joinedload(Appointment.slot).joinedload(Slot.doctor),
            joinedload(Appointment.sessions),
        )
        .where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end)
        .order_by(Appointment.appt_datetime)
    )
    return db.scalars(stmt).unique().all()


def daily_stats(db: Session):
    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())
    base = and_(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end)
    return {
        "total": db.scalar(select(func.count()).select_from(Appointment).where(base)) or 0,
        "P1": db.scalar(select(func.count()).select_from(Appointment).where(base, Appointment.priority_class == "P1")) or 0,
        "P2": db.scalar(select(func.count()).select_from(Appointment).where(base, Appointment.priority_class == "P2")) or 0,
        "P3": db.scalar(select(func.count()).select_from(Appointment).where(base, Appointment.priority_class == "P3")) or 0,
        "completed": db.scalar(select(func.count()).select_from(Appointment).where(base, Appointment.status == "completed")) or 0,
        "no_show": db.scalar(select(func.count()).select_from(Appointment).where(base, Appointment.status == "no_show")) or 0,
        "confirmed": db.scalar(select(func.count()).select_from(Appointment).where(base, Appointment.status == "confirmed")) or 0,
        "waitlisted": db.scalar(select(func.count()).select_from(Appointment).where(base, Appointment.status == "waitlisted")) or 0,
    }


def get_sessions(db: Session, target: date | None = None):
    target = target or date.today()
    start = datetime.combine(target, datetime.min.time())
    end = datetime.combine(target, datetime.max.time())
    stmt = (
        select(DoctorSession)
        .options(
            joinedload(DoctorSession.doctor),
            joinedload(DoctorSession.patient),
            joinedload(DoctorSession.appointment).joinedload(Appointment.patient),
            joinedload(DoctorSession.appointment).joinedload(Appointment.slot).joinedload(Slot.doctor),
        )
        .where(
            DoctorSession.session_datetime >= start,
            DoctorSession.session_datetime <= end,
        )
        .order_by(desc(DoctorSession.session_datetime))
    )
    return db.scalars(stmt).unique().all()


def get_recent_message_logs(db: Session, limit: int = 200):
    stmt = select(MessageLog).order_by(desc(MessageLog.created_at)).limit(limit)
    return db.scalars(stmt).all()


def _patient_id_for_telegram(db: Session, telegram_id: int) -> int | None:
    patient = db.scalar(select(Patient).where(Patient.telegram_id == telegram_id))
    return patient.patient_id if patient else None


def get_conversations(db: Session, limit: int = 200):
    stmt = (
        select(Conversation)
        .options(joinedload(Conversation.patient))
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    return db.scalars(stmt).all()


def get_conversation_messages(db: Session, telegram_id: int, limit: int = 200):
    conversation = db.scalar(select(Conversation).where(Conversation.telegram_id == telegram_id))
    if conversation:
        stmt = (
            select(MessageLog)
            .where(MessageLog.conversation_id == conversation.conversation_id)
            .order_by(MessageLog.created_at)
            .limit(limit)
        )
        messages = db.scalars(stmt).all()
        if messages:
            return messages

    # Backward-compatible fallback for old logs created before conversation_id existed.
    stmt = (
        select(MessageLog)
        .where(MessageLog.telegram_id == telegram_id)
        .order_by(MessageLog.created_at)
        .limit(limit)
    )
    return db.scalars(stmt).all()


def get_or_create_conversation(
    db: Session,
    telegram_id: int,
    username: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    role: str = "patient",
    patient_id: int | None = None,
):
    patient_id = patient_id or (_patient_id_for_telegram(db, telegram_id) if role == "patient" else None)
    conversation = db.scalar(select(Conversation).where(Conversation.telegram_id == telegram_id))
    if conversation:
        if patient_id and conversation.patient_id != patient_id:
            conversation.patient_id = patient_id
        if role and conversation.role != role:
            conversation.role = role
        if username and conversation.username != username:
            conversation.username = username
        if first_name and conversation.first_name != first_name:
            conversation.first_name = first_name
        if last_name and conversation.last_name != last_name:
            conversation.last_name = last_name
        conversation.updated_at = utcnow()
        db.add(conversation)
        db.flush()
        return conversation

    conversation = Conversation(
        telegram_id=telegram_id,
        patient_id=patient_id,
        role=role or "patient",
        username=username,
        first_name=first_name,
        last_name=last_name,
    )
    db.add(conversation)
    db.flush()
    return conversation


def attach_conversation_to_patient(db: Session, telegram_id: int, patient_id: int):
    """
    After the FSM creates the patient, attach the existing Telegram conversation
    and any earlier pre-registration logs to the patient record.
    """
    conversation = get_or_create_conversation(db, telegram_id=telegram_id, role="patient", patient_id=patient_id)
    db.query(MessageLog).filter(
        MessageLog.telegram_id == telegram_id,
        MessageLog.patient_id.is_(None),
    ).update({
        "patient_id": patient_id,
        "conversation_id": conversation.conversation_id,
    }, synchronize_session=False)
    db.flush()
    return conversation


# ── Scheduling ────────────────────────────────────────────────────────────────

def _slot_query(
    specialty: str,
    priority_class: str | None,
    preferred_date: str | None = None,
    allow_general_fallback: bool = False,
    doctor_id: int | None = None,
):
    """Build a slot query owned by an active doctor/clinic."""
    now = utcnow()
    specialties = [specialty]
    if allow_general_fallback and specialty != "general_practice":
        specialties.append("general_practice")

    stmt = (
        select(Slot)
        .join(Slot.doctor)
        .options(joinedload(Slot.doctor))
        .where(
            Slot.status == "available",
            Slot.slot_datetime >= now,
            Doctor.is_active.is_(True),
            Doctor.specialty.in_(specialties),
            or_(
                Slot.priority_class.is_(None),
                Slot.priority_class.in_([p for p in _allowed_slot_priorities(priority_class) if p]),
            ),
        )
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

    # Prefer the requested specialty before general-practice fallback.
    return stmt.order_by(
        (Doctor.specialty != specialty),
        Slot.slot_datetime,
        Slot.slot_id,
    )


def find_available_slots(
    db: Session,
    specialty: str,
    priority_class: str,
    preferred_date: str | None = None,
    patient_id: int | None = None,
    telegram_id: int | None = None,
    doctor_id: int | None = None,
    *,
    limit: int = 3,
) -> list:
    """Return up to `limit` candidate slots (same logic as find_next_available_slot)."""
    from config import USE_SLOT_POLICY

    patient_id = patient_id or (_patient_id_for_telegram(db, telegram_id) if telegram_id else None)

    if USE_SLOT_POLICY:
        from scheduler.slot_policy import select_slots

        return select_slots(
            db,
            specialty=specialty,
            priority_class=priority_class,
            preferred_date=preferred_date,
            patient_id=patient_id,
            doctor_id=doctor_id,
            limit=limit,
            conflict_checker=_slot_conflicts_with_patient,
        )

    seen: set[int] = set()
    results = []

    def _collect(stmt):
        nonlocal results
        for row in db.scalars(stmt).all():
            if row.slot_id in seen:
                continue
            if patient_id and _slot_conflicts_with_patient(db, patient_id, row):
                continue
            seen.add(row.slot_id)
            results.append(row)
            if len(results) >= limit:
                return

    if priority_class != "P1" and preferred_date:
        _collect(_slot_query(specialty, priority_class, preferred_date, doctor_id=doctor_id))
    if len(results) < limit:
        _collect(_slot_query(specialty, priority_class, doctor_id=doctor_id))
    if len(results) < limit and specialty != "general_practice":
        if priority_class != "P1" and preferred_date:
            _collect(_slot_query(specialty, priority_class, preferred_date, allow_general_fallback=True, doctor_id=doctor_id))
        if len(results) < limit:
            _collect(_slot_query(specialty, priority_class, allow_general_fallback=True, doctor_id=doctor_id))
    return results


def find_next_available_slot(
    db: Session,
    specialty: str,
    priority_class: str,
    preferred_date: str | None = None,
    patient_id: int | None = None,
    telegram_id: int | None = None,
    doctor_id: int | None = None,
):
    """
    Read-only lookup used by FSM preview.
    The final booking re-checks the slot and patient conflicts inside reserve_slot_and_create_appointment().
    """
    from config import USE_SLOT_POLICY

    patient_id = patient_id or (_patient_id_for_telegram(db, telegram_id) if telegram_id else None)

    if USE_SLOT_POLICY:
        from scheduler.slot_policy import select_best_slot

        return select_best_slot(
            db,
            specialty=specialty,
            priority_class=priority_class,
            preferred_date=preferred_date,
            patient_id=patient_id,
            doctor_id=doctor_id,
            conflict_checker=_slot_conflicts_with_patient,
        )

    # P1 should get the earliest safe slot; P2/P3 first try user's preferred date.
    if priority_class != "P1" and preferred_date:
        slot = _first_slot_without_patient_conflict(db, _slot_query(specialty, priority_class, preferred_date, doctor_id=doctor_id), patient_id)
        if slot:
            return slot

    slot = _first_slot_without_patient_conflict(db, _slot_query(specialty, priority_class, doctor_id=doctor_id), patient_id)
    if slot:
        return slot

    # If specialty-specific capacity is full, allow general-practice fallback.
    if specialty != "general_practice":
        if priority_class != "P1" and preferred_date:
            slot = _first_slot_without_patient_conflict(
                db,
                _slot_query(specialty, priority_class, preferred_date, allow_general_fallback=True, doctor_id=doctor_id),
                patient_id,
            )
            if slot:
                return slot
        return _first_slot_without_patient_conflict(
            db,
            _slot_query(specialty, priority_class, allow_general_fallback=True, doctor_id=doctor_id),
            patient_id,
        )

    return None


def create_patient_file(db: Session, patient: Patient, data: dict) -> PatientProfile:
    profile_payload = _build_patient_profile_payload(data)
    return upsert_profile(db, patient.telegram_id, profile_payload, patient_id=patient.patient_id)


def reserve_slot_and_create_appointment(db: Session, data: dict, slot_id: int | None, patient: Patient):
    """
    Atomic booking step:
      1) re-read the slot from DB
      2) verify it is still available
      3) block duplicate/overlapping active bookings for this patient
      4) create appointment
      5) mark slot as booked
    """
    slot = None
    if slot_id is not None:
        slot = db.scalar(
            select(Slot)
            .options(joinedload(Slot.doctor))
            .where(Slot.slot_id == slot_id)
        )
        if slot is None or slot.status != "available" or slot.doctor is None or not slot.doctor.is_active:
            return {"appointment": None, "slot_conflict": True, "booking_conflict": None}

        patient_priority = data.get("priority_class")
        if slot.priority_class not in set(_allowed_slot_priorities(patient_priority)):
            return {"appointment": None, "slot_conflict": True, "booking_conflict": None}

    appt_datetime = slot.slot_datetime if slot else None
    requested_specialty = data.get("specialty_hint") or data.get("specialty")
    specialty = slot.doctor.specialty if slot and slot.doctor else requested_specialty

    if slot is not None:
        conflict = find_patient_booking_conflict(db, patient.patient_id, appt_datetime, specialty)
        if conflict:
            return {"appointment": None, "slot_conflict": False, "booking_conflict": conflict}

    status = "confirmed" if appt_datetime else "waitlisted"
    appointment = Appointment(
        appt_id=_ensure_appt_id(data),
        patient_id=patient.patient_id,
        slot_id=slot.slot_id if slot else None,
        appt_datetime=appt_datetime,
        specialty=specialty,
        specialty_ar=data.get("specialty_ar"),
        priority_class=data.get("priority_class"),
        priority_score=data.get("priority_score"),
        complaint_summary=_complaint_raw(data),
        time_preference=data.get("time_pref"),
        status=status,
        updated_at=utcnow(),
    )
    if slot is not None and appt_datetime is not None:
        slot.status = "booked"
        slot.updated_at = utcnow()
        db.add(slot)
    db.add(appointment)
    db.flush()
    db.refresh(appointment)
    return {"appointment": appointment, "slot_conflict": False, "booking_conflict": None}


def create_patient_file_and_book(db: Session, telegram_id: int, data: dict, slot_id: int | None):
    """
    Main FSM persistence function.
    Creates/updates Patient, saves PatientProfile, reserves a Slot, then creates Appointment.
    One commit only after all steps succeed.
    """
    patient = get_or_create_patient(db, telegram_id=telegram_id, name=data.get("name"))
    attach_conversation_to_patient(db, telegram_id=telegram_id, patient_id=patient.patient_id)
    profile = create_patient_file(db, patient, data)
    booking = reserve_slot_and_create_appointment(db, data, slot_id, patient)
    appointment = booking.get("appointment")
    booking_conflict = booking.get("booking_conflict")
    if booking_conflict and booking_conflict.get("appointment") is not None:
        booking_conflict = dict(booking_conflict)
        booking_conflict["appointment"] = _appointment_brief(booking_conflict.get("appointment"))

    if booking.get("slot_conflict") or booking_conflict:
        db.rollback()
        return {
            "patient": patient,
            "profile": profile,
            "appointment": None,
            "slot_conflict": bool(booking.get("slot_conflict")),
            "booking_conflict": booking_conflict,
        }
    db.commit()
    db.refresh(patient)
    db.refresh(profile)
    if appointment:
        db.refresh(appointment)
    return {
        "patient": patient,
        "profile": profile,
        "appointment": appointment,
        "slot_conflict": False,
        "booking_conflict": None,
    }


def create_waitlist_appointment(db: Session, telegram_id: int, data: dict):
    patient = get_or_create_patient(db, telegram_id=telegram_id, name=data.get("name"))
    attach_conversation_to_patient(db, telegram_id=telegram_id, patient_id=patient.patient_id)
    profile = create_patient_file(db, patient, data)
    booking = reserve_slot_and_create_appointment(db, data, slot_id=None, patient=patient)
    appointment = booking.get("appointment")
    db.commit()
    db.refresh(patient)
    db.refresh(profile)
    if appointment:
        db.refresh(appointment)
    return {"patient": patient, "profile": profile, "appointment": appointment, "booking_conflict": booking.get("booking_conflict")}


def create_appointment(db: Session, data: dict, slot: object, patient: Patient):
    """Backward-compatible wrapper for older code paths."""
    slot_id = getattr(slot, "slot_id", None)
    booking = reserve_slot_and_create_appointment(db, data, slot_id, patient)
    appointment = booking.get("appointment")
    db.commit()
    if appointment:
        db.refresh(appointment)
    return appointment


def update_appointment_status(db: Session, appt_id: str, status: str):
    appt = db.scalar(select(Appointment).options(joinedload(Appointment.slot)).where(Appointment.appt_id == appt_id))
    if not appt:
        return None

    old_status = appt.status
    appt.status = status
    appt.updated_at = utcnow()

    # Cancelled appointments release their slot; completed/no_show keep historical slot booked.
    if status == "cancelled" and appt.slot is not None:
        appt.slot.status = "available"
        appt.slot.updated_at = utcnow()
        db.add(appt.slot)

    db.add(appt)
    db.commit()
    db.refresh(appt)
    return appt


# ── Doctor sessions ───────────────────────────────────────────────────────────

def _clean_name(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def find_patient_by_name(db: Session, patient_name: str | None):
    """Best-effort lookup used when the doctor note mentions a patient by name."""
    from nlp.normalizer import normalize

    patient_name = _clean_name(patient_name)
    if not patient_name:
        return None

    norm_query = normalize(patient_name)

    exact_stmt = (
        select(Patient)
        .where(func.lower(Patient.name) == patient_name.lower())
        .order_by(desc(Patient.updated_at), desc(Patient.created_at))
        .limit(1)
    )
    patient = db.scalar(exact_stmt)
    if patient:
        return patient

    fuzzy_stmt = (
        select(Patient)
        .where(Patient.name.ilike(f"%{patient_name}%"))
        .order_by(desc(Patient.updated_at), desc(Patient.created_at))
        .limit(1)
    )
    patient = db.scalar(fuzzy_stmt)
    if patient:
        return patient

    # Arabic orthography differs between NLP normalization and stored names.
    for candidate in db.scalars(
        select(Patient).order_by(desc(Patient.updated_at), desc(Patient.created_at))
    ).all():
        if not candidate.name:
            continue
        norm_stored = normalize(candidate.name)
        if norm_stored == norm_query or norm_query in norm_stored or norm_stored in norm_query:
            return candidate
    return None


def find_appointment_for_session(
    db: Session,
    patient_id: int | None = None,
    patient_name: str | None = None,
    appointment_id: str | None = None,
):
    """
    Link a doctor session to the most likely appointment.
    Priority: explicit appointment_id → today's appointment for patient → latest appointment for patient.
    """
    if appointment_id:
        appointment = db.scalar(
            select(Appointment)
            .options(
                joinedload(Appointment.patient),
                joinedload(Appointment.slot).joinedload(Slot.doctor),
            )
            .where(Appointment.appt_id == appointment_id)
        )
        if appointment:
            return appointment

    patient = db.get(Patient, patient_id) if patient_id else find_patient_by_name(db, patient_name)
    if not patient:
        return None

    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())

    base = (
        select(Appointment)
        .options(
            joinedload(Appointment.patient),
            joinedload(Appointment.slot).joinedload(Slot.doctor),
        )
        .where(Appointment.patient_id == patient.patient_id)
    )

    today_stmt = (
        base
        .where(Appointment.appt_datetime >= start, Appointment.appt_datetime <= end)
        .where(Appointment.status.in_(["confirmed", "arrived", "completed"]))
        .order_by(Appointment.appt_datetime, desc(Appointment.created_at))
        .limit(1)
    )
    appointment = db.scalar(today_stmt)
    if appointment:
        return appointment

    latest_stmt = (
        base
        .where(Appointment.status != "cancelled")
        .order_by(desc(Appointment.appt_datetime), desc(Appointment.created_at))
        .limit(1)
    )
    return db.scalar(latest_stmt)


def resolve_session_links(db: Session, session_data: dict):
    """Resolve patient and appointment for a doctor session before saving it."""
    patient_id = session_data.get("patient_id")
    appointment_id = session_data.get("appointment_id")
    patient_name = _clean_name(session_data.get("patient_name"))

    appointment = find_appointment_for_session(db, patient_id, patient_name, appointment_id)
    patient = None

    if appointment and appointment.patient:
        patient = appointment.patient
    elif patient_id:
        patient = db.get(Patient, patient_id)
    elif patient_name:
        patient = find_patient_by_name(db, patient_name)

    return patient, appointment


def create_session(db: Session, session_data: dict, doctor_id: int):
    patient, appointment = resolve_session_links(db, session_data)

    session = DoctorSession(
        doctor_id=doctor_id,
        patient_id=patient.patient_id if patient else session_data.get("patient_id"),
        appointment_id=appointment.appt_id if appointment else session_data.get("appointment_id"),
        patient_name=(patient.name if patient else _clean_name(session_data.get("patient_name"))),
        chief_complaint=session_data.get("chief_complaint"),
        diagnosis=session_data.get("diagnosis"),
        medications=session_data.get("medications"),
        investigations=session_data.get("investigations"),
        followup_days=session_data.get("followup_days"),
        raw_transcription=session_data.get("raw_transcription"),
    )
    db.add(session)

    # A saved clinical session means the related visit was completed.
    if appointment and appointment.status in {"confirmed", "arrived"}:
        appointment.status = "completed"
        appointment.updated_at = utcnow()
        db.add(appointment)

    db.commit()
    db.refresh(session)
    return session


# ── Doctors / clinic schedules / dashboard session entry ─────────────────────

def get_doctors(db: Session, active_only: bool = False):
    stmt = select(Doctor).order_by(Doctor.doctor_id)
    if active_only:
        stmt = stmt.where(Doctor.is_active.is_(True))
    return db.scalars(stmt).all()


def get_doctor_by_telegram(db: Session, telegram_id: int) -> Doctor | None:
    return db.scalar(select(Doctor).where(Doctor.telegram_id == telegram_id))


def get_doctor_slots(db: Session, target: date | None = None, doctor_id: int | None = None):
    target = target or date.today()
    start = datetime.combine(target, datetime.min.time())
    end = datetime.combine(target, datetime.max.time())
    stmt = (
        select(Slot)
        .options(
            joinedload(Slot.doctor),
            joinedload(Slot.appointment).joinedload(Appointment.patient),
        )
        .where(Slot.slot_datetime >= start, Slot.slot_datetime <= end)
        .order_by(Slot.slot_datetime, Slot.doctor_id)
    )
    if doctor_id is not None:
        stmt = stmt.where(Slot.doctor_id == doctor_id)
    return db.scalars(stmt).unique().all()


def get_doctor_schedule_summary(db: Session, target: date | None = None):
    target = target or date.today()
    start = datetime.combine(target, datetime.min.time())
    end = datetime.combine(target, datetime.max.time())
    doctors = get_doctors(db)
    slots = get_doctor_slots(db, target)
    grouped: dict[int, dict] = {
        doctor.doctor_id: {
            "doctor": doctor,
            "total": 0,
            "available": 0,
            "booked": 0,
        }
        for doctor in doctors
    }
    for slot in slots:
        summary = grouped.setdefault(slot.doctor_id, {"doctor": slot.doctor, "total": 0, "available": 0, "booked": 0})
        summary["total"] += 1
        if slot.status == "available":
            summary["available"] += 1
        elif slot.status == "booked":
            summary["booked"] += 1
    return list(grouped.values())


def get_session_candidates(db: Session, target: date | None = None):
    """Appointments that can be documented from the dashboard without Telegram IDs."""
    target = target or date.today()
    start = datetime.combine(target, datetime.min.time())
    end = datetime.combine(target, datetime.max.time())
    stmt = (
        select(Appointment)
        .options(
            joinedload(Appointment.patient),
            joinedload(Appointment.slot).joinedload(Slot.doctor),
            joinedload(Appointment.sessions),
        )
        .where(
            Appointment.appt_datetime >= start,
            Appointment.appt_datetime <= end,
            Appointment.status.in_(["confirmed", "arrived", "completed"]),
            Appointment.slot_id.is_not(None),
        )
        .order_by(Appointment.appt_datetime)
    )
    appointments = db.scalars(stmt).unique().all()
    return [appointment for appointment in appointments if not appointment.sessions]


def _dashboard_list(value: str | list | None, item_key: str) -> list[dict] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    values = [part.strip() for part in str(value).replace("\n", ",").split(",") if part.strip()]
    return [{item_key: item} for item in values] or None


def create_session_from_dashboard(db: Session, session_data: dict):
    """
    Create one clinical session for a booked appointment from the dashboard.
    Doctor, patient and clinic are derived from Appointment -> Slot -> Doctor,
    so no doctor Telegram ID is required.
    """
    appointment_id = (session_data.get("appointment_id") or "").strip()
    if not appointment_id:
        raise ValueError("appointment_id is required")

    appointment = db.scalar(
        select(Appointment)
        .options(
            joinedload(Appointment.patient),
            joinedload(Appointment.slot).joinedload(Slot.doctor),
            joinedload(Appointment.sessions),
        )
        .where(Appointment.appt_id == appointment_id)
    )
    if appointment is None:
        raise ValueError("Appointment not found")
    if appointment.slot is None or appointment.slot.doctor is None:
        raise ValueError("Appointment is not linked to a doctor slot")
    if appointment.sessions:
        raise ValueError("A session is already registered for this appointment")

    followup = session_data.get("followup_days")
    try:
        followup_days = int(followup) if followup not in (None, "") else None
    except (TypeError, ValueError):
        followup_days = None

    session = DoctorSession(
        doctor_id=appointment.slot.doctor.doctor_id,
        patient_id=appointment.patient_id,
        appointment_id=appointment.appt_id,
        patient_name=appointment.patient.name if appointment.patient else None,
        chief_complaint=session_data.get("chief_complaint") or appointment.complaint_summary,
        diagnosis=session_data.get("diagnosis"),
        medications=_dashboard_list(session_data.get("medications"), "name"),
        investigations=_dashboard_list(session_data.get("investigations"), "name_ar"),
        followup_days=followup_days,
        raw_transcription=session_data.get("raw_transcription"),
        session_datetime=appointment.appt_datetime or utcnow(),
    )
    db.add(session)
    appointment.status = "completed"
    appointment.updated_at = utcnow()
    db.add(appointment)
    db.commit()
    db.refresh(session)
    return session


# ── Message/profile logging ───────────────────────────────────────────────────

def log_message(
    db: Session,
    telegram_id: int,
    direction: str,
    message_type: str,
    content: str,
    role: str = "patient",
):
    conversation = get_or_create_conversation(db, telegram_id=telegram_id, role=role)
    log = MessageLog(
        conversation_id=conversation.conversation_id,
        patient_id=conversation.patient_id,
        telegram_id=telegram_id,
        direction=direction,
        message_type=message_type,
        content=content,
    )
    conversation.updated_at = utcnow()
    db.add(conversation)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def log_bot_reply(db: Session, telegram_id: int, content: str, role: str = "patient"):
    return log_message(db, telegram_id, "outbound", "bot_reply", content, role=role)


def get_profile(db: Session, telegram_id: int) -> dict | None:
    profile = db.scalar(select(PatientProfile).where(PatientProfile.telegram_id == telegram_id))
    if not profile:
        return None
    return profile.data or {}


def upsert_profile(db: Session, telegram_id: int, data: dict, patient_id: int | None = None) -> PatientProfile:
    profile = db.scalar(select(PatientProfile).where(PatientProfile.telegram_id == telegram_id))
    if not profile:
        profile = PatientProfile(telegram_id=telegram_id, patient_id=patient_id, data=data)
        db.add(profile)
        db.flush()
        db.refresh(profile)
        return profile

    existing = profile.data or {}
    merged = {**existing, **(data or {})}
    profile.data = merged
    if patient_id and not profile.patient_id:
        profile.patient_id = patient_id
    profile.updated_at = utcnow()
    db.add(profile)
    db.flush()
    db.refresh(profile)
    return profile


def delete_profile(db: Session, telegram_id: int) -> bool:
    profile = db.scalar(select(PatientProfile).where(PatientProfile.telegram_id == telegram_id))
    if not profile:
        return False
    db.delete(profile)
    db.commit()
    return True


# ── FSM session persistence ───────────────────────────────────────────────────

FSM_SESSION_TTL_HOURS = 24


def cleanup_stale_fsm_sessions(db: Session, *, ttl_hours: int = FSM_SESSION_TTL_HOURS) -> int:
    cutoff = utcnow() - timedelta(hours=ttl_hours)
    rows = db.scalars(select(FsmSession).where(FsmSession.updated_at < cutoff)).all()
    count = len(rows)
    for row in rows:
        db.delete(row)
    if count:
        db.commit()
    return count


def get_fsm_session(db: Session, telegram_id: int, role: str = "patient") -> FsmSession | None:
    return db.scalar(
        select(FsmSession).where(FsmSession.telegram_id == telegram_id, FsmSession.role == role)
    )


def upsert_fsm_session(
    db: Session,
    telegram_id: int,
    *,
    role: str,
    state: str,
    data_json: dict | None = None,
    slot_options_json: list | None = None,
    slot_index: int = 0,
    slot_json: dict | None = None,
    priority_json: dict | None = None,
    finalized_appointment_id: str | None = None,
) -> FsmSession:
    row = get_fsm_session(db, telegram_id, role=role)
    if row is None:
        row = FsmSession(
            telegram_id=telegram_id,
            role=role,
            state=state,
            data_json=data_json or {},
            slot_options_json=slot_options_json or [],
            slot_index=slot_index,
            slot_json=slot_json,
            priority_json=priority_json,
            finalized_appointment_id=finalized_appointment_id,
        )
        db.add(row)
    else:
        row.state = state
        row.data_json = data_json if data_json is not None else row.data_json
        row.slot_options_json = slot_options_json if slot_options_json is not None else row.slot_options_json
        row.slot_index = slot_index
        row.slot_json = slot_json if slot_json is not None else row.slot_json
        row.priority_json = priority_json if priority_json is not None else row.priority_json
        row.finalized_appointment_id = finalized_appointment_id
        row.updated_at = utcnow()
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_fsm_session(db: Session, telegram_id: int, role: str = "patient") -> bool:
    row = get_fsm_session(db, telegram_id, role=role)
    if not row:
        return False
    db.delete(row)
    db.commit()
    return True
