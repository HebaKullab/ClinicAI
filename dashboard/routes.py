"""FastAPI dashboard routes connected to the same SQLite DB as the Telegram bot."""
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import crud
from database.db import get_db_dependency

router = APIRouter()
templates = Jinja2Templates(directory="dashboard/templates")


class SessionCreatePayload(BaseModel):
    appointment_id: str
    chief_complaint: str | None = None
    diagnosis: str | None = None
    medications: str | None = None
    investigations: str | None = None
    followup_days: int | None = None
    raw_transcription: str | None = None


def _target_day(day: str | None) -> date:
    try:
        return date.fromisoformat(day) if day else date.today()
    except ValueError:
        return date.today()


# ── Pages ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db_dependency)):
    stats = crud.daily_stats(db)
    appts = crud.get_todays_queue(db)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "stats": stats,
            "appointments": appts,
            "today": date.today().strftime("%A، %d/%m/%Y"),
        },
    )


@router.get("/appointments", response_class=HTMLResponse)
async def appointments_page(
    request: Request,
    day: str | None = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    target = _target_day(day)
    appts = crud.get_todays_queue(db, target)
    return templates.TemplateResponse(
        request,
        "appointments.html",
        {
            "request": request,
            "appointments": appts,
            "target_date": target.isoformat(),
            "display_date": target.strftime("%A، %d/%m/%Y"),
            "prev_date": (target - timedelta(days=1)).isoformat(),
            "next_date": (target + timedelta(days=1)).isoformat(),
        },
    )


@router.get("/doctors", response_class=HTMLResponse)
async def doctors_page(
    request: Request,
    day: str | None = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    target = _target_day(day)
    summaries = crud.get_doctor_schedule_summary(db, target)
    return templates.TemplateResponse(
        request,
        "doctors.html",
        {
            "request": request,
            "summaries": summaries,
            "target_date": target.isoformat(),
            "display_date": target.strftime("%A، %d/%m/%Y"),
            "prev_date": (target - timedelta(days=1)).isoformat(),
            "next_date": (target + timedelta(days=1)).isoformat(),
        },
    )


@router.get("/slots", response_class=HTMLResponse)
async def slots_page(
    request: Request,
    day: str | None = Query(default=None),
    doctor_id: int | None = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    target = _target_day(day)
    doctors = crud.get_doctors(db, active_only=True)
    slots = crud.get_doctor_slots(db, target, doctor_id=doctor_id)
    selected_doctor = next((doctor for doctor in doctors if doctor.doctor_id == doctor_id), None)
    return templates.TemplateResponse(
        request,
        "slots.html",
        {
            "request": request,
            "slots": slots,
            "doctors": doctors,
            "selected_doctor": selected_doctor,
            "doctor_id": doctor_id,
            "target_date": target.isoformat(),
            "display_date": target.strftime("%A، %d/%m/%Y"),
            "prev_date": (target - timedelta(days=1)).isoformat(),
            "next_date": (target + timedelta(days=1)).isoformat(),
        },
    )


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(
    request: Request,
    day: str | None = Query(default=None),
    appointment_id: str | None = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    target = _target_day(day)
    sessions = crud.get_sessions(db, target)
    candidates = crud.get_session_candidates(db, target)
    selected_appointment = next(
        (appointment for appointment in candidates if appointment.appt_id == appointment_id),
        None,
    )
    return templates.TemplateResponse(
        request,
        "sessions.html",
        {
            "request": request,
            "sessions": sessions,
            "candidates": candidates,
            "selected_appointment": selected_appointment,
            "appointment_id": appointment_id,
            "display_date": target.strftime("%A، %d/%m/%Y"),
            "target_date": target.isoformat(),
            "prev_date": (target - timedelta(days=1)).isoformat(),
            "next_date": (target + timedelta(days=1)).isoformat(),
        },
    )


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, db: Session = Depends(get_db_dependency)):
    logs = crud.get_recent_message_logs(db, limit=200)
    return templates.TemplateResponse(
        request,
        "logs.html",
        {"request": request, "logs": logs},
    )


@router.get("/conversations", response_class=HTMLResponse)
async def conversations_page(
    request: Request,
    telegram_id: int | None = Query(default=None),
    db: Session = Depends(get_db_dependency),
):
    conversations = crud.get_conversations(db, limit=200)
    selected = None
    messages = []
    if telegram_id is not None:
        selected = next((c for c in conversations if c.telegram_id == telegram_id), None)
        if selected is not None:
            messages = crud.get_conversation_messages(db, telegram_id, limit=200)

    return templates.TemplateResponse(
        request,
        "conversations.html",
        {
            "request": request,
            "conversations": conversations,
            "selected": selected,
            "messages": messages,
            "telegram_id": telegram_id,
        },
    )


@router.get("/patients", response_class=HTMLResponse)
async def patients_page(
    request: Request,
    q: str = Query(default=""),
    db: Session = Depends(get_db_dependency),
):
    patients = crud.search_patient(db, q) if q else []
    return templates.TemplateResponse(
        request,
        "patients.html",
        {"request": request, "patients": patients, "query": q},
    )


# ── JSON API ──────────────────────────────────────────────────────────────────

@router.get("/api/queue/today")
async def api_queue(db: Session = Depends(get_db_dependency)):
    appts = crud.get_todays_queue(db)
    return JSONResponse([
        {
            "appt_id": a.appt_id,
            "patient_name": a.patient.name if a.patient else "—",
            "time": a.appt_datetime.strftime("%H:%M") if a.appt_datetime else "—",
            "specialty": a.specialty_ar or a.specialty,
            "doctor_name": a.slot.doctor.name if a.slot and a.slot.doctor else "—",
            "clinic_name": a.slot.doctor.clinic_name if a.slot and a.slot.doctor else "—",
            "clinic_code": a.slot.doctor.clinic_code if a.slot and a.slot.doctor else "—",
            "priority_class": a.priority_class,
            "priority_score": a.priority_score,
            "complaint": a.complaint_summary,
            "status": a.status,
        }
        for a in appts
    ])


@router.get("/api/stats/today")
async def api_stats(db: Session = Depends(get_db_dependency)):
    return JSONResponse(crud.daily_stats(db))


@router.post("/api/appointment/{appt_id}/status")
async def update_status(
    appt_id: str,
    status: str = Query(...),
    db: Session = Depends(get_db_dependency),
):
    appt = crud.update_appointment_status(db, appt_id, status)
    return JSONResponse({"ok": bool(appt)})


@router.post("/api/sessions")
async def create_dashboard_session(
    payload: SessionCreatePayload,
    db: Session = Depends(get_db_dependency),
):
    try:
        session = crud.create_session_from_dashboard(db, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(
        {
            "ok": True,
            "session_id": session.session_id,
            "appointment_id": session.appointment_id,
            "patient_id": session.patient_id,
            "doctor_id": session.doctor_id,
        }
    )
