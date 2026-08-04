from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .db import Base
from utils.datetime_utils import utcnow
from utils.datetime_utils import utcnow


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    phone = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    appointments = relationship("Appointment", back_populates="patient")
    profile = relationship("PatientProfile", back_populates="patient", uselist=False)
    conversations = relationship("Conversation", back_populates="patient")
    message_logs = relationship("MessageLog", back_populates="patient")
    sessions = relationship("Session", back_populates="patient")


class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id = Column(Integer, primary_key=True, autoincrement=True)
    # Optional: a doctor can exist as a clinic/scheduling resource without using Telegram.
    telegram_id = Column(Integer, unique=True, index=True, nullable=True)
    name = Column(String(255), nullable=False)
    specialty = Column(String(128), nullable=False, index=True)
    clinic_code = Column(String(32), unique=True, nullable=False, index=True)
    clinic_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, default=utcnow)

    slots = relationship("Slot", back_populates="doctor")
    sessions = relationship("Session", back_populates="doctor")


class Appointment(Base):
    __tablename__ = "appointments"

    appt_id = Column(String(64), primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=True)
    slot_id = Column(Integer, ForeignKey("slots.slot_id"), nullable=True, index=True)
    appt_datetime = Column(DateTime, nullable=True)
    specialty = Column(String(128), nullable=True)
    specialty_ar = Column(String(128), nullable=True)
    priority_class = Column(String(8), nullable=True)
    priority_score = Column(Float, nullable=True)
    complaint_summary = Column(Text, nullable=True)
    time_preference = Column(JSON, nullable=True)
    status = Column(String(32), nullable=False, default="confirmed")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    patient = relationship("Patient", back_populates="appointments")
    slot = relationship("Slot", back_populates="appointment")
    sessions = relationship("Session", back_populates="appointment")


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=True, index=True)
    appointment_id = Column(String(64), ForeignKey("appointments.appt_id"), nullable=True, index=True)
    patient_name = Column(String(255), nullable=True)
    chief_complaint = Column(Text, nullable=True)
    diagnosis = Column(Text, nullable=True)
    medications = Column(JSON, nullable=True)
    investigations = Column(JSON, nullable=True)
    followup_days = Column(Integer, nullable=True)
    raw_transcription = Column(Text, nullable=True)
    session_datetime = Column(DateTime, default=utcnow)

    doctor = relationship("Doctor", back_populates="sessions")
    patient = relationship("Patient", back_populates="sessions")
    appointment = relationship("Appointment", back_populates="sessions")


class Slot(Base):
    __tablename__ = "slots"

    slot_id = Column(Integer, primary_key=True, autoincrement=True)
    # The doctor is the clinic/scheduling resource that owns this slot.
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), nullable=False, index=True)
    slot_datetime = Column(DateTime, nullable=False, index=True)
    # Kept for backward compatibility and fast reporting; doctor.specialty is authoritative.
    specialty = Column(String(128), nullable=False, default="general_practice", index=True)
    priority_class = Column(String(8), nullable=True)
    status = Column(String(32), nullable=False, default="available", index=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    doctor = relationship("Doctor", back_populates="slots")
    appointment = relationship("Appointment", back_populates="slot", uselist=False)


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=True, index=True)
    role = Column(String(32), nullable=False, default="patient")
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    patient = relationship("Patient", back_populates="conversations")
    messages = relationship("MessageLog", back_populates="conversation")


class MessageLog(Base):
    __tablename__ = "message_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id"), nullable=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=True, index=True)
    telegram_id = Column(Integer, nullable=False, index=True)
    direction = Column(String(16), nullable=False)
    message_type = Column(String(32), nullable=False)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, index=True)

    conversation = relationship("Conversation", back_populates="messages")
    patient = relationship("Patient", back_populates="message_logs")


class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    profile_id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), unique=True, nullable=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    data = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    patient = relationship("Patient", back_populates="profile")


class FsmSession(Base):
    __tablename__ = "fsm_sessions"

    session_id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    role = Column(String(32), nullable=False, default="patient", index=True)
    state = Column(String(64), nullable=False)
    data_json = Column(JSON, nullable=True)
    slot_options_json = Column(JSON, nullable=True)
    slot_index = Column(Integer, nullable=False, default=0)
    slot_json = Column(JSON, nullable=True)
    priority_json = Column(JSON, nullable=True)
    finalized_appointment_id = Column(String(64), nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
