"""Scheduler package — Tier-1 real-time booking."""

from scheduler.scheduler import (
    AppointmentSlot,
    ScheduleDecision,
    WaitlistEntry,
    plan_appointment,
)
from scheduler.slot_policy import (
    BLOCK_ACCESS,
    FALLBACK_SPECIALTY,
    WAVE_HORIZON_DAYS,
    filter_by_block_rules,
    filter_by_wave_rules,
    rank_slots,
    select_best_slot,
    select_slots,
)

__all__ = [
    "plan_appointment",
    "AppointmentSlot",
    "ScheduleDecision",
    "WaitlistEntry",
    "BLOCK_ACCESS",
    "FALLBACK_SPECIALTY",
    "WAVE_HORIZON_DAYS",
    "filter_by_block_rules",
    "filter_by_wave_rules",
    "rank_slots",
    "select_best_slot",
    "select_slots",
]
