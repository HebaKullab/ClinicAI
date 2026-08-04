"""Dependency injection for patient booking FSM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class BookingServices:
    classify: Callable
    classify_with_fallback: Callable
    score: Callable
    find_slots: Callable
    book: Callable
    enqueue_waitlist: Callable
    detect_unsupported: Callable
    gemini: Optional[object] = None

    @classmethod
    def default(cls) -> BookingServices:
        from database import crud
        from nlp.gemini_client import gemini
        from scheduler.classifier import (
            classify_specialty,
            classify_with_gemini_fallback,
            detect_unsupported_specialty,
        )
        from scheduler.priority import score_and_classify
        from scheduler.scheduler import enqueue_waitlist

        return cls(
            classify=classify_specialty,
            classify_with_fallback=classify_with_gemini_fallback,
            score=score_and_classify,
            find_slots=crud.find_available_slots,
            book=crud.create_patient_file_and_book,
            enqueue_waitlist=enqueue_waitlist,
            detect_unsupported=detect_unsupported_specialty,
            gemini=gemini,
        )
