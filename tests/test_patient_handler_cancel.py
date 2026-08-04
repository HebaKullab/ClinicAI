"""Unit tests for patient handler cancel-intent detection."""
from __future__ import annotations

import pytest

from bot.handlers import patient as patient_handler


@pytest.mark.parametrize(
    "text",
    [
        "❌ إلغاء موعد",
        "اريد الغاء الموعد",
        "أريد إلغاء الموعد",
        "بدي الغي الموعد",
        "الغاء الحجز",
        "cancel appointment",
    ],
)
def test_menu_cancel_request_recognizes_explicit_and_natural_phrases(text: str):
    assert patient_handler._is_menu_cancel_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "❌ إلغاء",
        "❌",
        "لا",
    ],
)
def test_menu_cancel_request_rejects_confirm_keyboard_abort_only(text: str):
    assert patient_handler._is_menu_cancel_request(text) is False


def test_menu_cancel_request_rejects_unrelated_text():
    assert patient_handler._is_menu_cancel_request("مرحبا") is False
