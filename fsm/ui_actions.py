"""UI actions returned by FSM — handlers map these to Telegram keyboards."""
from __future__ import annotations

from enum import Enum, auto


class UIAction(Enum):
    NONE = auto()
    SHOW_URGENCY = auto()
    SHOW_TIME = auto()
    SHOW_SPECIALTY = auto()
    SHOW_CONFIRM = auto()
    SHOW_MAIN_MENU = auto()
