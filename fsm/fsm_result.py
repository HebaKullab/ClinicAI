"""Helpers for FSM handler/test integration."""
from __future__ import annotations

from fsm.ui_actions import UIAction


def keyboard_for_action(action: UIAction):
    from bot.keyboards import (
        confirm_keyboard,
        main_menu_keyboard,
        specialty_keyboard,
        time_pref_keyboard,
        urgency_keyboard,
    )

    mapping = {
        UIAction.SHOW_URGENCY: urgency_keyboard,
        UIAction.SHOW_TIME: time_pref_keyboard,
        UIAction.SHOW_SPECIALTY: specialty_keyboard,
        UIAction.SHOW_CONFIRM: confirm_keyboard,
        UIAction.SHOW_MAIN_MENU: main_menu_keyboard,
    }
    factory = mapping.get(action)
    return factory() if factory else None
