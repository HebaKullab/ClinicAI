from datetime import datetime, timedelta

import pytest

from scheduler.scheduler import AppointmentSlot, _build_slot_sort_key


def make_slot(slot_id, when, specialty="neurology", priority_class=None, doctor_id=None):
    return AppointmentSlot(
        slot_id=slot_id,
        slot_datetime=when,
        specialty=specialty,
        priority_class=priority_class,
        status="available",
        doctor_id=doctor_id,
        notes=f"doctor_id:{doctor_id}" if doctor_id else None,
    )


@pytest.fixture
def ranking_slots():
    now = datetime(2026, 7, 11, 8, 0, 0)
    soon_busy = make_slot(1, now + timedelta(hours=2))
    later_quiet = make_slot(2, now + timedelta(days=3))
    clinic_load = {
        ("neurology", soon_busy.slot_datetime.date()): 5,
        ("neurology", later_quiet.slot_datetime.date()): 0,
    }
    return soon_busy, later_quiet, clinic_load


def _rank(slots, priority_class, clinic_load):
    return sorted(
        slots,
        key=lambda s: _build_slot_sort_key(
            s,
            pref_day=None,
            priority_class=priority_class,
            clinic_load=clinic_load,
            doctor_load={},
            utilization={},
        ),
    )


def test_p1_prefers_earliest_slot_despite_higher_load(ranking_slots):
    soon_busy, later_quiet, clinic_load = ranking_slots
    ranked = _rank([later_quiet, soon_busy], "P1", clinic_load)
    assert ranked[0].slot_id == 1


@pytest.mark.parametrize("priority_class", ["P3", "P2"])
def test_non_p1_prefers_lower_load_over_earliest_slot(ranking_slots, priority_class):
    soon_busy, later_quiet, clinic_load = ranking_slots
    ranked = _rank([later_quiet, soon_busy], priority_class, clinic_load)
    assert ranked[0].slot_id == 2


@pytest.mark.parametrize("priority_class", ["P1", "P2", "P3"])
def test_identical_load_falls_back_to_earliest(ranking_slots, priority_class):
    soon_busy, later_quiet, _ = ranking_slots
    tied_load = {
        ("neurology", soon_busy.slot_datetime.date()): 2,
        ("neurology", later_quiet.slot_datetime.date()): 2,
    }
    ranked = _rank([later_quiet, soon_busy], priority_class, tied_load)
    assert ranked[0].slot_id == 1
