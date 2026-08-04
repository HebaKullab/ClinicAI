from datetime import date, timedelta

import pytest

from scheduler.priority import THETA_P1, WEIGHTS, score_and_classify


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)


def test_empty_data_defaults_to_p3():
    assert score_and_classify({}).priority_class == "P3"


def test_high_urgency_explicit_complaint_score_pushes_p1():
    data = {
        "complaint": {"urgency_score": 1.0},
        "urgency_score": 1.0,
        "is_followup": False,
        "specialty_hint": "neurology",
        "time_pref": {"date": date.today().isoformat()},
    }
    assert score_and_classify(data).priority_class == "P1"


def test_low_everything_gives_p3():
    data = {
        "complaint": {"urgency_score": 0.1},
        "urgency_score": 0.1,
        "is_followup": False,
        "specialty_hint": "dermatology",
        "time_pref": {"date": (date.today() + timedelta(days=30)).isoformat()},
    }
    assert score_and_classify(data).priority_class == "P3"


def test_boundary_just_below_theta_p1():
    data = {
        "complaint": {"urgency_score": 0.60},
        "urgency_score": 0.60,
        "is_followup": False,
        "specialty_hint": "orthopedics",
        "time_pref": {"date": (date.today() + timedelta(days=7)).isoformat()},
    }
    result = score_and_classify(data)
    assert result.score < THETA_P1
    assert result.priority_class == "P2"


@pytest.mark.parametrize(
    "time_pref",
    [
        "أي وقت متاح",
        None,
        {"phrase": "أي وقت متاح"},
        {"date": "not-a-date"},
    ],
)
def test_timing_score_variants_no_crash(time_pref):
    score_and_classify({"time_pref": time_pref})


def test_timing_score_past_date_clamped_to_zero_delta():
    past = (date.today() - timedelta(days=10)).isoformat()
    result = score_and_classify({"time_pref": {"date": past}})
    assert result.breakdown["f5"] == 1.0


def test_specialty_score_unknown_specialty_falls_back():
    result = score_and_classify({"specialty_hint": "made_up_specialty_xyz"})
    assert result.breakdown["f4"] == 0.3


def test_complaint_not_dict_no_crash():
    score_and_classify({"complaint": "just a raw string, not a dict"})


def test_urgency_score_non_numeric_no_crash():
    score_and_classify({"urgency_score": "very high"})
