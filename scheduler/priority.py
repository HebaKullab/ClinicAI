"""
scheduler/priority.py — Task: "Priority engine: from conversation to priority score"

Converts raw extracted patient fields into a weighted score,
then maps that score to a priority class (P1 / P2 / P3).
All weights are expert-defined and sum to 1.0.
"""
from dataclasses import dataclass
from datetime import date
from typing import Dict, Any

# ── Weight table (must sum to 1.0) ────────────────────────────────────────────
WEIGHTS = {
    "complaint":  0.38,   # f1 — what the patient has (most important signal)
    "urgency":    0.28,   # f2 — patient's stated urgency
    "followup":   0.12,   # f3 — follow-up vs new case
    "specialty":  0.12,   # f4 — specialty urgency level
    "timing":     0.10,   # f5 — how soon they want the appointment
}

# ── Priority thresholds ────────────────────────────────────────────────────────
THETA_P1 = 0.66 # score >= 0.66 →  P1 (High / عاجل)
THETA_P2 = 0.38 # score >= 0.38 →  P2 (Medium / متوسط)
                  # score <  0.38 →  P3 (Routine / روتيني)

# ── Specialty urgency levels (f4 encoding) ────────────────────────────────────
SPECIALTY_SCORES = {
    # "cardiology":       1.0,
    "neurology":        0.9,
    "orthopedics":      0.5,
    "gynecology":       0.55,
    # "pediatrics":       0.6,
    # "ophthalmology":    0.45,
    "dermatology":      0.3,
    # "dentistry":        0.35,
    # "pulmonology":      0.60,
    "gastroenterology": 0.45,
    "chronic_diseases": 0.55,
    "elderly":          0.65,
    "general_practice": 0.3,
}

# ── Timing preference encoding (f5) ───────────────────────────────────────────
TIMING_SCORES = {
    0:  1.0,   # today
    1:  0.85,  # tomorrow
    2:  0.7,   # day after tomorrow
    3:  0.55,  # within 3 days
    7:  0.4,   # this week
    14: 0.25,  # next week
    30: 0.1,   # this month
}

# ── Complaint text urgency hints (used when urgency_score absent) ──
COMPLAINT_URGENCY_KEYWORDS = {
    "high":   ["حاد", "شديد", "جدا", "مفاجئ", "فجأة", "لا أستطيع", "لا يستطيع", "عاجل", "أزمة"],
    "medium": ["متوسط", "مزمن", "متكرر", "مستمر"],
    "low":    ["خفيف", "بسيط", "أحيان", "دوري", "روتيني", "متابعة"],
}
COMPLAINT_KEYWORD_SCORES = {"high": 0.85, "medium": 0.55, "low": 0.25}
DEFAULT_COMPLAINT_SCORE = 0.2

@dataclass
class PriorityResult:
    score:        float
    priority_class: str          # "P1" | "P2" | "P3"
    label_ar:     str            # Arabic label for bot responses
    label_color:  str            # for dashboard: red / yellow / green
    breakdown:    dict           # individual factor scores for audit


def score_and_classify(data: dict) -> PriorityResult:
    """
    Main entry point.  data = accumulated FSM fields dict.

    Expected keys (all optional — missing → 0):
        complaint      → {"urgency_score": float, "specialty": str, ...}
        urgency_score  → float 0–1  (from patient's stated urgency words)
        is_followup    → bool
        specialty_hint → str
        time_pref      → {"date": "YYYY-MM-DD"} or None
    """
    f1 = _complaint_score(data)
    f2 = _urgency_score(data)
    f3 = _followup_score(data)
    f4 = _specialty_score(data)
    f5 = _timing_score(data)

    score = (
        WEIGHTS["complaint"] * f1
        + WEIGHTS["urgency"]  * f2
        + WEIGHTS["followup"] * f3
        + WEIGHTS["specialty"] * f4
        + WEIGHTS["timing"]   * f5
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    PRIORITY_LABELS_AR = {
        "P1": "عاجل",
        "P2": "متوسط",
        "P3": "روتيني",
    }

    if score >= THETA_P1:
        cls, label_ar, color = "P1", PRIORITY_LABELS_AR["P1"], "red"
    elif score >= THETA_P2:
        cls, label_ar, color = "P2", PRIORITY_LABELS_AR["P2"], "yellow"
    else:
        cls, label_ar, color = "P3", PRIORITY_LABELS_AR["P3"], "green"

    return PriorityResult(
        score=score,
        priority_class=cls,
        label_ar=label_ar,
        label_color=color,
        breakdown={"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5},
    )

def _score_from_complaint_text(raw: str) -> float | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    for level in ("high", "medium", "low"):
        if any(kw in text for kw in COMPLAINT_URGENCY_KEYWORDS[level]):
            return COMPLAINT_KEYWORD_SCORES[level]
    return None

# ── Factor encoders ────────────────────────────────────────────────────────────

def _complaint_score(data: Dict[str, Any]) -> float:
    complaint = data.get("complaint") or {}
    raw = ""
    if isinstance(complaint, dict):
        # Prefer explicit urgency score if available
        if "urgency_score" in complaint:
            try:
                return float(complaint["urgency_score"])
            except (TypeError, ValueError):
                return 0.2
        # Otherwise fallback by complaint category if available
        category = str(complaint.get("category", "")).lower().strip()
        category_map = {
            "critical": 1.0,
            "high": 0.85,
            "medium": 0.55,
            "low": 0.25,
        }
        if category in category_map:
            return category_map[category]
        raw = str(complaint.get("raw", "")).strip()
        text_score = _score_from_complaint_text(raw)
        if text_score is not None:
            return text_score
    return DEFAULT_COMPLAINT_SCORE

def _urgency_score(data: Dict[str, Any]) -> float:
    try:
        return float(data.get("urgency_score", 0.3))
    except (TypeError, ValueError):
        return 0.3


def _followup_score(data: Dict[str, Any]) -> float:
    # Follow-ups get a small bump — they already have a relationship with the clinic
    return 0.5 if data.get("is_followup") else 0.2


def _specialty_score(data: Dict[str, Any]) -> float:
    # Use specialty from complaint first, then NLP hint
    complaint = data.get("complaint") or {}
    if not isinstance(complaint, dict):
        complaint = {}
    specialty = (
        data.get("specialty_hint")
        or complaint.get("specialty")
        or "general_practice"
    )
    specialty = str(specialty).lower().strip()
    return SPECIALTY_SCORES.get(specialty, 0.3)

def _timing_score(data: Dict[str, Any]) -> float:
    time_pref = data.get("time_pref")
    # Type-safety first
    if not isinstance(time_pref, dict):
        return 0.4
    pref_date_raw = time_pref.get("date")
    if not pref_date_raw:
        return 0.4 # no explicit date preference
    try:
        pref_date = date.fromisoformat(str(pref_date_raw))
        delta = (pref_date - date.today()).days
        delta = max(delta, 0)
    except (ValueError, TypeError):
        return 0.4

    for threshold in sorted(TIMING_SCORES.keys()):
        if delta <= threshold:
            return TIMING_SCORES[threshold]
    return 0.1
