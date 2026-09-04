"""
nlp/extractor.py — Extract structured fields from normalized Arabic text.
"""
import re
import json
from pathlib import Path
from datetime import date, timedelta
from nlp.normalizer import normalize

_DATA = Path(__file__).parent.parent / "data" / "levantine"


def _load(name: str) -> dict:
    p = _DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def extract_patient_fields(raw_text: str) -> dict:
    norm = normalize(raw_text)
    symptoms  = _load("symptoms.json")
    time_map  = _load("time_phrases.json")

    complaint = _match_complaint(norm, symptoms)
    return {
        "name":           _extract_name(norm),
        "complaint":      complaint,
        "urgency_score":  _score_urgency(norm, complaint),
        "specialty_hint": complaint.get("specialty") if complaint else None,
        "time_pref":      _extract_time(norm, time_map),
        "is_followup":    _is_followup(norm),
    }


def _extract_name(text: str):
    for p in [r"اسمي\s+(\S+(?:\s+\S+)?)", r"انا\s+(\S+)", r"المريض\s+(\S+(?:\s+\S+)?)"]:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    return None


def _match_complaint(text: str, symptoms: dict) -> dict | None:
    for key, data in symptoms.items():
        for variant in data.get("variants", []):
            if variant in text:
                return {
                    "raw":           variant,
                    "category":      data.get("category", "general"),
                    "urgency_score": data.get("urgency_score", 0.2),
                    "specialty":     data.get("specialty", "general_practice"),
                }
    return None


def _score_urgency(text: str, complaint: dict | None) -> float:
    # Explicit patient priority labels beat symptom baseline.
    if any(w in text for w in ["عاجل", "طارئ", "فوري", "خطير"]):
        return 0.9
    if any(w in text for w in ["متوسط", "عادي"]):
        return 0.5
    if any(w in text for w in ["روتيني", "مش عاجل", "اي وقت", "بالراحه"]):
        return 0.2

    base = complaint.get("urgency_score", 0.3) if complaint else 0.3
    urgent = ["شديد", "الان"]
    for w in urgent:
        if w in text:
            base = min(base + 0.2, 1.0)
    return round(base, 2)


def _extract_time(text: str, time_map: dict) -> dict:
    today = date.today()
    for phrase, delta in time_map.items():
        if phrase in text:
            return {"date": str(today + timedelta(days=int(delta))), "phrase": phrase}
    day_names = {
        "الاحد": 6, "الاثنين": 0, "الثلاثاء": 1,
        "الاربعاء": 2, "الخميس": 3, "الجمعه": 4, "السبت": 5,
    }
    for name, wd in day_names.items():
        if name in text:
            ahead = (wd - today.weekday()) % 7 or 7
            return {"date": str(today + timedelta(days=ahead)), "phrase": name}
    return {"date": None, "phrase": None}


def _is_followup(text: str) -> bool:
    return any(w in text for w in ["متابعه", "مراجعه", "نتيجه", "المره اللي فاتت"])
