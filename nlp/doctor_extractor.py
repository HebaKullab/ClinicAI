"""
nlp/doctor_extractor.py — Extract structured session fields from doctor voice notes.
"""
import re
import json
from pathlib import Path
from nlp.normalizer import normalize

_DATA = Path(__file__).parent.parent / "data" / "levantine"

NUM_MAP = {
    "مية": 100, "مائه": 100, "مئه": 100, "مئتين": 200,
    "تلاتميه": 300, "اربعميه": 400, "خمسميه": 500,
    "خمسه": 5, "عشره": 10, "خمستعش": 15, "عشرين": 20,
    "خمسين": 50, "مية وخمسه": 125, "مية وخمسين": 150,
}

FREQ_MAP = {
    "مره في اليوم":       "مرة يومياً",
    "مره يوميا":          "مرة يومياً",
    "مرتين في اليوم":    "مرتين يومياً",
    "مرتين يوميا":       "مرتين يومياً",
    "ثلاث مرات في اليوم": "ثلاث مرات يومياً",
    "كل ثماني ساعات":    "كل 8 ساعات",
    "كل اثنعش ساعه":     "كل 12 ساعة",
    "عند الحاجه":         "عند الحاجة",
    "قبل الاكل":          "قبل الأكل",
    "بعد الاكل":          "بعد الأكل",
}

INV_KEYWORDS = {
    "تحليل دم": "CBC", "صوره دم": "CBC",
    "سكر الدم": "Blood glucose", "ضغط الدم": "Blood pressure",
    "اشعه": "X-Ray", "رنين": "MRI", "سونار": "Ultrasound",
    "ايكو": "Echocardiogram", "تخطيط قلب": "ECG",
    "كلسترول": "Lipid profile", "وظائف الكلى": "Kidney function",
    "وظائف الكبد": "Liver function", "هرمونات": "Hormone panel",
}


def extract_session_fields(raw: str) -> dict:
    norm = normalize(raw)
    meds_raw = _load("medications.json")
    return {
        "patient_name":     _name(norm),
        "chief_complaint":  _complaint(norm),
        "symptom_duration": _duration(norm),
        "diagnosis":        _diagnosis(norm),
        "medications":      _medications(norm, meds_raw),
        "investigations":   _investigations(norm),
        "followup_days":    _followup(norm),
    }


def _load(name):
    p = _DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _name(t):
    for p in [
        r"المريض\s+(?:اسمه\s+)?(\S+(?:\s+\S+){0,3})",
        r"اسم المريض\s+(\S+(?:\s+\S+){0,3})",
    ]:
        m = re.search(p, t)
        if m:
            return m.group(1).strip().rstrip("،,")
    return None


def _complaint(t):
    for p in [r"شاكي من\s+(.+?)(?:منذ|من|لمده|،|$)",
              r"يشكو من\s+(.+?)(?:منذ|من|،|$)",
              r"عنده\s+(.+?)(?:منذ|من|،|$)"]:
        m = re.search(p, t)
        if m: return m.group(1).strip()
    return None


def _duration(t):
    m = re.search(r"منذ\s+(.+?)(?:،|وعنده|$)", t)
    if m: return m.group(1).strip()
    m = re.search(r"من\s+(\d+\s+(?:يوم|يومين|ايام|اسبوع|اسبوعين|شهر|اشهر))", t)
    return m.group(1) if m else None


def _diagnosis(t):
    m = re.search(r"التشخيص\s+(.+?)(?:،|\.|$)", t)
    return m.group(1).strip() if m else None


def _medications(t, meds_raw):
    found = []
    for med_key, data in meds_raw.items():
        for variant in data.get("variants", []):
            if variant in t:
                dose_m = re.search(rf"{re.escape(variant)}\s+(\S+)", t)
                dose_raw = dose_m.group(1) if dose_m else ""
                dose_num = NUM_MAP.get(dose_raw, dose_raw)
                freq = next((v for k, v in FREQ_MAP.items() if k in t), "غير محدد")
                dur_m = re.search(r"لمده\s+(\S+\s+(?:يوم|ايام|اسبوع|شهر))", t)
                found.append({
                    "name":      data.get("standard_name", med_key),
                    "dose":      f"{dose_num}{data.get('unit','')}",
                    "frequency": freq,
                    "duration":  dur_m.group(1) if dur_m else "غير محدد",
                    "route":     data.get("route", "فموي"),
                })
                break
    return found


def _investigations(t):
    return [{"name_ar": ar, "name_en": en}
            for ar, en in INV_KEYWORDS.items() if ar in t]


def _followup(t):
    MAP = {
        r"بعد\s+يومين": 2, r"بعد\s+ثلاثه ايام": 3,
        r"بعد\s+اسبوع": 7, r"بعد\s+اسبوعين": 14,
        r"بعد\s+شهر": 30, r"بعد\s+شهرين": 60,
        r"متابعه شهريه": 30,
    }
    for pat, days in MAP.items():
        if re.search(pat, t): return days
    return None
