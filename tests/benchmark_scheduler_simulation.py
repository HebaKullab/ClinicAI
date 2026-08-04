"""Benchmark scheduler classifier + priority against 120 Arabic cases.

Default mode mirrors production: rules first, then real Gemini fallback
(via scheduler._classify) when no rule matches.

Usage:
    python -m tests.benchmark_scheduler              # production path + Gemini
    python -m tests.benchmark_scheduler --rules-only   # rules only (no API)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.classifier import classify_specialty
from scheduler.priority import score_and_classify
from scheduler.scheduler import _classify, normalize_priority_class, sanitize_input
import json
import os

with open(os.path.join(os.path.dirname(__file__), "data/benchmarks.json"), "r", encoding="utf-8") as f:
    TEST_DATASET = json.load(f)


from tests.helpers import infer_urgency_score

def _build_patient_data(row: dict) -> dict:
    """Build FSM-like payload, then sanitize like plan_appointment()."""
    text = row["input"]
    urgency = row.get("urgency_score", 0.2)
    raw = {
        "complaint": {"raw": text},
        "urgency_score": urgency,
        "is_followup": row.get("is_followup", False), # القيمة الحقيقية من الملف
        "time_pref": row.get("time_pref", {"date": None, "phrase": "أي وقت"}), # التاريخ الحقيقي
    }
    safe = sanitize_input(raw)
    safe["complaint"]["urgency_score"] = urgency
    safe["complaint"]["specialty"] = None
    safe["specialty_hint"] = row.get("specialty_hint") # استخدام تلميح التخصص
    return safe

async def predict_case(row: dict, gemini_client=None, *, gemini_delay: float = 0.0) -> dict:
    """Production-aligned path: sanitize → _classify → score_and_classify."""
    text = row["input"]
    if gemini_client is not None and classify_specialty(text)["method"] == "default":
        if gemini_delay > 0:
            await asyncio.sleep(gemini_delay)
    safe_data = _build_patient_data(row)

    spec_result = await _classify(safe_data, gemini_client)
    safe_data["specialty_hint"] = spec_result["specialty"]
    safe_data["specialty_ar"] = spec_result["specialty_ar"]
    safe_data["complaint"]["specialty"] = spec_result["specialty"]

    pr = score_and_classify(safe_data)

    return {
        "pred_clinic": spec_result["specialty"],
        "pred_prio": normalize_priority_class(pr.priority_class),
        "method": spec_result.get("method", "unknown"),
        "confidence": float(spec_result.get("confidence", 0.0)),
        "priority_score": pr.score,
        "breakdown": pr.breakdown,
    }

async def measure_performance_async(
    dataset, predict_fn, gemini_client=None, *, gemini_delay: float = 0.0, warmup: int = 3,
):
    for i in range(min(warmup, len(dataset))):
        await predict_fn(dataset[i], gemini_client, gemini_delay=gemini_delay)

    per_case_times = []
    t0 = time.perf_counter()
    for case in dataset:
        c0 = time.perf_counter()
        await predict_fn(case, gemini_client, gemini_delay=gemini_delay)
        per_case_times.append((time.perf_counter() - c0) * 1000.0)
    total_sec = time.perf_counter() - t0
    total_cases = len(dataset)
    throughput = (total_cases / total_sec) if total_sec > 0 else 0.0

    return {
        "total_cases": total_cases,
        "total_time_sec": round(total_sec, 4),
        "avg_case_ms": round(mean(per_case_times), 3) if per_case_times else 0.0,
        "median_case_ms": round(median(per_case_times), 3) if per_case_times else 0.0,
        "p95_case_ms": round(sorted(per_case_times)[int(0.95 * (len(per_case_times) - 1))], 3) if per_case_times else 0.0,
        "throughput_cases_per_sec": round(throughput, 2),
    }


def measure_performance_rules_only(dataset, warmup: int = 5):
    """Timing for rules-only path (no network)."""
    def run_one(case):
        cls = classify_specialty(case["input"])
        urgency = infer_urgency_score(case["input"])
        data = _build_patient_data(case["input"])
        data["specialty_hint"] = cls["specialty"]
        score_and_classify(data)
        return cls["specialty"]

    for i in range(min(warmup, len(dataset))):
        run_one(dataset[i])

    per_case_times = []
    t0 = time.perf_counter()
    for case in dataset:
        c0 = time.perf_counter()
        run_one(case)
        per_case_times.append((time.perf_counter() - c0) * 1000.0)
    total_sec = time.perf_counter() - t0
    n = len(dataset)
    return {
        "total_cases": n,
        "total_time_sec": round(total_sec, 4),
        "avg_case_ms": round(mean(per_case_times), 3) if per_case_times else 0.0,
        "median_case_ms": round(median(per_case_times), 3) if per_case_times else 0.0,
        "p95_case_ms": round(sorted(per_case_times)[int(0.95 * (len(per_case_times) - 1))], 3) if per_case_times else 0.0,
        "throughput_cases_per_sec": round(n / total_sec, 2) if total_sec > 0 else 0.0,
    }
async def run(
    gemini_client=None,
    *,
    rules_only: bool = False,
    gemini_delay: float = 4.0,
    sample: int | None = None,
):
    mode = "rules-only" if rules_only else "production (rules + Gemini fallback)"
    dataset = TEST_DATASET[:sample] if sample else TEST_DATASET

    print(f"Mode: {mode}")
    print(f"Cases: {len(dataset)}/{len(TEST_DATASET)}")
    if rules_only:
        print("Gemini: disabled (--rules-only)")
    elif gemini_client is None:
        print("Gemini: unavailable (missing API key or library) — using rules-only fallback")
    else:
        print("Gemini: enabled (real API, same path as plan_appointment)")
        if gemini_delay > 0:
            print(f"Gemini delay: {gemini_delay}s between fallback calls (rate-limit safe)")

    clinic_ok = 0
    prio_ok = 0
    wrong = []
    method_counts: Counter = Counter()
    confidence_by_method: dict[str, list[float]] = {}
    gemini_candidates = 0

    for row in dataset:
        text = row["input"]
        exp_clinic = row["expected"]["Clinic"]
        exp_prio = row["expected"]["Priority"]

        rule_only = classify_specialty(text)
        if rule_only["method"] == "default":
            gemini_candidates += 1

        result = await predict_case(row, gemini_client, gemini_delay=0.0 if rules_only else gemini_delay)
        pred_clinic = result["pred_clinic"]
        pred_prio = result["pred_prio"]
        method = result["method"]
        confidence = result["confidence"]

        method_counts[method] += 1
        confidence_by_method.setdefault(method, []).append(confidence)

        c_ok = pred_clinic == exp_clinic
        p_ok = pred_prio == exp_prio
        clinic_ok += int(c_ok)
        prio_ok += int(p_ok)

        if not (c_ok and p_ok):
            wrong.append({
                "id": row["id"],
                "text": text,
                "expected": (exp_clinic, exp_prio),
                "pred": (pred_clinic, pred_prio),
                "method": method,
                "confidence": confidence,
                "priority_score": result["priority_score"],
                "breakdown": result.get("breakdown"),
            })

    n = len(dataset)
    print(f"\nTotal: {n}")
    print(f"Clinic accuracy:   {clinic_ok}/{n} = {clinic_ok/n:.2%}")
    print(f"Priority accuracy: {prio_ok}/{n} = {prio_ok/n:.2%}")

    print("\n=== Classification methods ===")
    for method, count in method_counts.most_common():
        scores = confidence_by_method.get(method, [])
        avg_conf = mean(scores) if scores else 0.0
        print(f"{method}: {count}/{n} ({count/n:.1%}), avg confidence={avg_conf:.2f}")

    if not rules_only and gemini_client is not None:
        gemini_used = method_counts.get("gemini", 0)
        print(f"\nGemini fallback candidates (rule=default): {gemini_candidates}/{n}")
        print(f"Gemini successfully classified: {gemini_used}/{gemini_candidates or n}")

    if wrong:
        print("\nMismatches:")
        for w in wrong:
            print(
                f"[{w['id']}] exp={w['expected']} pred={w['pred']} "
                f"method={w['method']} confidence={w['confidence']:.2f} "
                f"priority_score={w['priority_score']:.3f} | {w['text']}"
                f"score={w['priority_score']:.3f} | breakdown={w['breakdown']}"
            )

    print("\n=== Performance ===")
    if rules_only or gemini_client is None:
        perf = measure_performance_rules_only(dataset)
        print("(rules-only timing, no network)")
    else:
        perf = await measure_performance_async(
            dataset, predict_case, gemini_client, gemini_delay=gemini_delay,
        )
        print("(includes real Gemini API latency)")
    for k, v in perf.items():
        print(f"{k}: {v}")


def _resolve_gemini_client(rules_only: bool):
    if rules_only:
        return None
    try:
        from nlp.gemini_client import gemini
        return gemini if gemini._available else None
    except Exception as exc:
        print(f"Gemini init failed: {exc}")
        return None


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Benchmark scheduler on 120 Arabic cases")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Use classify_specialty only (no Gemini API calls)",
    )
    parser.add_argument(
        "--gemini-delay",
        type=float,
        default=4.0,
        help="Seconds to wait before each Gemini fallback call (default 4, free-tier safe)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only the first N cases (quick smoke test)",
    )
    args = parser.parse_args()
    client = _resolve_gemini_client(args.rules_only)
    asyncio.run(
        run(
            client,
            rules_only=args.rules_only,
            gemini_delay=max(0.0, args.gemini_delay),
            sample=args.sample,
        )
    )


if __name__ == "__main__":
    main()