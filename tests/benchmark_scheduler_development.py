from collections import Counter
from scheduler.classifier import classify_specialty, classify_with_gemini_fallback
import asyncio
from nlp.gemini_client import gemini
from scheduler.priority import score_and_classify
import time
from statistics import mean, median

import json
import os
import logging

logging.getLogger("scheduler.classifier").setLevel(logging.CRITICAL)

with open(os.path.join(os.path.dirname(__file__), "data/benchmarks.json"), "r", encoding="utf-8") as f:
    TEST_DATASET = json.load(f)


from tests.helpers import infer_urgency_score
def measure_performance(dataset, run_case_fn, warmup: int = 5):
    """
    Measures benchmark runtime stats.

    Args:
        dataset: list of test cases
        run_case_fn: function(case) -> any  (runs one case prediction)
        warmup: warm-up iterations before actual timing

    Returns:
        dict with performance metrics
    """
    # Warmup to stabilize interpreter/cache behavior
    for i in range(min(warmup, len(dataset))):
        run_case_fn(dataset[i])

    per_case_times = []
    t0 = time.perf_counter()

    for case in dataset:
        c0 = time.perf_counter()
        run_case_fn(case)
        c1 = time.perf_counter()
        per_case_times.append((c1 - c0) * 1000.0)  # ms

    t1 = time.perf_counter()
    total_sec = t1 - t0
    total_cases = len(dataset)
    throughput = (total_cases / total_sec) if total_sec > 0 else 0.0

    return {
        "total_cases": total_cases,
        "total_time_sec": round(total_sec, 4),
        "avg_case_ms": round(mean(per_case_times), 3) if per_case_times else 0.0,
        "median_case_ms": round(median(per_case_times), 3) if per_case_times else 0.0,
        "p95_case_ms": round(sorted(per_case_times)[int(0.95 * (len(per_case_times)-1))], 3) if per_case_times else 0.0,
        "throughput_cases_per_sec": round(throughput, 2),
    }

def predict_case(row: dict) -> dict:
    """Run classifier + priority for one benchmark row."""
    text = row["input"]
    cls = classify_specialty(text)

    if cls.get("method") == "default" and gemini is not None and getattr(gemini, "_available", False):
        try:
            cls_gemini = asyncio.run(classify_with_gemini_fallback(text, gemini))
            if cls_gemini.get("method") == "gemini":
                cls = cls_gemini
        except Exception:
            pass

    pred_clinic = cls["specialty"]
    urgency = infer_urgency_score(text)

    data = {
        "complaint": {
            "raw": text,
            "urgency_score": row.get("urgency_score", 0.2),
            "specialty": pred_clinic,
        },
        "urgency_score": row.get("urgency_score", 0.2),
        "is_followup": row.get("is_followup", False),
        "specialty_hint": row.get("specialty_hint", pred_clinic),
        "time_pref": row.get("time_pref", {"date": None}),
    }
    pr = score_and_classify(data)

    return {
        "pred_clinic": pred_clinic,
        "pred_prio": pr.priority_class,
        "method": cls.get("method", "unknown"),
        "confidence": cls.get("confidence", 0.0),
        "priority_score": pr.score,
        "breakdown": pr.breakdown,
    }


def run():
    clinic_ok = 0
    prio_ok = 0
    wrong = []
    method_counts: Counter = Counter()
    confidence_by_method: dict[str, list[float]] = {}

    for row in TEST_DATASET:
        text = row["input"]
        exp_clinic = row["expected"]["Clinic"]
        exp_prio = row["expected"]["Priority"]

        result = predict_case(row)
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
                "breakdown": result["breakdown"],
            })

    n = len(TEST_DATASET)
    print(f"Total: {n}")
    print(f"Clinic accuracy:   {clinic_ok}/{n} = {clinic_ok/n:.2%}")
    print(f"Priority accuracy: {prio_ok}/{n} = {prio_ok/n:.2%}")

    print("\n=== Classification methods ===")
    for method, count in method_counts.most_common():
        scores = confidence_by_method.get(method, [])
        avg_conf = mean(scores) if scores else 0.0
        print(f"{method}: {count}/{n} ({count/n:.1%}), avg confidence={avg_conf:.2f}")

    if wrong:
        print("\nMismatches:")
        for w in wrong:
            print(
                f"[{w['id']}] exp={w['expected']} pred={w['pred']} "
                f"method={w['method']} confidence={w['confidence']:.2f} "
                f"priority_score={w['priority_score']:.3f} | {w['text']}"
            )
            print(f"Details   : {w['breakdown']}") # print f1, f2, f3, f4, f5

    perf = measure_performance(TEST_DATASET, run_one_case, warmup=5)

    print("\n=== Performance ===")
    for k, v in perf.items():
        print(f"{k}: {v}")

def run_one_case(case):
    result = predict_case(case)
    return (
        result["pred_clinic"],
        result["pred_prio"],
        result["method"],
        result["confidence"],
    )

if __name__ == "__main__":
    run()