"""Unit tests for nlp/booking_agent.py."""
from nlp.booking_agent import (
    apply_extracted_to_data,
    fallback_reply,
    merge_rule_extracted,
    missing_required_fields,
    reply_claims_booking_done,
    run_booking_turn,
)
from tests.helpers import run_async


def test_merge_rule_extracted_name():
    out = merge_rule_extracted("انا اسمي أحمد", {})
    assert out.get("name")
    assert "حمد" in out["name"]


def test_merge_rule_tingling_complaint():
    out = merge_rule_extracted("عندي تنميل خفيف في ايدي", {})
    assert out.get("complaint")
    assert "تنميل" in str(out["complaint"])


def test_merge_rule_medium_urgency():
    out = merge_rule_extracted("متوسط", {"name": "منال", "complaint": {"raw": "تنميل"}})
    assert out.get("urgency_score") == 0.5


def test_reply_claims_booking_done():
    assert reply_claims_booking_done("✅ تم تأكيد حجزك وحفظ ملفك في النظام!")
    assert reply_claims_booking_done("تم الحجز بنجاح رقم الحجز 123")
    assert not reply_claims_booking_done("وجدت موعداً مناسباً! مناسبلك؟")


def test_apply_extracted_complaint_and_urgency():
    data: dict = {}
    apply_extracted_to_data(
        data,
        {"complaint": "صداع", "urgency": "روتيني", "time_pref": "بكرا"},
        score_from_label=lambda _: 0.2,
    )
    assert data["complaint"]["raw"] == "صداع"
    assert data["urgency_score"] == 0.2
    assert data["time_pref"]["phrase"] == "بكرا"


def test_missing_required_fields():
    data = {"name": "سارة", "complaint": {"raw": "كحة"}}
    missing = missing_required_fields(data, ["name", "complaint", "urgency_score", "time_pref"])
    assert "urgency_score" in missing
    assert "time_pref" in missing


def test_fallback_reply_asks_for_name():
    turn = fallback_reply({}, ["name", "complaint", "urgency_score", "time_pref"], "مرحبا")
    assert "اسم" in turn.reply
    assert turn.intent == "continue"


def test_run_booking_turn_offline():
    turn = run_async(run_booking_turn("أحمد", "CHATTING", {}, []))
    assert turn.reply
    assert turn.intent == "continue"
