"""Unit tests for fsm/patient_fsm.py — patient booking FSM."""
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.keyboards import specialty_keyboard
from fsm.patient_fsm import FIELD_QUESTIONS_AR, REQUIRED_FIELDS, SPECIALTY_LABEL_TO_KEY, PatientFSM, State
from scheduler.classifier import SPECIALTY_NAMES_AR
from tests.helpers import run_async, unpack_fsm


def _high_confidence_classify(*args, **kwargs):
    return {
        "specialty": "general_practice",
        "specialty_ar": "الطب العام",
        "method": "rule",
        "confidence": 0.95,
    }


@pytest.fixture
def patient_fsm():
    fsm = PatientFSM(user_id=80002)
    fsm.services.classify = _high_confidence_classify
    return fsm


def test_parse_time_label_today_and_tomorrow():
    fsm = PatientFSM(user_id=80001)
    today = fsm._parse_time_label("بدي موعد اليوم")
    assert today["date"] == str(date.today())
    tomorrow = fsm._parse_time_label("بكرا")
    assert tomorrow["date"] == str(date.today() + timedelta(days=1))


def test_parse_specialty_label_arabic():
    fsm = PatientFSM(user_id=80001)
    assert fsm._parse_specialty_label("عظام") == "orthopedics"
    assert fsm._parse_specialty_label("هضمي") == "gastroenterology"


def test_absorb_urgency_levels():
    fsm = PatientFSM(user_id=80001)
    fsm._absorb_urgency("عاجل جدا")
    assert fsm.data["urgency_score"] >= 0.85

    fsm.data.clear()
    fsm._absorb_urgency("روتيني")
    assert fsm.data["urgency_score"] <= 0.25


def test_missing_fields_detects_empty_time_pref():
    fsm = PatientFSM(user_id=80001)
    fsm.data = {
        "name": "أحمد",
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.4,
        "time_pref": {"date": None, "phrase": ""},
    }
    assert "time_pref" in fsm._missing_fields()


def test_reset_clears_booking_state():
    fsm = PatientFSM(user_id=80001)
    fsm.state = State.FINALIZED
    fsm.data = {"name": "x"}
    fsm.slot = {"slot_id": 1}
    fsm._reset()
    assert fsm.state == State.CHATTING
    assert fsm.data == {}
    assert fsm.chat_history == []
    assert fsm.slot is None


def test_greeting_asks_for_name(patient_fsm):
    reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("مرحبا")))
    assert patient_fsm.state == State.CHATTING
    assert FIELD_QUESTIONS_AR["name"] in reply
    assert keyboard is None


def test_collect_name_then_complaint(patient_fsm):
    run_async(patient_fsm.handle("start"))
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("سارة محمود")))
    assert patient_fsm.state == State.CHATTING
    assert "سارة" in reply


def test_collect_complaint_from_plain_text(patient_fsm):
    patient_fsm.state = State.CHATTING
    reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("عندي صداع من يومين")))
    assert patient_fsm.state == State.CHATTING
    assert "complaint" in patient_fsm.data
    assert keyboard is None


def test_urgency_callback_moves_to_time(patient_fsm):
    patient_fsm.state = State.CHATTING
    patient_fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}
    reply, keyboard = unpack_fsm(run_async(patient_fsm.handle_callback("urgency:P2")))
    assert patient_fsm.state == State.CHATTING
    assert patient_fsm.data["urgency_score"] == 0.5
    assert keyboard is None


@patch("fsm.patient_fsm.gemini")
@patch("fsm.patient_fsm.score_and_classify")
def test_full_flow_reaches_confirm_with_slot(mock_score, mock_gemini, patient_fsm):
    mock_gemini._available = False
    mock_gemini.extract_missing_field = AsyncMock(return_value=None)
    priority = MagicMock(
        priority_class="P3",
        score=0.35,
        label_ar="روتيني",
        breakdown={"f1": 0.3},
    )
    mock_score.return_value = priority

    slot_dt = datetime.utcnow() + timedelta(days=2)
    slot = MagicMock(
        slot_id=99,
        slot_datetime=slot_dt,
        specialty="general_practice",
        priority_class="P3",
        doctor=MagicMock(
            doctor_id=1,
            name="د. اختبار",
            specialty="general_practice",
            clinic_code="CLINIC-GP",
            clinic_name="عيادة عامة",
        ),
    )

    @contextmanager
    def fake_get_db():
        yield MagicMock()

    with patch("database.db.get_db", fake_get_db):
        patient_fsm.services.classify = _high_confidence_classify
        patient_fsm.services.score = mock_score
        patient_fsm.services.find_slots = MagicMock(return_value=[slot])
        patient_fsm.services.gemini = mock_gemini
        run_async(patient_fsm.handle("مرحبا"))
        run_async(patient_fsm.handle("أحمد"))
        run_async(patient_fsm.handle("صداع خفيف"))
        run_async(patient_fsm.handle("روتيني"))
        reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("بكرا")))

    assert patient_fsm.state == State.CONFIRM
    assert "وجدت موعد" in reply
    assert "درجة الأولوية" not in reply
    assert keyboard is None
    assert patient_fsm.slot["slot_id"] == 99


@patch("fsm.patient_fsm.score_and_classify")
def test_confirm_yes_finalizes_booking(mock_score, patient_fsm):
    priority = MagicMock(priority_class="P3", score=0.3, label_ar="روتيني", breakdown={})
    mock_score.return_value = priority

    patient_fsm.state = State.CONFIRM
    patient_fsm.data = {
        "name": "أحمد",
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.3,
        "time_pref": {"date": str(date.today() + timedelta(days=1)), "phrase": "بكرا"},
        "specialty_hint": "general_practice",
        "specialty_ar": "الطب العام",
        "priority_class": "P3",
    }
    patient_fsm.slot = {
        "slot_id": 5,
        "slot_datetime": datetime.utcnow() + timedelta(days=1),
        "doctor_name": "د. اختبار",
        "clinic_name": "عيادة",
    }
    patient_fsm.priority = priority

    appt = MagicMock()
    appt.appt_id = "appt_test_99"
    appt.appt_datetime = patient_fsm.slot["slot_datetime"]

    @contextmanager
    def fake_get_db():
        yield MagicMock()

    with patch("database.db.get_db", fake_get_db):
        patient_fsm.services.book = MagicMock(
            return_value={"appointment": appt, "slot_conflict": False, "booking_conflict": None},
        )
        reply, _ = unpack_fsm(run_async(patient_fsm.handle("نعم")))

    assert patient_fsm.state == State.FINALIZED
    assert "تم تأكيد حجزك" in reply
    assert patient_fsm.finalized_appointment_id == "appt_test_99"


def test_confirm_no_cancels(patient_fsm):
    patient_fsm.state = State.CONFIRM
    patient_fsm.slot = {"slot_id": 1}
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("لا")))
    assert patient_fsm.state == State.CANCELLED
    assert "ألغ" in reply or "إلغاء" in reply


def test_low_confidence_auto_picks_specialty(patient_fsm):
    patient_fsm.state = State.CHATTING
    patient_fsm.data = {
        "name": "ليلى",
        "complaint": {"raw": "سؤال عام"},
        "urgency_score": 0.3,
        "time_pref": {"date": str(date.today()), "phrase": "اليوم"},
    }

    with patch("fsm.patient_fsm.gemini") as mock_gemini:
        mock_gemini._available = False
        patient_fsm.services.classify = MagicMock(
            return_value={
                "specialty": "general_practice",
                "specialty_ar": "الطب العام",
                "method": "default",
                "confidence": 0.5,
            },
        )
        reply, keyboard = unpack_fsm(run_async(patient_fsm.handle("اليوم")))

    assert patient_fsm.state in {State.FIND_SLOT, State.CONFIRM, State.WAITLISTED}
    assert patient_fsm.data.get("specialty_method") == "auto_fallback"
    assert "تخصص" not in reply or "الطب العام" in reply
    assert keyboard is None


def test_required_fields_match_questions():
    for field_name in REQUIRED_FIELDS:
        assert field_name in FIELD_QUESTIONS_AR


def test_validate_blocks_scheduling_until_checklist_complete():
    fsm = PatientFSM(user_id=81001)
    fsm.state = State.CHATTING
    fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}, "urgency_score": 0.4}

    with patch.object(fsm.services, "classify") as mock_classify:
        mock_classify.assert_not_called()
        reply, _ = unpack_fsm(run_async(fsm.handle("")))
    assert fsm.state == State.CHATTING
    assert FIELD_QUESTIONS_AR["time_pref"] in reply


def test_clarification_does_not_hijack_normal_booking_phrase():
    fsm = PatientFSM(user_id=81002)
    fsm.state = State.CHATTING
    fsm.data = {"name": "سارة"}

    reply, _ = unpack_fsm(run_async(fsm.handle("فهمت، عندي صداع")))

    assert fsm.state == State.CHATTING
    assert "فهمت إن" not in reply


def test_parse_specialty_arabic_full_label():
    fsm = PatientFSM(user_id=81003)
    assert fsm._parse_specialty_label("الطب العام") == "general_practice"


def test_unclear_urgency_stays_in_collect_urgency():
    fsm = PatientFSM(user_id=81005)
    fsm.state = State.CHATTING
    fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}

    reply, keyboard = unpack_fsm(run_async(fsm.handle("xyz gibberish")))

    assert fsm.state == State.CHATTING
    assert "urgency_score" not in fsm.data or fsm.data.get("urgency_score") is None
    assert keyboard is None


def test_specialty_label_keys_match_classifier():
    for key in set(SPECIALTY_LABEL_TO_KEY.values()):
        assert key in SPECIALTY_NAMES_AR, f"missing classifier entry for {key}"


def test_keyboard_specialty_labels_parse():
    fsm = PatientFSM(user_id=81006)
    allowed = {
        "gastroenterology",
        "neurology",
        "orthopedics",
        "gynecology",
        "chronic_diseases",
        "dermatology",
        "general_practice",
        "elderly",
    }
    for row in specialty_keyboard().keyboard:
        for button in row:
            key = fsm._parse_specialty_label(button.text)
            assert key is not None, f"failed to parse specialty button: {button.text!r}"
            assert key in allowed


@patch("fsm.patient_fsm.score_and_classify")
def test_slot_conflict_retries_with_next_available_slot(mock_score):
    priority = MagicMock(priority_class="P3", score=0.3, label_ar="روتيني", breakdown={})
    mock_score.return_value = priority

    fsm = PatientFSM(user_id=81004)
    fsm.state = State.CONFIRM
    fsm.data = {
        "name": "أحمد",
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.3,
        "time_pref": {"date": str(date.today() + timedelta(days=1)), "phrase": "بكرا"},
        "specialty_hint": "general_practice",
        "specialty_ar": "الطب العام",
        "priority_class": "P3",
    }
    fsm.priority = priority
    fsm.slot = {
        "slot_id": 1,
        "slot_datetime": datetime.utcnow() + timedelta(days=1),
        "doctor_name": "د.",
        "clinic_name": "عيادة",
    }

    new_slot = MagicMock(
        slot_id=2,
        slot_datetime=datetime.utcnow() + timedelta(days=2),
        specialty="general_practice",
        priority_class="P3",
        doctor=MagicMock(
            doctor_id=1,
            name="د. بديل",
            specialty="general_practice",
            clinic_code="CLINIC-GP",
            clinic_name="عيادة عامة",
        ),
    )

    @contextmanager
    def fake_get_db():
        yield MagicMock()

    with patch("database.db.get_db", fake_get_db):
        fsm.services.book = MagicMock(
            return_value={"appointment": None, "slot_conflict": True, "booking_conflict": None},
        )
        fsm.services.find_slots = MagicMock(return_value=[new_slot])
        reply, keyboard = unpack_fsm(run_async(fsm.handle("نعم")))

    fsm.services.find_slots.assert_called_once()
    assert fsm.state == State.CONFIRM
    assert fsm.slot["slot_id"] == 2
    assert "انحجز قبل التأكيد" in reply
    assert "وجدت موعد" in reply
    assert keyboard is None


def test_unsupported_specialty_offers_gp_fallback():
    fsm = PatientFSM(user_id=81007)
    fsm.state = State.CHATTING
    fsm.data = {"name": "أحمد"}
    reply, keyboard = unpack_fsm(run_async(fsm.handle("بدي موعد عند طبيب عيون")))
    assert fsm.state == State.OFFER_GP_FALLBACK
    assert "غير متوفرة" in reply
    assert keyboard is None


def test_urgency_free_text_accepted():
    fsm = PatientFSM(user_id=81008)
    fsm.state = State.CHATTING
    fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}}
    reply, keyboard = unpack_fsm(run_async(fsm.handle("عاجل")))
    assert fsm.state == State.CHATTING
    assert fsm.data["urgency_score"] >= 0.85
    assert keyboard is None


def test_unclear_time_stays_in_collect_time():
    fsm = PatientFSM(user_id=81010)
    fsm.state = State.CHATTING
    fsm.data = {"name": "أحمد", "complaint": {"raw": "صداع"}, "urgency_score": 0.4}

    reply, keyboard = unpack_fsm(run_async(fsm.handle("xyz gibberish")))

    assert fsm.state == State.CHATTING
    assert "time_pref" not in fsm.data or fsm.data.get("time_pref") is None
    assert "متى" in reply or "موعد" in reply
    assert keyboard is None


def test_greeting_accepts_name_without_second_prompt(patient_fsm):
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("سارة محمود")))
    assert patient_fsm.state == State.CHATTING
    assert "سارة" in reply
    assert FIELD_QUESTIONS_AR["name"] not in reply


def test_greeting_only_does_not_accept_as_name(patient_fsm):
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("مرحبا")))
    assert patient_fsm.state == State.CHATTING
    assert FIELD_QUESTIONS_AR["name"] in reply


def test_ahlan_is_greeting_not_name(patient_fsm):
    patient_fsm.state = State.CHATTING
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("اهلا")))
    assert patient_fsm.state == State.CHATTING
    assert "name" not in patient_fsm.data or patient_fsm.data.get("name") != "اهلا"
    assert FIELD_QUESTIONS_AR["name"] in reply


def test_ahlan_wa_sahlan_is_greeting_not_name(patient_fsm):
    patient_fsm.state = State.CHATTING
    reply, _ = unpack_fsm(run_async(patient_fsm.handle("اهلا وسهلا")))
    assert patient_fsm.state == State.CHATTING
    assert FIELD_QUESTIONS_AR["name"] in reply


def test_rule_reply_inadequate_when_question_gets_bare_prompt(patient_fsm):
    patient_fsm.state = State.CHATTING
    assert patient_fsm.rule_reply_seems_inadequate(
        "شو ساعات العيادة؟",
        FIELD_QUESTIONS_AR["name"],
    ) is True


def test_rule_reply_inadequate_when_greeting_stored_as_name(patient_fsm):
    patient_fsm.state = State.CHATTING
    patient_fsm.data["name"] = "اهلا"
    assert patient_fsm.rule_reply_seems_inadequate("اهلا", "أهلاً اهلا! 😊") is True


def test_rule_reply_adequate_for_valid_name(patient_fsm):
    patient_fsm.state = State.CHATTING
    reply = f"أهلاً سارة! 😊\n{FIELD_QUESTIONS_AR['complaint']}"
    assert patient_fsm.rule_reply_seems_inadequate("سارة", reply) is False


@patch("fsm.patient_fsm.gemini")
def test_maybe_gemini_fallback_undoes_greeting_as_name(mock_gemini, patient_fsm):
    mock_gemini.is_ready = True
    mock_gemini.build_response = AsyncMock(return_value="أهلاً! ما اسمك الكريم؟")

    patient_fsm.state = State.CHATTING
    patient_fsm.data["name"] = "اهلا"
    reply = run_async(patient_fsm.maybe_gemini_fallback("اهلا"))
    assert reply == "أهلاً! ما اسمك الكريم؟"
    assert patient_fsm.state == State.CHATTING
    assert "name" not in patient_fsm.data


def test_confirm_edit_returns_to_time_selection():
    fsm = PatientFSM(user_id=81009)
    fsm.state = State.CONFIRM
    fsm.slot = {"slot_id": 1, "slot_datetime": datetime.utcnow()}
    fsm.data = {"name": "أحمد", "time_pref": {"date": str(date.today()), "phrase": "اليوم"}}
    reply, keyboard = unpack_fsm(run_async(fsm.handle("✏️ تعديل الموعد")))
    assert fsm.state == State.CHATTING
    assert "متى" in reply
    assert keyboard is None


@patch("fsm.patient_fsm.gemini")
@patch("fsm.patient_fsm.score_and_classify")
def test_gp_fallback_locks_general_practice_after_checklist(mock_score, mock_gemini):
    mock_gemini.is_ready = False
    mock_gemini.extract_missing_field = AsyncMock(return_value=None)
    priority = MagicMock(priority_class="P3", score=0.3, label_ar="روتيني", breakdown={})
    mock_score.return_value = priority

    fsm = PatientFSM(user_id=81012)
    fsm.state = State.CHATTING
    fsm.data = {"name": "هبة"}

    with patch.object(fsm, "_find_slot", new_callable=AsyncMock) as mock_find:
        mock_find.return_value = ("وجدت موعد", None, {})
        unpack_fsm(run_async(fsm.handle("وجع في الصدر")))
        assert fsm.state == State.OFFER_GP_FALLBACK
        assert fsm.data["complaint"]["raw"] == "وجع في الصدر"

        unpack_fsm(run_async(fsm.handle("✅ تأكيد الحجز")))
        assert fsm.data["specialty_method"] == "gp_fallback"

        run_async(fsm.handle("🔴 عاجل"))
        run_async(fsm.handle("اليوم"))

        assert fsm.data["specialty_hint"] == "general_practice"
        assert fsm.data["specialty_ar"] == SPECIALTY_NAMES_AR["general_practice"]
        mock_find.assert_called()


@patch("fsm.patient_fsm.score_and_classify")
def test_confirm_informal_ah_finalizes(mock_score, patient_fsm):
    priority = MagicMock(priority_class="P3", score=0.3, label_ar="روتيني", breakdown={})
    mock_score.return_value = priority

    patient_fsm.state = State.CONFIRM
    patient_fsm.data = {
        "name": "هبة",
        "complaint": {"raw": "صداع"},
        "urgency_score": 0.3,
        "time_pref": {"date": str(date.today() + timedelta(days=1)), "phrase": "بكرا"},
        "specialty_hint": "general_practice",
        "specialty_ar": "الطب العام",
    }
    patient_fsm.slot = {
        "slot_id": 5,
        "slot_datetime": datetime.utcnow() + timedelta(days=1),
        "doctor_name": "د. اختبار",
        "clinic_name": "عيادة",
    }
    patient_fsm.priority = priority

    appt = MagicMock()
    appt.appt_id = "appt_ah_test"
    appt.appt_datetime = patient_fsm.slot["slot_datetime"]

    with patch.object(patient_fsm, "_finalize", new_callable=AsyncMock, return_value={"appointment": appt}):
        reply, _ = unpack_fsm(run_async(patient_fsm.handle("اه")))
        assert patient_fsm.state == State.FINALIZED
        assert "تم تأكيد حجزك" in reply


def test_confirm_why_question_gets_answer():
    fsm = PatientFSM(user_id=81013)
    fsm.state = State.CONFIRM
    fsm.slot = {
        "slot_id": 1,
        "slot_datetime": datetime.utcnow() + timedelta(days=1),
        "doctor_name": "د. اختبار",
        "clinic_name": "عيادة",
    }
    fsm.data = {
        "name": "هبة",
        "specialty_ar": "الطب العام",
        "complaint": {"raw": "وجع في الصدر"},
    }

    reply, keyboard = unpack_fsm(run_async(fsm.handle("ليش")))
    assert fsm.state == State.CONFIRM
    assert "موافقتك" in reply or "تأكيد" in reply
    assert "✅ لتأكيد" not in reply
    assert keyboard is None


def test_confirm_dot_gets_nudge():
    fsm = PatientFSM(user_id=81018)
    fsm.state = State.CONFIRM
    fsm.slot = {
        "slot_id": 1,
        "slot_datetime": datetime.utcnow() + timedelta(days=1),
        "doctor_name": "د. أ",
        "clinic_name": "عيادة",
    }
    fsm.slot_options = [fsm.slot]
    reply, _ = unpack_fsm(run_async(fsm.handle(".")))
    assert fsm.state == State.CONFIRM
    assert "لسا معك" in reply or "بدك" in reply
    assert "✅ لتأكيد" not in reply


def test_name_dispute_clears_wrong_name():
    fsm = PatientFSM(user_id=81014)
    fsm.state = State.CHATTING
    fsm.data = {"name": "منال", "complaint": {"raw": "صداع"}}
    reply, _ = unpack_fsm(run_async(fsm.handle("مين قال انه اسمي منال")))
    assert fsm.state == State.CHATTING
    assert "name" not in fsm.data
    assert "اسمك" in reply


@patch("fsm.patient_fsm.gemini")
def test_greeting_meta_question_does_not_assume_preloaded_name(mock_gemini):
    mock_gemini.is_ready = False
    fsm = PatientFSM(user_id=81015)
    fsm.state = State.CHATTING
    fsm.data = {"name": "منال"}
    reply, _ = unpack_fsm(run_async(fsm.handle("ماذا تريد")))
    assert fsm.state == State.CHATTING
    assert "name" not in fsm.data
    assert "مساعد حجز" in reply


def test_confirm_lists_available_slots():
    fsm = PatientFSM(user_id=81016)
    fsm.state = State.CONFIRM
    dt = datetime.utcnow() + timedelta(days=1)
    fsm.slot_options = [
        {"slot_id": 1, "slot_datetime": dt, "doctor_name": "د. أ", "clinic_name": "عيادة أ"},
        {"slot_id": 2, "slot_datetime": dt + timedelta(hours=1), "doctor_name": "د. ب", "clinic_name": "عيادة ب"},
    ]
    fsm.slot_index = 0
    fsm.slot = fsm.slot_options[0]
    fsm.data = {"specialty_ar": "العظام"}

    reply, keyboard = unpack_fsm(run_async(fsm.handle("فرجيمي المواعيد الي ضايلة")))
    assert fsm.state == State.CONFIRM
    assert "📋 المواعيد المتاحة" in reply
    assert "د. أ" in reply
    assert "د. ب" in reply
    assert keyboard is None


def test_confirm_decline_cancels():
    fsm = PatientFSM(user_id=81017)
    fsm.state = State.CONFIRM
    dt = datetime.utcnow() + timedelta(days=1)
    fsm.slot = {"slot_id": 1, "slot_datetime": dt, "doctor_name": "د. أ", "clinic_name": "عيادة"}
    fsm.slot_options = [fsm.slot]
    fsm.data = {"name": "هبة"}

    reply, _ = unpack_fsm(run_async(fsm.handle("ما بدي اشي")))
    assert fsm.state == State.CANCELLED
    assert "ألغيت" in reply


def test_cancelled_frustration_gets_explanation():
    fsm = PatientFSM(user_id=81021)
    fsm.state = State.CANCELLED
    reply, _ = unpack_fsm(run_async(fsm.handle("ليش هيك")))
    assert fsm.state == State.CANCELLED
    assert "آسف" in reply or "انتهى" in reply
    assert "حجز موعد جديد" in reply


def test_cancelled_appointment_inquiry_shows_stored_appt():
    fsm = PatientFSM(user_id=81022)
    fsm.state = State.CANCELLED
    appt = MagicMock()
    appt.appt_id = "appt_x"
    appt.appt_datetime = datetime.utcnow() + timedelta(days=1)
    appt.status = "confirmed"
    appt.specialty_ar = "الطب العام"
    appt.specialty = "general_practice"
    appt.slot = MagicMock(doctor=MagicMock(name="د. أ"))

    @contextmanager
    def fake_get_db():
        yield MagicMock()

    with patch("database.db.get_db", fake_get_db):
        with patch("database.crud.get_latest_patient_appointment", return_value=appt):
            reply, _ = unpack_fsm(run_async(fsm.handle("شو المواعيد الموجودة")))
    assert "موعدك المسجل" in reply or "appt_x" in reply


def test_cancelled_meta_question_gets_role_reply():
    fsm = PatientFSM(user_id=81018)
    fsm.state = State.CANCELLED
    reply, _ = unpack_fsm(run_async(fsm.handle("شو وظيفتك")))
    assert fsm.state == State.CANCELLED
    assert "مساعد حجز" in reply


def test_cancelled_admin_request_shows_dashboard():
    fsm = PatientFSM(user_id=81019)
    fsm.state = State.CANCELLED
    reply, _ = unpack_fsm(run_async(fsm.handle("بدي اشوف الادارة")))
    assert fsm.state == State.CANCELLED
    assert "لوحة" in reply
    assert "8000" in reply


def test_cancelled_decline_after_cancel_is_polite():
    fsm = PatientFSM(user_id=81020)
    fsm.state = State.CANCELLED
    reply, _ = unpack_fsm(run_async(fsm.handle("لا ما بدي")))
    assert fsm.state == State.CANCELLED
    assert "تمام" in reply


def test_tingling_complaint_and_medium_urgency_offline():
    """Reproduce Heba transcript: تنميل + متوسط must be captured without LLM."""
    fsm = PatientFSM(user_id=81021)
    fsm.state = State.CHATTING
    unpack_fsm(run_async(fsm.handle("منال")))
    assert fsm.data.get("name") == "منال"
    unpack_fsm(run_async(fsm.handle("عندي تنميل خفيف في ايدي")))
    assert fsm.data.get("complaint")
    assert "تنميل" in fsm.data["complaint"]["raw"]
    unpack_fsm(run_async(fsm.handle("متوسط")))
    assert fsm.data.get("urgency_score") == 0.5
    reply, _ = unpack_fsm(run_async(fsm.handle("بعد بكرا")))
    assert fsm.data.get("time_pref")
    assert fsm.state in (State.CONFIRM, State.FIND_SLOT, State.WAITLISTED, State.CHATTING)


@patch("fsm.patient_fsm.gemini")
def test_confirm_hallucinated_success_triggers_finalize(mock_gemini):
    """Weaker models say 'تم الحجز' with continue — must still finalize."""
    from nlp.booking_agent import BookingTurnResult
    from unittest.mock import AsyncMock

    mock_gemini.is_ready = True
    mock_gemini.ask = AsyncMock(return_value="")

    fsm = PatientFSM(user_id=81022)
    fsm.state = State.CONFIRM
    fsm.data = {
        "name": "منال",
        "complaint": {"raw": "تنميل"},
        "urgency_score": 0.5,
        "time_pref": {"date": None, "phrase": "بكرا"},
        "specialty_ar": "طب الأعصاب",
        "specialty_hint": "neurology",
    }
    dt = datetime.utcnow() + timedelta(days=1)
    fsm.slot = {
        "slot_id": 999001,
        "slot_datetime": dt,
        "doctor_name": "د. لينا",
        "clinic_name": "عيادة الأعصاب",
        "clinic_code": "CLINIC-NEURO",
        "specialty": "neurology",
        "priority_class": "P2",
        "doctor_id": 1,
    }
    fsm.slot_options = [fsm.slot]
    fsm.priority = type("P", (), {"priority_class": "P2", "priority_score": 0.5})()

    async def fake_turn(*_a, **_k):
        return BookingTurnResult(
            reply="✅ تم تأكيد حجزك وحفظ ملفك في النظام! رقم الحجز: fake_123",
            intent="continue",
            extracted={},
        )

    fsm._ai_turn = fake_turn  # type: ignore[method-assign]
    fsm._finalize = AsyncMock(return_value={"appointment": type("A", (), {
        "appt_id": "appt_real_1",
        "appt_datetime": dt,
    })()})

    # Neutral clarifying question — does NOT soft-confirm; LLM invents success.
    reply, _ = unpack_fsm(run_async(fsm.handle("ليش هاد الموعد؟")))
    assert "تم تأكيد حجزك" in reply
    assert "appt_real_1" in reply
    assert "fake_123" not in reply
    assert fsm.state == State.FINALIZED
    fsm._finalize.assert_awaited()


def test_confirm_munasib_finalizes_without_llm_intent():
    """Modern models: user says مناسب — rules must book even if LLM would continue."""
    from unittest.mock import AsyncMock

    fsm = PatientFSM(user_id=81023)
    fsm.state = State.CONFIRM
    fsm.data = {
        "name": "منال",
        "complaint": {"raw": "تنميل"},
        "urgency_score": 0.5,
        "time_pref": {"phrase": "بكرا"},
        "specialty_ar": "طب الأعصاب",
        "specialty_hint": "neurology",
    }
    dt = datetime.utcnow() + timedelta(days=1)
    fsm.slot = {
        "slot_id": 42,
        "slot_datetime": dt,
        "doctor_name": "د. لينا",
        "clinic_name": "عيادة الأعصاب",
        "clinic_code": "CLINIC-NEURO",
        "specialty": "neurology",
        "priority_class": "P2",
        "doctor_id": 1,
    }
    fsm.slot_options = [fsm.slot]
    fsm.priority = type("P", (), {"priority_class": "P2", "priority_score": 0.5})()
    fsm._finalize = AsyncMock(return_value={"appointment": type("A", (), {
        "appt_id": "appt_munasib",
        "appt_datetime": dt,
    })()})

    reply, _ = unpack_fsm(run_async(fsm.handle("مناسب")))
    assert fsm.state == State.FINALIZED
    assert "appt_munasib" in reply
    fsm._finalize.assert_awaited()
