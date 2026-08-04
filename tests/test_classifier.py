from unittest.mock import AsyncMock

import pytest

from scheduler.classifier import (
    SPECIALTY_KEYS,
    classify_specialty,
    classify_with_gemini_fallback,
)


SPECIALTY_RULE_CASES = [
    ("عندي صداع شديد جداً مع تنميل بإيدي", "neurology"),
    ("صار عندي كسر بذراعي بعد ما وقعت", "orthopedics"),
    ("الم في الساق", "orthopedics"),
    ("وجع في الساق", "orthopedics"),
    ("أنا بالشهر السابع من الحمل وصار عندي نزيف رحمي", "gynecology"),
    ("طلع عندي طفح جلدي على إيدي", "dermatology"),
    ("عندي مغص وإسهال شديد من الصبح", "gastroenterology"),
    ("السكر عندي طلع فوق 400 اليوم", "chronic_diseases"),
    ("جدي عمره 78 سنة وبده فحص دوري", "elderly"),
    ("عندي زكام بسيط من أيام", "general_practice"),
]

_VAGUE = "بدي أسأل سؤال عادي مش طبي"


@pytest.mark.parametrize("text,expected", SPECIALTY_RULE_CASES)
def test_rule_based_specialty(text, expected):
    result = classify_specialty(text)
    assert result["specialty"] == expected
    assert result["method"] == "rule"
    assert result["specialty"] in SPECIALTY_KEYS


def test_no_match_falls_to_default():
    result = classify_specialty("بدي أسأل سؤال عادي مش طبي")
    assert result["specialty"] == "general_practice"
    assert result["method"] == "default"
    assert result["confidence"] == 0.5


@pytest.mark.parametrize("text", ["", None])
def test_empty_and_none_text_do_not_crash(text):
    assert classify_specialty(text)["method"] == "default"


def test_orthopedics_wataar_word_boundary():
    assert classify_specialty("عندي شد بمنطقة وتر الركبة")["specialty"] == "orthopedics"


@pytest.mark.asyncio
async def test_gemini_used_only_when_rules_default():
    mock_client = AsyncMock()
    mock_client.ask.return_value = "chronic_diseases"
    result = await classify_with_gemini_fallback(_VAGUE, mock_client)
    assert result["method"] == "gemini"
    assert result["specialty"] == "chronic_diseases"
    mock_client.ask.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_not_called_when_rule_already_matched():
    mock_client = AsyncMock()
    result = await classify_with_gemini_fallback("صار عندي كسر بذراعي", mock_client)
    assert result["method"] == "rule"
    mock_client.ask.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_returns_unknown_key_falls_back_to_auto():
    mock_client = AsyncMock()
    mock_client.ask.return_value = "radiology"
    result = await classify_with_gemini_fallback(_VAGUE, mock_client)
    assert result["method"] == "auto_fallback"
    assert result["specialty"] == "general_practice"


@pytest.mark.asyncio
async def test_gemini_parses_trailing_punctuation():
    mock_client = AsyncMock()
    mock_client.ask.return_value = "chronic_diseases."
    result = await classify_with_gemini_fallback(_VAGUE, mock_client)
    assert result["method"] == "gemini"
    assert result["specialty"] == "chronic_diseases"


@pytest.mark.asyncio
async def test_gemini_parses_prefixed_arabic_text():
    mock_client = AsyncMock()
    mock_client.ask.return_value = "الجواب: chronic_diseases"
    result = await classify_with_gemini_fallback(_VAGUE, mock_client)
    assert result["method"] == "gemini"
    assert result["specialty"] == "chronic_diseases"


@pytest.mark.asyncio
async def test_gemini_empty_response_handled_gracefully():
    mock_client = AsyncMock()
    mock_client.ask.return_value = ""
    result = await classify_with_gemini_fallback(_VAGUE, mock_client)
    assert result["method"] == "auto_fallback"


@pytest.mark.asyncio
async def test_gemini_raises_exception_handled_gracefully():
    mock_client = AsyncMock()
    mock_client.ask.side_effect = TimeoutError("network down")
    result = await classify_with_gemini_fallback(_VAGUE, mock_client)
    assert result["method"] == "auto_fallback"
