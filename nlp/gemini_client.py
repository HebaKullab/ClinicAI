"""
nlp/gemini_client.py — LLM client: OpenRouter (primary) + Google Gemma (fallback).

All booking AI calls go through this module. The exported singleton ``gemini``
keeps backward compatibility with existing imports.
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

from config import (
    CLINIC_NAME,
    GEMINI_API_KEY,
    LLM_FALLBACK_MODEL,
    LLM_PRIMARY_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)

try:
    import google.genai as genai
    from google.genai import types as google_types
except ImportError:  # pragma: no cover
    genai = None
    google_types = None

GEMINI_AVAILABLE = bool(GEMINI_API_KEY and genai is not None)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_AVAILABLE else None
OPENROUTER_AVAILABLE = bool(OPENROUTER_API_KEY and str(OPENROUTER_API_KEY).strip())

OFF_TOPIC_REPLY = (
    "أنا مساعد حجز المواعيد في العيادة 🏥 — ما بقدر أساعدك بمواضيع برّا الحجز، "
    "بس إذا بدك موعد أو استعلام عن حجزك خبرني."
)

SYSTEM_CONTEXT = f"""
أنت موظف/ة استقبال في {CLINIC_NAME}.
مهمتك مساعدة المريض في حجز الموعد خطوة بخطوة.
تحدث باللهجة الفلسطينية العامية — ودود، مباشر، وقريب من الكلام البشري.
لا تقدم تشخيصاً طبياً ولا نصائح علاجية.
لا تطلب رقم الهاتف أو البريد — النظام يجمع فقط: الاسم، الشكوى، الأولوية، وقت الموعد.
لا تتصرف كـ ChatGPT عام — ابقَ ضمن الحجز والاستعلامات الإدارية عن العيادة.
الردود قصيرة (جملة إلى ثلاث جمل) إلا إذا طُلب توضيح أكثر.
""".strip()

_OFF_TOPIC_MARKERS = (
    "chatgpt", "gpt", "ذكاء اصطناعي", "برمجة", "سياسة", "رياضة", "فلسطين",
    "اسرائيل", "Trump", "ترامب", "bitcoin", "crypto", "وصفة", "طبخ",
)


class GeminiClient:
    """Primary: OpenRouter (gpt-4.1-mini). Fallback: Google Gemma."""

    def __init__(self):
        self._primary_model = LLM_PRIMARY_MODEL if OPENROUTER_AVAILABLE else None
        self._fallback_model = (
            LLM_FALLBACK_MODEL if GEMINI_AVAILABLE and LLM_FALLBACK_MODEL else None
        )
        if self._primary_model == self._fallback_model:
            self._fallback_model = None

        self._available = bool(self._primary_model or self._fallback_model)
        self._model = self._primary_model or self._fallback_model
        self._using_fallback = not bool(self._primary_model)
        self._primary_cooldown_until = 0.0

        if not OPENROUTER_AVAILABLE and self._primary_model:
            logger.info("OpenRouter disabled: OPENROUTER_API_KEY is not set in .env")
        if not GEMINI_AVAILABLE and LLM_FALLBACK_MODEL:
            if not GEMINI_API_KEY:
                logger.info("Gemma fallback disabled: GEMINI_API_KEY is not set in .env")
            elif genai is None:
                logger.warning("Gemma fallback disabled: google-genai package is not installed")

    @property
    def is_ready(self) -> bool:
        if not self._available:
            return False
        if self._using_fallback:
            return bool(self._fallback_model)
        return time.monotonic() >= self._primary_cooldown_until

    @property
    def active_model(self) -> str | None:
        return self._model

    def _maybe_restore_primary_model(self) -> None:
        if not self._using_fallback or not self._primary_model:
            return
        if self._primary_cooldown_until <= 0:
            return
        if time.monotonic() >= self._primary_cooldown_until:
            self._using_fallback = False
            self._model = self._primary_model
            self._primary_cooldown_until = 0.0
            logger.info("Restored primary LLM model: %s", self._primary_model)

    def _activate_fallback(self, reason: str) -> bool:
        if not self._fallback_model:
            return False
        if not self._using_fallback:
            self._using_fallback = True
            self._model = self._fallback_model
            logger.warning(
                "Primary model unavailable — switched to fallback %s (%s)",
                self._fallback_model,
                reason[:120],
            )
        return True

    def _enter_primary_cooldown(self, seconds: float, reason: str) -> None:
        self._primary_cooldown_until = time.monotonic() + seconds
        logger.warning("Primary LLM model paused for %.0fs (%s)", seconds, reason)

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        err = str(exc).lower()
        return (
            "429" in err
            or "resource_exhausted" in err
            or "quota" in err
            or "rate limit" in err
            or "insufficient" in err
        )

    @staticmethod
    def _cooldown_seconds(exc: Exception) -> float | None:
        if GeminiClient._is_quota_error(exc):
            return 60.0
        return None

    def _generate_openrouter_sync(self, model: str, prompt: str, max_tokens: int) -> str:
        url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_CONTEXT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=60.0) as http:
            response = http.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                raise Exception(f"429 rate limit: {response.text[:200]}")
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()

    def _generate_gemini_sync(self, model: str, prompt: str, max_tokens: int) -> str:
        if gemini_client is None or google_types is None:
            raise RuntimeError("Google GenAI client is not configured")
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=google_types.GenerateContentConfig(
                system_instruction=SYSTEM_CONTEXT,
                max_output_tokens=max_tokens,
            ),
        )
        return (response.text or "").strip()

    def _generate_sync(self, model: str, prompt: str, max_tokens: int) -> str:
        if model == self._fallback_model:
            return self._generate_gemini_sync(model, prompt, max_tokens)
        return self._generate_openrouter_sync(model, prompt, max_tokens)

    async def ask(self, prompt: str, max_tokens: int = 300) -> str:
        if not self.is_ready or not self._model:
            return ""

        self._maybe_restore_primary_model()
        model = self._model
        is_primary = model == self._primary_model

        try:
            return await asyncio.to_thread(self._generate_sync, model, prompt, max_tokens)
        except Exception as exc:
            if (
                is_primary
                and self._is_quota_error(exc)
                and self._activate_fallback(str(exc)[:120])
            ):
                cooldown = self._cooldown_seconds(exc)
                if cooldown:
                    self._enter_primary_cooldown(cooldown, "API quota/rate limit")
                try:
                    return await asyncio.to_thread(
                        self._generate_sync,
                        self._fallback_model,
                        prompt,
                        max_tokens,
                    )
                except Exception as fallback_exc:
                    logger.warning("Fallback model ask failed: %s", fallback_exc)
                    return ""

            cooldown = self._cooldown_seconds(exc)
            if cooldown and is_primary:
                self._enter_primary_cooldown(cooldown, "API quota/rate limit")
            else:
                logger.warning("LLM ask failed (%s): %s", model, exc)
            return ""

    @staticmethod
    def looks_like_question(text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        if lowered.endswith("?") or lowered.endswith("؟"):
            return True
        tokens = set(lowered.replace("؟", " ").replace("?", " ").split())
        single_word_markers = {
            "ليش", "لماذا", "ليه", "ماذا", "كيف", "متى", "وين", "أين", "اين",
            "هل", "شو", "ايش", "إيش", "مين", "كم",
            "what", "why", "how", "when", "where",
        }
        if tokens & single_word_markers:
            return True
        phrase_markers = ("ممكن",)
        return any(marker in lowered for marker in phrase_markers)

    @staticmethod
    def looks_off_topic(text: str) -> bool:
        lowered = (text or "").strip().lower()
        return any(marker in lowered for marker in _OFF_TOPIC_MARKERS)

    async def answer_in_booking_context(
        self,
        user_message: str,
        fsm_state: str,
        data: dict,
        current_question: str,
    ) -> str | None:
        if not self.is_ready:
            return None

        if self.looks_off_topic(user_message):
            return f"{OFF_TOPIC_REPLY}\n\n{current_question}"

        prompt = (
            f"رسالة المريض: {user_message}\n"
            f"مرحلة الحجز: {fsm_state}\n"
            f"سياق: {current_question}\n\n"
            "اكتب رداً قصيراً بالفلسطيني (جملة أو جملتين فقط). "
            "جاوب على سؤاله بشكل طبيعي وودود. "
            "لا تذكر قائمة أزرار ولا تكرر «اكتب ✅» أو «اكتب ❌». "
            "لا تكرر بياناته ولا تذكر تفاصيل تقنية."
        )
        reply = await self.ask(prompt, max_tokens=100)
        return reply or None

    async def build_response(self, fsm_state: str, data: dict) -> str:
        if not self.is_ready:
            return ""

        last_message = data.get("last_user_message", "")
        current_question = data.get("current_question", "")
        prompt = (
            f"حالة المحادثة: {fsm_state}\n"
            f"آخر رسالة من المريض: {last_message}\n"
            f"المعلومة المطلوبة الآن: {current_question}\n"
            "اكتب رداً قصيراً ودوداً بالفلسطيني (جملة أو جملتين) يساعد المريض يكمّل الحجز. "
            "لا تطلب رقم هاتف. لا تخترع معلومات طبية ولا تكرر بياناته."
        )
        return await self.ask(prompt, max_tokens=90)

    async def extract_missing_field(self, text: str, missing_field: str) -> str:
        if not self.is_ready:
            return ""

        field_prompts = {
            "name": "ما اسم المريض في هذه الجملة؟ أجب بالاسم فقط بدون شرح. إذا لا يوجد اسم أجب: NONE",
            "complaint": "ما الشكوى أو سبب الزيارة؟ أجب بجملة قصيرة. إذا لا توجد شكوى أجب: NONE",
            "urgency": "هل الحالة عاجلة أم متوسطة أم روتينية؟ أجب بكلمة واحدة فقط. إذا غير واضح أجب: NONE",
            "time_pref": "متى يريد الموعد؟ (اليوم/بكرا/الأسبوع الجاي/أي وقت). إذا غير واضح أجب: NONE",
            "specialty": (
                "ما التخصص الطبي الأقرب؟ اختر واحداً فقط من: "
                "gastroenterology, neurology, orthopedics, gynecology, chronic_diseases, "
                "dermatology, general_practice, elderly. إذا غير واضح أجب: NONE"
            ),
        }
        instruction = field_prompts.get(missing_field, "استخرج المعلومة المطلوبة أو NONE.")
        prompt = f"الرسالة: '{text}'\n{instruction}"
        raw = await self.ask(prompt, max_tokens=60)
        cleaned = raw.strip().strip(".")
        if cleaned.upper() == "NONE" or not cleaned:
            return ""
        return cleaned

    async def generate_voice_response(self, text: str) -> str:
        if not self.is_ready:
            return text

        prompt = (
            "حوّل هذا النص إلى جملة عربية طبيعية تصلح للتحويل إلى صوت "
            f"(بدون رموز أو نقاط أو أرقام):\n{text}"
        )
        spoken = await self.ask(prompt, max_tokens=150)
        return spoken or text


gemini = GeminiClient()
