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

PRIMARY_MAX_RETRIES = 3
PRIMARY_RETRY_BACKOFF_S = (0.5, 1.0, 2.0)
TRANSIENT_PRIMARY_COOLDOWN_S = 30.0
PRIMARY_BILLING_COOLDOWN_S = 300.0  # OpenRouter 402 — credits won't recover quickly

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
    def _is_payment_required_error(exc: Exception) -> bool:
        """OpenRouter 402 — insufficient credits / payment required."""
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            return exc.response.status_code == 402
        err = str(exc).lower()
        return "402" in err or "payment required" in err or "payment_required" in err

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            if exc.response.status_code == 429:
                return True
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
        if GeminiClient._is_payment_required_error(exc):
            return PRIMARY_BILLING_COOLDOWN_S
        if GeminiClient._is_quota_error(exc):
            return 60.0
        if GeminiClient._is_transient_error(exc):
            return TRANSIENT_PRIMARY_COOLDOWN_S
        return None

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Network / connection errors worth retrying or falling back."""
        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.NetworkError,
                ConnectionError,
                TimeoutError,
            ),
        ):
            return True
        err = str(exc).lower()
        markers = (
            "10054",
            "10053",
            "10060",
            "winerror",
            "connection reset",
            "connection aborted",
            "forcibly closed",
            "broken pipe",
            "timed out",
            "timeout",
            "eof occurred",
            "connection refused",
            "server disconnected",
            "remote end closed",
            "unexpected eof",
        )
        return any(marker in err for marker in markers)

    @staticmethod
    def _should_fallback_to_gemini(exc: Exception) -> bool:
        return (
            GeminiClient._is_payment_required_error(exc)
            or GeminiClient._is_quota_error(exc)
            or GeminiClient._is_transient_error(exc)
        )

    async def _call_primary_with_retries(self, prompt: str, max_tokens: int) -> str:
        last_exc: Exception | None = None
        for attempt in range(PRIMARY_MAX_RETRIES):
            try:
                return await asyncio.to_thread(
                    self._generate_sync,
                    self._primary_model,
                    prompt,
                    max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                if not self._is_transient_error(exc) or attempt >= PRIMARY_MAX_RETRIES - 1:
                    raise
                wait_s = PRIMARY_RETRY_BACKOFF_S[min(attempt, len(PRIMARY_RETRY_BACKOFF_S) - 1)]
                logger.warning(
                    "Primary LLM transient error (attempt %s/%s): %s — retry in %.1fs",
                    attempt + 1,
                    PRIMARY_MAX_RETRIES,
                    exc,
                    wait_s,
                )
                await asyncio.sleep(wait_s)
        if last_exc is not None:
            raise last_exc
        return ""

    async def _invoke_fallback(self, prompt: str, max_tokens: int) -> str:
        if not self._fallback_model:
            return ""
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
        with httpx.Client(timeout=15.0, trust_env=False, proxy=None) as http:
            response = http.post(url, headers=headers, json=payload)
            if response.status_code in (402, 429):
                body = (response.text or "")[:300]
                raise Exception(f"{response.status_code}: {body}")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = (exc.response.text or "")[:300] if exc.response is not None else ""
                raise Exception(f"{exc.response.status_code if exc.response else 'http'}: {body}") from exc
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()

    def _generate_gemini_sync(self, model: str, prompt: str, max_tokens: int) -> str:
        if gemini_client is None or google_types is None:
            raise RuntimeError("Google GenAI client is not configured")

        # Gemini 2.5 Flash counts "thinking" tokens against max_output_tokens.
        # Default dynamic thinking often eats the whole budget → truncated JSON mid-string.
        # Disable thinking for booking turns (faster + complete JSON). Soft-fail if unsupported.
        config_kwargs: dict = {
            "system_instruction": SYSTEM_CONTEXT,
            # Leave headroom even after thinking is off (short replies still need full JSON).
            "max_output_tokens": max(max_tokens, 1024),
            "response_mime_type": "application/json",
        }
        try:
            thinking_cfg = google_types.ThinkingConfig(thinking_budget=0)
            config_kwargs["thinking_config"] = thinking_cfg
        except Exception:
            # Older SDK / models that reject ThinkingConfig — continue without it.
            pass

        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=google_types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            # Some models reject response_mime_type or thinking_config — retry plain config.
            err = str(exc).lower()
            if "thinking" in err or "mime" in err or "json" in err or "invalid" in err:
                logger.warning("Gemini config rejected (%s) — retrying without extras", exc)
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=google_types.GenerateContentConfig(
                        system_instruction=SYSTEM_CONTEXT,
                        max_output_tokens=max(max_tokens, 1024),
                    ),
                )
            else:
                raise
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
            if is_primary:
                return await self._call_primary_with_retries(prompt, max_tokens)
            return await asyncio.to_thread(self._generate_sync, model, prompt, max_tokens)
        except Exception as exc:
            if (
                is_primary
                and self._should_fallback_to_gemini(exc)
                and self._activate_fallback(str(exc)[:120])
            ):
                cooldown = self._cooldown_seconds(exc)
                if cooldown:
                    if self._is_payment_required_error(exc):
                        reason = "OpenRouter payment required (insufficient credits)"
                    elif self._is_quota_error(exc):
                        reason = "API quota/rate limit"
                    else:
                        reason = "transient network error"
                    self._enter_primary_cooldown(cooldown, reason)
                return await self._invoke_fallback(prompt, max_tokens)

            cooldown = self._cooldown_seconds(exc)
            if cooldown and is_primary:
                if self._is_payment_required_error(exc):
                    reason = "OpenRouter payment required (insufficient credits)"
                elif self._is_quota_error(exc):
                    reason = "API quota/rate limit"
                else:
                    reason = "transient network error"
                self._enter_primary_cooldown(cooldown, reason)
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
            "إذا سأل عن وظيفتك: أنت مساعد حجز مواعيد العيادة فقط. "
            "إذا سأل عن الإدارة: لوحة العيادة للموظفين وليست من البوت. "
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

    async def booking_turn(
        self,
        user_message: str,
        phase: str,
        collected: dict,
        chat_history: list[dict] | None = None,
        slot_context: dict | None = None,
        operation_context: dict | None = None,
    ) -> str:
        """Single unified LLM turn: natural reply + intent + field extraction."""
        if not self.is_ready:
            return ""

        import json as _json

        history_lines = []
        for item in (chat_history or [])[-12:]:
            role = item.get("role", "user")
            content = (item.get("content") or "").strip()
            if content:
                prefix = "المريض" if role == "user" else "المساعد"
                history_lines.append(f"{prefix}: {content}")

        collected_summary = []
        if collected.get("name"):
            collected_summary.append(f"الاسم: {collected['name']}")
        complaint = collected.get("complaint")
        if isinstance(complaint, dict) and complaint.get("raw"):
            collected_summary.append(f"الشكوى: {complaint['raw']}")
        elif isinstance(complaint, str) and complaint:
            collected_summary.append(f"الشكوى: {complaint}")
        if collected.get("urgency_score") is not None:
            collected_summary.append(f"الأولوية: {collected['urgency_score']}")
        tp = collected.get("time_pref")
        if isinstance(tp, dict) and (tp.get("phrase") or tp.get("date")):
            collected_summary.append(f"وقت الموعد: {tp.get('phrase') or tp.get('date')}")

        op_ctx = operation_context or {}
        op_lines = []
        if op_ctx.get("missing_fields"):
            op_lines.append(f"حقول ناقصة: {', '.join(op_ctx['missing_fields'])}")
        if op_ctx.get("stored_appointment"):
            op_lines.append(f"موعد مسجل: {op_ctx['stored_appointment']}")
        if op_ctx.get("unsupported_specialty"):
            op_lines.append(f"تخصص غير متوفر: {op_ctx['unsupported_specialty']}")
        if op_ctx.get("proposed_slot"):
            op_lines.append(f"موعد مقترح: {op_ctx['proposed_slot']}")
        if op_ctx.get("slot_options_count", 0) > 1:
            op_lines.append(f"بدائل متاحة: {op_ctx['slot_options_count']} مواعيد")
        if op_ctx.get("terminal_state"):
            op_lines.append(f"حالة الطلب: {op_ctx['terminal_state']}")
        if op_ctx.get("rule_hint"):
            op_lines.append(f"تلميح النظام: {op_ctx['rule_hint']}")

        slot_line = ""
        if slot_context and not op_ctx.get("proposed_slot"):
            when = slot_context.get("when") or "—"
            slot_line = f"\nموعد مقترح للتأكيد: {when}"

        phase_hints = {
            "CHATTING": "اجمع الاسم والشكوى والأولوية ووقت الموعد المفضل.",
            "CONFIRM": (
                "المريض بمرحلة تأكيد موعد مقترح. حوّل كلامه الطبيعي إلى intent واحد فقط:\n"
                "- confirm: نعم، تمام، موافق، احجز، يلا، مناسب، حاضر، طيب، اوكي\n"
                "- decline: لا، ما بدي، إلغاء\n"
                "- next_slot: موعد آخر، وقت تاني، بدي أغيره، مش هاد الموعد\n"
                "- edit_time: تعديل، غير الوقت، بدي موعد بكرا/اليوم\n"
                "- slot_list: شو المواعيد، فرجيني الخيارات\n"
                "- cancel: إلغاء الحجز\n"
                "لا تترك intent=continue إذا النية واضحة.\n"
                "ممنوع كتابة تم الحجز/رقم الحجز — النظام يحجز بعد confirm فقط."
            ),
            "GP_FALLBACK": "التخصص المطلوب غير متوفر — اعرض الطب العام وافهم موافقة أو رفض.",
            "TERMINAL": "انتهى الحجز السابق — ساعد بالاستعلام أو حجز جديد أو توضيح الحالة.",
        }

        prompt = (
            f"مرحلة المحادثة: {phase}\n"
            f"{phase_hints.get(phase, '')}\n"
            f"البيانات المجمّعة: {', '.join(collected_summary) or 'لا شيء بعد'}\n"
            + (f"سياق النظام (حقائق فقط — لا تخترع):\n" + "\n".join(op_lines) + "\n" if op_lines else "")
            + f"{slot_line}\n"
            + ("سجل المحادثة:\n" + "\n".join(history_lines) + "\n" if history_lines else "")
            + f"رسالة المريض: {user_message}\n\n"
            "مهمتك: ردّ طبيعي بالفلسطيني (جملة إلى ثلاث) + تحديد intent + استخراج حقول جديدة.\n"
            "الحجز والأولوية والتصنيف والإلغاء يقررها النظام — أنت تفهم النية والرد فقط.\n"
            "لا تشخيص طبي. لا أزرار. لا تكرر أسئلة عن معلومات موجودة.\n"
            "ممنوع تماماً أن تقول «تم الحجز» أو «تم تأكيد حجزك» أو تعطي رقم حجز — "
            "التأكيد يتم فقط عبر intent=confirm والنظام ينفّذ الحفظ.\n"
            "إذا المرحلة CONFIRM والمريض موافق: intent=confirm ورد قصير مثل «لحظة بتأكد الحجز».\n\n"
            "intents: continue, confirm, decline, cancel, accept_gp, reject_gp, inquiry, contact, "
            "new_booking, next_slot, slot_list, edit_time, off_topic\n\n"
            "مهم: intent يحدد العملية التي ينفذها النظام (ليس مجرد رد).\n"
            "أمثلة CONFIRM: «وقت تاني»→next_slot، «بدي أغيره»→next_slot، «تعديل»→edit_time، «نعم»→confirm.\n\n"
            "أرجع JSON فقط:\n"
            + _json.dumps(
                {
                    "reply": "...",
                    "intent": "continue",
                    "off_topic": False,
                    "extracted": {
                        "name": None,
                        "complaint": None,
                        "urgency": None,
                        "time_pref": None,
                    },
                },
                ensure_ascii=False,
            )
        )
        return await self.ask(prompt, max_tokens=1024)

    async def classify_and_reply(self, complaint_text: str, patient_name: str = "") -> dict:
        """Single LLM call to classify specialty and generate a warm Palestinian Arabic reply simultaneously."""
        if not self.is_ready:
            return {}

        prompt = (
            f"المريض ({patient_name or 'عزيزي المريض'}) يقول: '{complaint_text}'\n\n"
            "المطلوب القيام بالمهمتين التاليتين وتقديم النتيجة كـ JSON فقط:\n"
            "1. اختر التخصص الأنسب فقط من المفاتيح التالية:\n"
            "   gastroenterology, neurology, orthopedics, gynecology, chronic_diseases, dermatology, general_practice, elderly\n"
            "2. اكتب رداً فلسطينياً عامياً ودوداً وقصيراً جداً (جملة واحدة) يؤكد فهم الشكوى وتوجيهه للعيادة.\n\n"
            "صيغة الإجابة المطلوب إرجاعها (JSON فقط بدون أي أسطر إضافية أو ملاحق):\n"
            '{"specialty": "<المفتاح>", "reply": "<الرد العربي العامي>"}'
        )

        raw = await self.ask(prompt, max_tokens=150)
        if not raw:
            return {}

        import json
        try:
            # Clean JSON markers if returned
            clean_raw = raw.strip()
            if clean_raw.startswith("```json"):
                clean_raw = clean_raw[7:]
            if clean_raw.startswith("```"):
                clean_raw = clean_raw[3:]
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3]
            return json.loads(clean_raw.strip())
        except Exception as exc:
            logger.warning("Failed to parse JSON from classify_and_reply: %s (Raw: %s)", exc, raw)
            return {}

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
