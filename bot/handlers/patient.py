"""Patient-facing Telegram handler with text/voice input and active TTS replies."""
from __future__ import annotations

import io
import logging
import asyncio

from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ContextTypes

from bot.keyboards import main_menu_keyboard
from config import TTS_ENABLED, TTS_RESPONSE_MODE
from database import crud
from database.db import get_db
from fsm.fsm_result import keyboard_for_action
from fsm.patient_fsm import PatientFSM, State
from nlp.normalizer import normalize
from voice.stt import transcribe_voice
from voice.tts import text_to_ogg

from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _is_repeat_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return any(
        token in lowered
        for token in [
            "repeat",
            "كرر",
            "تكرار",
            "من جديد",
            "من اول",
            "restart",
            "ابدأ من جديد",
            "بدء جديد",
            "حجز موعد جديد",
            "📅 حجز موعد جديد",
        ]
    )


def _is_inquiry_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "استعلام" in lowered or "موعدي" in lowered or "🔍" in lowered


_EXPLICIT_CANCEL_PHRASES = (
    "❌ إلغاء موعد",
    "إلغاء موعد",
    "الغاء موعد",
    "الغي موعد",
    "إلغي موعد",
    "الغاء الموعد",
    "إلغاء الموعد",
    "اريد الغاء الموعد",
    "أريد إلغاء الموعد",
    "اريد الغاء موعد",
    "أريد إلغاء موعد",
    "بدي الغي",
    "بدي الغاء",
    "الغاء الحجز",
    "إلغاء الحجز",
    "الغي حجزي",
    "cancel appointment",
)


def _menu_text(text: str) -> str:
    """Normalize Arabic menu/cancel phrases for robust substring matching."""
    return normalize((text or "").strip()).lower()


def _is_natural_language_cancel_request(text: str) -> bool:
    """Free-text cancel intent, e.g. «أريد إلغاء الموعد»."""
    norm = _menu_text(text)
    cancel_tokens = ("الغاء", "الغي", "cancel")
    if not any(token in norm for token in cancel_tokens):
        return False
    appt_tokens = ("موعد", "حجز", "appointment")
    intent_tokens = ("بدي", "اريد", "حاب", "بدها")
    if any(token in norm for token in appt_tokens):
        return True
    return any(token in norm for token in intent_tokens)


def _is_menu_cancel_request(text: str) -> bool:
    """Cancel an existing DB appointment — not bare ❌/❌ إلغاء during confirm keyboard."""
    raw = (text or "").strip()
    if raw in {"❌", "لا", "❌ إلغاء"}:
        return False
    norm = _menu_text(text)
    if any(_menu_text(phrase) in norm for phrase in _EXPLICIT_CANCEL_PHRASES):
        return True
    return _is_natural_language_cancel_request(text)


def _is_contact_request(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return "تواصل" in lowered or "📞" in lowered


def _get_fsm(user_id: int, reset: bool = False) -> PatientFSM:
    if reset:
        with get_db() as db:
            crud.delete_fsm_session(db, user_id, role="patient")
        fsm = PatientFSM(user_id=user_id)
        _persist_fsm(fsm)
        return fsm

    with get_db() as db:
        row = crud.get_fsm_session(db, user_id, role="patient")
        if row:
            return PatientFSM.from_snapshot(user_id, row)
    fsm = PatientFSM(user_id=user_id)
    _persist_fsm(fsm)
    return fsm


def _persist_fsm(fsm: PatientFSM) -> None:
    with get_db() as db:
        snapshot = fsm.to_snapshot()
        crud.upsert_fsm_session(db, fsm.user_id, role="patient", **snapshot)


def _format_appointment(appt) -> str:
    if not appt:
        return "لا يوجد موعد مسجل باسمك حالياً. يمكنك اختيار 📅 حجز موعد جديد."
    date_text = (
        appt.appt_datetime.strftime("%A، %d/%m/%Y — %H:%M")
        if appt.appt_datetime
        else "قائمة الانتظار"
    )
    status_ar = {
        "confirmed": "مؤكد",
        "waitlisted": "قائمة انتظار",
        "completed": "مكتمل",
        "no_show": "غياب",
        "cancelled": "ملغي",
    }.get(appt.status, appt.status)
    patient_name = appt.patient.name if appt.patient else "—"
    specialty = appt.specialty_ar or appt.specialty or "—"
    doctor = appt.slot.doctor if appt.slot and appt.slot.doctor else None
    return (
        "📌 آخر موعد مسجل:\n"
        f"رقم الحجز: {appt.appt_id}\n"
        f"المريض: {patient_name}\n"
        f"الوقت: {date_text}\n"
        f"التخصص: {specialty}\n"
        f"الطبيب: {doctor.name if doctor else '—'}\n"
        f"العيادة: {doctor.clinic_name if doctor else '—'}\n"
        f"الحالة: {status_ar}"
    )


def _should_send_voice(incoming_was_voice: bool) -> bool:
    if not TTS_ENABLED or TTS_RESPONSE_MODE == "text":
        return False
    if TTS_RESPONSE_MODE in {"voice", "both"}:
        return True
    return TTS_RESPONSE_MODE == "auto" and incoming_was_voice


def _begin_booking(user_id: int) -> None:
    fsm = _get_fsm(user_id, reset=True)
    fsm.state = State.COLLECT_NAME
    _persist_fsm(fsm)


async def _send_patient_reply(
    message,
    reply: str,
    keyboard=None,
    incoming_was_voice: bool = False,
) -> bool:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            await message.reply_text(reply, reply_markup=keyboard)
            last_exc = None
            break
        except (TimedOut, NetworkError) as exc:
            last_exc = exc
            logger.warning("Telegram send timeout (attempt %s/3): %s", attempt + 1, exc)
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    if not _should_send_voice(incoming_was_voice):
        return False

    try:
        audio_bytes = await text_to_ogg(reply)
        voice_file = io.BytesIO(audio_bytes)
        voice_file.name = "clinicai_reply.ogg"
        voice_file.seek(0)
        await message.reply_voice(voice=voice_file)
        return True
    except Exception as exc:
        logger.warning("TTS voice reply failed; text reply was sent: %s", exc)
        return False


def _log_outbound(user_id: int, reply: str, voice_sent: bool = False) -> None:
    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply)
        if voice_sent:
            crud.log_message(db, user_id, "outbound", "bot_voice", reply)


def _menu_response(user_id: int, text: str):
    """Return (reply, keyboard) for main-menu commands, or None."""
    if _is_repeat_request(text):
        _begin_booking(user_id)
        return "📅 تمام، خلينا نبدأ حجز جديد. ما اسمك الكريم؟", None

    if _is_inquiry_request(text):
        with get_db() as db:
            appt = crud.get_latest_patient_appointment(db, user_id)
            reply = _format_appointment(appt)
        return reply, main_menu_keyboard()

    if _is_menu_cancel_request(text):
        with get_db() as db:
            appt = crud.cancel_latest_patient_appointment(db, user_id)
        _get_fsm(user_id, reset=True)
        reply = (
            "تم إلغاء آخر موعد مؤكد/منتظر وإرجاع الـ slot كمتاح إذا كان محجوزاً. ✅"
            if appt
            else "لا يوجد موعد مؤكد أو منتظر لإلغائه حالياً."
        )
        return reply, main_menu_keyboard()

    if _is_contact_request(text):
        return (
            "📞 تواصلك وصل. يمكنك كتابة رسالتك هنا، وسيتم حفظها في سجل المحادثات للعيادة.",
            main_menu_keyboard(),
        )

    return None


async def _handle_fsm_turn(user_id: int, text: str) -> tuple[str, object | None]:
    fsm = _get_fsm(user_id)
    reply, action, _payload = await fsm.handle(text)

    if fsm.state not in (State.CONFIRM, State.OFFER_GP_FALLBACK):
        if fsm.rule_reply_seems_inadequate(text, reply):
            fallback = await fsm.maybe_gemini_fallback(text)
            if fallback:
                reply = fallback

    _persist_fsm(fsm)
    return reply, keyboard_for_action(action)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user is None or update.message is None:
        return

    _get_fsm(user.id, reset=True)
    fsm = _get_fsm(user.id)
    fsm.state = State.COLLECT_NAME
    _persist_fsm(fsm)
    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "command", "/start")

    reply = "👋 مرحباً! أنا مساعد الحجز. اختاري/اختر من القائمة أو أرسل اسمك وسبب زيارتك لأساعدك بالحجز."
    voice_sent = await _send_patient_reply(update.message, reply, main_menu_keyboard())
    _log_outbound(user.id, reply, voice_sent)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.edited_message or update.channel_post or update.effective_message
    if message is None:
        return
    user = message.from_user or update.effective_user
    if user is None:
        return

    text = message.text or ""
    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "text", text)

    if text.strip() == "/start":
        await handle_start(update, context)
        return

    menu = _menu_response(user.id, text)
    if menu:
        reply, keyboard = menu
        voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=False)
        _log_outbound(user.id, reply, voice_sent)
        return

    reply, keyboard = await _handle_fsm_turn(user.id, text)
    voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=False)
    _log_outbound(user.id, reply, voice_sent)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.effective_message
    user = update.effective_user
    if message is None or user is None or message.voice is None:
        return

    try:
        voice_file = await context.bot.get_file(message.voice.file_id)
        ogg_bytes = await voice_file.download_as_bytearray()
        uname = (user.username or user.first_name or str(user.id)).strip()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        result = transcribe_voice(bytes(ogg_bytes), filename_prefix=f"{uname}_{ts}")
        text = result.get("text", "").strip()
    except Exception as exc:
        logger.warning("Voice transcription failed: %s", exc)
        text = ""

    with get_db() as db:
        crud.get_or_create_conversation(db, user.id, user.username, user.first_name, user.last_name)
        crud.log_message(db, user.id, "inbound", "voice", text)

    if not text:
        reply = "🎙️ عذراً، لم أستطع التعرف على الصوت. هل يمكنك إعادة الإرسال كنص أو كرسالة صوتية أوضح؟"
        voice_sent = await _send_patient_reply(message, reply, incoming_was_voice=True)
        _log_outbound(user.id, reply, voice_sent)
        return

    await message.reply_text(f"🎙️ فهمت رسالتك كالتالي:\n{text}")

    menu = _menu_response(user.id, text)
    if menu:
        reply, keyboard = menu
        voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=True)
        _log_outbound(user.id, reply, voice_sent)
        return

    reply, keyboard = await _handle_fsm_turn(user.id, text)
    voice_sent = await _send_patient_reply(message, reply, keyboard, incoming_was_voice=True)
    _log_outbound(user.id, reply, voice_sent)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query is None:
        return
    data = query.data or ""
    user_id = query.from_user.id
    await query.answer()

    with get_db() as db:
        crud.get_or_create_conversation(
            db,
            user_id,
            query.from_user.username,
            query.from_user.first_name,
            query.from_user.last_name,
        )
        crud.log_message(db, user_id, "inbound", "callback", data)

    if data == "menu:book":
        _begin_booking(user_id)
        reply, keyboard = "📅 جيد! ما اسمك الكريم؟", None
    elif data == "menu:inquiry":
        with get_db() as db:
            reply = _format_appointment(crud.get_latest_patient_appointment(db, user_id))
        keyboard = main_menu_keyboard()
    elif data == "menu:cancel":
        with get_db() as db:
            appt = crud.cancel_latest_patient_appointment(db, user_id)
        _get_fsm(user_id, reset=True)
        reply = "تم إلغاء آخر موعد مؤكد/منتظر. ✅" if appt else "لا يوجد موعد لإلغائه حالياً."
        keyboard = main_menu_keyboard()
    elif data == "menu:contact":
        reply, keyboard = "📞 اكتب رسالتك هنا، وسيتم حفظها في سجل محادثات العيادة.", main_menu_keyboard()
    else:
        fsm = _get_fsm(user_id)
        reply, action, _ = await fsm.handle_callback(data)
        _persist_fsm(fsm)
        keyboard = keyboard_for_action(action)

    await query.edit_message_text(reply, reply_markup=keyboard)
    voice_sent = False
    if query.message is not None and _should_send_voice(incoming_was_voice=False):
        try:
            audio_bytes = await text_to_ogg(reply)
            voice_file = io.BytesIO(audio_bytes)
            voice_file.name = "clinicai_reply.ogg"
            voice_file.seek(0)
            await query.message.reply_voice(voice=voice_file)
            voice_sent = True
        except Exception as exc:
            logger.warning("Callback voice reply failed: %s", exc)
    _log_outbound(user_id, reply, voice_sent=voice_sent)
