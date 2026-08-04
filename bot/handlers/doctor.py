"""
bot/handlers/doctor.py — Doctor-facing handler for session data-entry.
"""
from __future__ import annotations

import io

from telegram import Update
from telegram.ext import ContextTypes

from bot.keyboards import doctor_menu_keyboard, session_confirm_keyboard
from database.db import get_db
from database import crud
from fsm.doctor_fsm import DoctorFSM, DoctorState
from voice.stt import transcribe_voice
from voice.tts import text_to_ogg


def _load_fsm(doctor) -> DoctorFSM | None:
    if doctor is None:
        return None
    with get_db() as db:
        row = crud.get_fsm_session(db, doctor.telegram_id, role="doctor")
        if row:
            return DoctorFSM.from_snapshot(doctor.doctor_id, doctor.telegram_id, row)
    return DoctorFSM(doctor_id=doctor.doctor_id, telegram_id=doctor.telegram_id)


def _persist_fsm(fsm: DoctorFSM) -> None:
    snapshot = fsm.to_snapshot()
    with get_db() as db:
        crud.upsert_fsm_session(
            db,
            fsm.telegram_id,
            role="doctor",
            state=snapshot["state"],
            data_json=snapshot["data_json"],
        )


def _get_fsm(doctor) -> DoctorFSM | None:
    fsm = _load_fsm(doctor)
    if fsm and fsm.state == DoctorState.SAVED:
        fsm = DoctorFSM(doctor_id=doctor.doctor_id, telegram_id=doctor.telegram_id)
    return fsm


async def handle_doctor_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with get_db() as db:
        crud.log_message(db, user_id, "inbound", "command", "/start", role="doctor")

    reply = "👨‍⚕️ مرحباً دكتور! ماذا تريد؟"
    await update.message.reply_text(reply, reply_markup=doctor_menu_keyboard())

    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply, role="doctor")


async def handle_doctor_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    with get_db() as db:
        crud.get_or_create_conversation(
            db,
            user_id,
            update.effective_user.username,
            update.effective_user.first_name,
            update.effective_user.last_name,
            role="doctor",
        )
        crud.log_message(db, user_id, "inbound", "text", text, role="doctor")
        doctor = crud.get_doctor_by_telegram(db, user_id)

    if doctor is None:
        await update.message.reply_text(
            "⚠️ حسابك غير مرتبط بملف طبيب في النظام. تواصل مع إدارة العيادة.",
            reply_markup=doctor_menu_keyboard(),
        )
        return

    fsm = _get_fsm(doctor)
    reply = await fsm.handle(text)
    markup = session_confirm_keyboard() if fsm.state == DoctorState.REVIEW else doctor_menu_keyboard()
    await update.message.reply_text(reply, reply_markup=markup, parse_mode="Markdown")
    _persist_fsm(fsm)

    with get_db() as db:
        crud.log_bot_reply(db, user_id, reply, role="doctor")


async def handle_doctor_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    voice_file = await context.bot.get_file(update.message.voice.file_id)
    ogg_bytes = await voice_file.download_as_bytearray()

    result = transcribe_voice(bytes(ogg_bytes))
    text = result["text"]

    with get_db() as db:
        crud.get_or_create_conversation(
            db,
            user_id,
            update.effective_user.username,
            update.effective_user.first_name,
            update.effective_user.last_name,
            role="doctor",
        )
        crud.log_message(db, user_id, "inbound", "voice", text, role="doctor")
        doctor = crud.get_doctor_by_telegram(db, user_id)

    if doctor is None:
        await update.message.reply_text("⚠️ حسابك غير مرتبط بملف طبيب في النظام.")
        return

    await update.message.reply_text(f"🎙️ تم التعرف: _{text}_", parse_mode="Markdown")

    with get_db() as db:
        crud.log_bot_reply(db, user_id, f"🎙️ تم التعرف: {text}", role="doctor")

    fsm = _get_fsm(doctor)
    reply = await fsm.handle(text, is_voice=True)
    markup = session_confirm_keyboard() if fsm.state == DoctorState.REVIEW else doctor_menu_keyboard()
    await update.message.reply_text(reply, reply_markup=markup, parse_mode="Markdown")
    _persist_fsm(fsm)


async def handle_doctor_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    await query.answer()

    with get_db() as db:
        crud.get_or_create_conversation(
            db,
            user_id,
            query.from_user.username,
            query.from_user.first_name,
            query.from_user.last_name,
            role="doctor",
        )
        crud.log_message(db, user_id, "inbound", "callback", data or "", role="doctor")
        doctor = crud.get_doctor_by_telegram(db, user_id)

    if doctor is None:
        await query.edit_message_text("⚠️ حسابك غير مرتبط بملف طبيب في النظام.")
        return

    fsm = _get_fsm(doctor)

    if data == "doc:session":
        reply = await fsm.handle("/session")
        _persist_fsm(fsm)
        await query.edit_message_text(reply)

    elif data == "session:confirm":
        reply = await fsm.handle("تأكيد")
        _persist_fsm(fsm)
        await query.edit_message_text(reply)

    elif data == "session:discard":
        fsm.discard()
        with get_db() as db:
            crud.delete_fsm_session(db, user_id, role="doctor")
        await query.edit_message_text("🗑️ تم إلغاء الجلسة.", reply_markup=doctor_menu_keyboard())

    elif data == "doc:today":
        with get_db() as db:
            appts = crud.get_todays_queue(db)
        if not appts:
            await query.edit_message_text("لا توجد مواعيد اليوم.")
            return
        lines = [
            f"{a.appt_datetime.strftime('%H:%M')} — {a.patient.name or '؟'} — {a.priority_class}"
            for a in appts
        ]
        await query.edit_message_text("📋 مواعيد اليوم:\n" + "\n".join(lines))
