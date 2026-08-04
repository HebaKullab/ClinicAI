"""
bot/keyboards.py — Task: "Search for a button chat in Telegram to make easier choices"
All inline keyboard layouts live here. Import the one you need in handlers.
"""
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove


# ── Patient keyboards (use reply keyboard for conversational flow) ──────────

def urgency_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["🔴 عاجل", "🟡 خلال أسبوع"],
        ["🟢 روتيني / عادي", "أي وقت"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, one_time_keyboard=True)


def time_pref_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["اليوم", "بكرا"],
        ["بعد بكرا", "الأسبوع الجاي"],
        ["أي وقت متاح", "لا يهم"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, one_time_keyboard=True)


def confirm_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        ["✅ تأكيد الحجز", "✏️ تعديل الموعد"],
        ["🔄 موعد آخر", "❌ إلغاء"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, one_time_keyboard=True)


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    """Shown after /start — patient main menu (replaces system keyboard)."""
    rows = [
        ["📅 حجز موعد جديد", "🔍 استعلام عن موعد"],
        ["❌ إلغاء موعد", "📞 تواصل مع العيادة"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# ── Doctor keyboards ──────────────────────────────────────────────────────────

def doctor_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([
        ["🎙️ تسجيل جلسة جديدة"],
        ["📋 مواعيد اليوم"],
        ["✅ تأكيد وصول مريض"],
        ["❌ تسجيل غياب"],
    ], resize_keyboard=True)


def session_confirm_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([
        ["✅ تأكيد وحفظ"],
        ["✏️ تعديل"],
        ["🗑️ إلغاء"],
    ], resize_keyboard=True)


# ── Specialty selection ───────────────────────────────────────────────────────

def specialty_keyboard() -> ReplyKeyboardMarkup:
    """Shown when classifier confidence is low — keys match scheduler.classifier.SPECIALTY_NAMES_AR."""
    rows = [
        # ["🫀 قلب وأوعية", "🧠 أعصاب"],
        ["🫃 جهاز هضمي", "🧠 أعصاب"],
        ["🦴 عظام ومفاصل", "🌸 نساء وتوليد"],
        # ["👶 أطفال", "🦷 أسنان"],
        # ["👁️ عيون", "🧴 جلدية"],
        ["💊 أمراض مزمنة", "🧴 جلدية"],
        ["🩺 طب عام", "🧓 كبار السن"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, one_time_keyboard=True)


# ── Persistent reply keyboard (always visible) ────────────────────────────────

def patient_persistent_keyboard() -> ReplyKeyboardMarkup:
    """Stays at bottom of chat like a custom keyboard."""
    return ReplyKeyboardMarkup(
        [
            ["📅 حجز موعد", "🔍 موعدي"],
            ["❌ إلغاء موعد", "📞 تواصل"],
        ],
        resize_keyboard=True,
        input_field_placeholder="اكتب أو اختر من القائمة...",
    )
