"""Lightweight Telegram Update/Message mocks for handler E2E tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock


@dataclass
class RecordedReply:
    text: str
    reply_markup: Any = None
    parse_mode: str | None = None


class RecordingMessage:
    """Captures bot replies instead of calling the real Telegram API."""

    def __init__(self, text: str = "", *, from_user: Any = None):
        self.text = text
        self.from_user = from_user
        self.replies: list[RecordedReply] = []
        self.voice_replies: list[Any] = []

    async def reply_text(self, text: str, reply_markup=None, parse_mode=None):
        self.replies.append(RecordedReply(text=text, reply_markup=reply_markup, parse_mode=parse_mode))

    async def reply_voice(self, voice=None):
        self.voice_replies.append(voice)


class MockTelegramUser:
    def __init__(
        self,
        user_id: int,
        *,
        username: str = "test_user",
        first_name: str = "Test",
        last_name: str = "User",
    ):
        self.id = user_id
        self.username = username
        self.first_name = first_name
        self.last_name = last_name


def make_text_update(user_id: int, text: str) -> tuple[Any, RecordingMessage]:
    user = MockTelegramUser(user_id)
    message = RecordingMessage(text=text, from_user=user)
    update = MagicMock()
    update.effective_user = user
    update.message = message
    update.edited_message = None
    update.channel_post = None
    update.effective_message = message
    return update, message


def make_start_update(user_id: int) -> tuple[Any, RecordingMessage]:
    return make_text_update(user_id, "/start")


def make_context() -> MagicMock:
    return MagicMock()


def make_callback_update(user_id: int, data: str) -> tuple[Any, RecordingMessage]:
    """Inline callback query mock; query.message captures edit_message_text replies."""
    user = MockTelegramUser(user_id)
    message = RecordingMessage(from_user=user)
    query = MagicMock()
    query.from_user = user
    query.data = data
    query.message = message

    async def _answer():
        return True

    query.answer = _answer

    async def _edit_message_text(text: str, reply_markup=None, parse_mode=None):
        message.replies.append(RecordedReply(text=text, reply_markup=reply_markup, parse_mode=parse_mode))

    query.edit_message_text = _edit_message_text

    update = MagicMock()
    update.effective_user = user
    update.callback_query = query
    update.message = None
    return update, message


def format_reply_keyboard(reply_markup) -> str:
    """Human-readable keyboard rows for chat preview output."""
    if reply_markup is None:
        return ""
    keyboard = getattr(reply_markup, "keyboard", None)
    if not keyboard:
        return ""
    rows = [" | ".join(getattr(btn, "text", str(btn)) for btn in row) for row in keyboard]
    return "\n".join(f"   ⌨️  {row}" for row in rows)


def last_reply_text(message: RecordingMessage) -> str:
    assert message.replies, "expected at least one bot reply"
    return message.replies[-1].text
