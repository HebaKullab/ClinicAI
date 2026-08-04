"""Arabic speech generation for Telegram responses using edge-tts."""
from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path
from uuid import uuid4

from config import TEMP_DIR, TTS_VOICE


async def text_to_ogg(text: str) -> bytes:
    """Convert Arabic response text to Telegram-compatible OGG/Opus bytes."""
    clean = _strip_markdown(text)
    if not clean:
        raise ValueError("Cannot synthesize an empty response")

    try:
        import edge_tts
    except Exception as exc:
        raise ImportError("Install edge-tts to enable voice responses") from exc

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    mp3_path = TEMP_DIR / f"tts_{token}.mp3"
    ogg_path = TEMP_DIR / f"tts_{token}.ogg"

    try:
        communicate = edge_tts.Communicate(text=clean, voice=TTS_VOICE, rate="-5%")
        await communicate.save(str(mp3_path))
        _mp3_to_ogg(mp3_path, ogg_path)
        return ogg_path.read_bytes()
    finally:
        mp3_path.unlink(missing_ok=True)
        ogg_path.unlink(missing_ok=True)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"[*_`]", "", text or "")
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(
        r"📅|📆|✅|🏥|🏢|🔴|🟡|🟢|👤|👨‍⚕️|🩺|⏱|🔬|💊|🔭|👋|😊|🌿|🎙️|📋|✏️|❓|📌|⏰",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def _mp3_to_ogg(mp3_path: Path, ogg_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
                str(ogg_path),
            ],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for Telegram voice responses") from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to generate OGG audio: {error[:300]}") from exc


def synthesize_sync(text: str) -> bytes:
    return asyncio.run(text_to_ogg(text))
