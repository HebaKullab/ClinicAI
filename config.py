from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()  # Load environment variables from .env file


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
TTS_VOICE = os.getenv("TTS_VOICE", "ar-PS-SamaNeural")
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
TTS_RESPONSE_MODE = os.getenv("TTS_RESPONSE_MODE", "auto").strip().lower()
if TTS_RESPONSE_MODE not in {"text", "voice", "both", "auto"}:
    TTS_RESPONSE_MODE = "auto"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemma-4-31b-it")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", "openai/gpt-4.1-mini")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", GEMINI_FALLBACK_MODEL)

CLINIC_NAME = os.getenv("CLINIC_NAME", "العيادة")
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = _get_int_env("DASHBOARD_PORT", 8000)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip() or None
# Broken system HTTP_PROXY often causes Telegram timeouts on Windows — direct by default.
TELEGRAM_TRUST_ENV = _get_bool_env("TELEGRAM_TRUST_ENV", False)
TELEGRAM_CONNECT_TIMEOUT = _get_float_env("TELEGRAM_CONNECT_TIMEOUT", 30.0)
TELEGRAM_READ_TIMEOUT = _get_float_env("TELEGRAM_READ_TIMEOUT", 30.0)
TELEGRAM_WRITE_TIMEOUT = _get_float_env("TELEGRAM_WRITE_TIMEOUT", 30.0)
TELEGRAM_POOL_TIMEOUT = _get_float_env("TELEGRAM_POOL_TIMEOUT", 10.0)
TELEGRAM_BOOTSTRAP_RETRIES = _get_int_env("TELEGRAM_BOOTSTRAP_RETRIES", 5)

if not TELEGRAM_TRUST_ENV and not TELEGRAM_PROXY:
    for _proxy_key in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        os.environ.pop(_proxy_key, None)

USE_SLOT_POLICY = os.getenv("USE_SLOT_POLICY", "true").strip().lower() in {"1", "true", "yes", "on"}
