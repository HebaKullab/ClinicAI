"""
main.py — Task: "Server connecting all components"
Single entry point. Starts:
  1. SQLite DB (creates tables if new)
  2. FastAPI dashboard server (background thread)
  3. Telegram bot (main thread, long-polling)
"""
import logging
import socket
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from bot.router import route_callback, route_start, route_text, route_voice
from config import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    TELEGRAM_BOOTSTRAP_RETRIES,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CONNECT_TIMEOUT,
    TELEGRAM_POOL_TIMEOUT,
    TELEGRAM_PROXY,
    TELEGRAM_READ_TIMEOUT,
    TELEGRAM_TRUST_ENV,
    TELEGRAM_WRITE_TIMEOUT,
)
from dashboard.routes import router as dashboard_router
from database.db import init_db

logger = logging.getLogger(__name__)


# ── FastAPI app ────────────────────────────────────────────────────────────────

def create_fastapi_app() -> FastAPI:
    app = FastAPI(title="ClinicAI Dashboard", docs_url=None)
    app.include_router(dashboard_router)
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception:
        pass
    return app


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def run_dashboard():
    app = create_fastapi_app()
    uvicorn.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",   # quiet — bot logs are enough
    )


# ── Telegram bot ───────────────────────────────────────────────────────────────

def build_bot() -> Application:
    httpx_kwargs: dict = {"trust_env": TELEGRAM_TRUST_ENV}
    if not TELEGRAM_TRUST_ENV and not TELEGRAM_PROXY:
        httpx_kwargs["proxy"] = None

    request = HTTPXRequest(
        connect_timeout=TELEGRAM_CONNECT_TIMEOUT,
        read_timeout=TELEGRAM_READ_TIMEOUT,
        write_timeout=TELEGRAM_WRITE_TIMEOUT,
        pool_timeout=TELEGRAM_POOL_TIMEOUT,
        proxy=TELEGRAM_PROXY,
        httpx_kwargs=httpx_kwargs,
    )
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Telegram update failed: %s", context.error, exc_info=context.error)

    app.add_error_handler(on_error)

    app.add_handler(CommandHandler("start",   route_start))
    app.add_handler(MessageHandler(filters.VOICE,                    route_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  route_text))
    app.add_handler(CallbackQueryHandler(route_callback))

    return app

# Database initialization is now handled by the database package.
# ── Startup ────────────────────────────────────────────────────────────────────

def main():
    print("🏥 ClinicAI starting...")

    # 1. Initialize database
    init_db()
    print("✅ Database ready")

    # 2. Start dashboard in a background daemon thread
    if _port_in_use(DASHBOARD_HOST, DASHBOARD_PORT):
        print(
            f"⚠️ Dashboard port {DASHBOARD_PORT} is already in use — "
            "skipping dashboard (bot will still run). Stop the other process or change DASHBOARD_PORT."
        )
    else:
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        print(f"✅ Dashboard → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    # 3. Start Telegram bot (blocking — keeps the process alive)
    proxy_mode = TELEGRAM_PROXY or ("system/env" if TELEGRAM_TRUST_ENV else "direct")
    print(f"✅ Bot is running (Telegram: {proxy_mode}). Press Ctrl+C to stop.\n")
    bot = build_bot()
    bot.run_polling(
        drop_pending_updates=True,
        timeout=30,
        bootstrap_retries=TELEGRAM_BOOTSTRAP_RETRIES,
    )


if __name__ == "__main__":
    main()
