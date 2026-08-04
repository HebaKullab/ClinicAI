"""Timezone-aware UTC timestamps stored as naive datetimes (SQLite-friendly)."""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
