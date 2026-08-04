from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path

from sqlalchemy import create_engine, event, func, inspect, select, text
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "clinic.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """Enforce declared foreign keys on every SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Register all models with SQLAlchemy metadata.
from . import models  # noqa: E402, F401


DEFAULT_DOCTORS = [
    {
        "name": "د. أحمد الخطيب",
        "specialty": "general_practice",
        "clinic_code": "CLINIC-GP",
        "clinic_name": "عيادة الطب العام",
    },
    {
        "name": "د. عمر نصار",
        "specialty": "cardiology",
        "clinic_code": "CLINIC-CARD",
        "clinic_name": "عيادة القلب",
    },
    {
        "name": "د. لينا الشامي",
        "specialty": "neurology",
        "clinic_code": "CLINIC-NEURO",
        "clinic_name": "عيادة الأعصاب",
    },
    {
        "name": "د. سامر حمدان",
        "specialty": "orthopedics",
        "clinic_code": "CLINIC-ORTHO",
        "clinic_name": "عيادة العظام",
    },
    {
        "name": "د. رانيا أبو عيشة",
        "specialty": "pediatrics",
        "clinic_code": "CLINIC-PED",
        "clinic_name": "عيادة الأطفال",
    },
    {
        "name": "د. هالة المصري",
        "specialty": "gynecology",
        "clinic_code": "CLINIC-GYN",
        "clinic_name": "عيادة النساء والولادة",
    },
    {
        "name": "د. يوسف جودة",
        "specialty": "dentistry",
        "clinic_code": "CLINIC-DENT",
        "clinic_name": "عيادة الأسنان",
    },
    {
        "name": "د. نور بركات",
        "specialty": "dermatology",
        "clinic_code": "CLINIC-DERM",
        "clinic_name": "عيادة الجلدية",
    },
    {
        "name": "د. ماجد عواد",
        "specialty": "gastroenterology",
        "clinic_code": "CLINIC-GI",
        "clinic_name": "عيادة الجهاز الهضمي",
    },
    {
        "name": "د. سلمى قاسم",
        "specialty": "chronic_diseases",
        "clinic_code": "CLINIC-CHR",
        "clinic_name": "عيادة الأمراض المزمنة",
    },
    {
        "name": "د. فؤاد زيدان",
        "specialty": "elderly",
        "clinic_code": "CLINIC-ELD",
        "clinic_name": "عيادة كبار السن",
    },
]


def _sqlite_column_exists(table_name: str, column_name: str) -> bool:
    inspector = inspect(engine)
    try:
        return column_name in [col["name"] for col in inspector.get_columns(table_name)]
    except Exception:
        return False


def _sqlite_add_column(table_name: str, column_name: str, column_sql: str) -> None:
    """Safe additive migration for existing demo SQLite files."""
    if _sqlite_column_exists(table_name, column_name):
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone() is not None


def _rebuild_doctors_table_if_needed() -> None:
    """
    Root migration for the doctors table.

    It makes telegram_id optional and adds clinic identity fields while preserving
    all existing doctor_id values and data. SQLite cannot drop a NOT NULL constraint
    in place, so the table is rebuilt once when an old schema is detected.
    """
    if not DB_FILE.exists():
        return

    engine.dispose()
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        if not _sqlite_table_exists(conn, "doctors"):
            return

        info = conn.execute("PRAGMA table_info(doctors)").fetchall()
        by_name = {row["name"]: row for row in info}
        required = {"clinic_code", "clinic_name", "is_active"}
        telegram_is_required = bool(by_name.get("telegram_id") and by_name["telegram_id"]["notnull"])
        if required.issubset(by_name) and not telegram_is_required:
            return

        rows = conn.execute("SELECT * FROM doctors ORDER BY doctor_id").fetchall()
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE IF EXISTS doctors_new")
        conn.execute(
            """
            CREATE TABLE doctors_new (
                doctor_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER,
                name VARCHAR(255) NOT NULL,
                specialty VARCHAR(128) NOT NULL,
                clinic_code VARCHAR(32) NOT NULL UNIQUE,
                clinic_name VARCHAR(255) NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at DATETIME
            )
            """
        )

        old_columns = set(by_name)
        used_codes: set[str] = set()
        for row in rows:
            doctor_id = row["doctor_id"]
            name = (row["name"] if "name" in old_columns else None) or f"طبيب {doctor_id}"
            specialty = (row["specialty"] if "specialty" in old_columns else None) or "general_practice"
            telegram_id = row["telegram_id"] if "telegram_id" in old_columns else None
            code = row["clinic_code"] if "clinic_code" in old_columns else None
            code = code or f"CLINIC-{doctor_id:02d}"
            while code in used_codes:
                code = f"{code}-{doctor_id}"
            used_codes.add(code)
            clinic_name = (
                row["clinic_name"] if "clinic_name" in old_columns else None
            ) or f"عيادة {name}"
            is_active = row["is_active"] if "is_active" in old_columns else 1
            created_at = row["created_at"] if "created_at" in old_columns else datetime.utcnow().isoformat()

            conn.execute(
                """
                INSERT INTO doctors_new (
                    doctor_id, telegram_id, name, specialty,
                    clinic_code, clinic_name, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doctor_id,
                    telegram_id,
                    name,
                    specialty,
                    code,
                    clinic_name,
                    1 if is_active is None else int(bool(is_active)),
                    created_at,
                ),
            )

        conn.execute("DROP TABLE doctors")
        conn.execute("ALTER TABLE doctors_new RENAME TO doctors")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_doctors_telegram_id ON doctors (telegram_id)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_doctors_clinic_code ON doctors (clinic_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_doctors_specialty ON doctors (specialty)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_doctors_is_active ON doctors (is_active)")
        conn.commit()


def seed_default_doctors() -> None:
    """Create/update clinic doctors without requiring Telegram accounts."""
    from .models import Doctor

    with SessionLocal() as db:
        for item in DEFAULT_DOCTORS:
            doctor = db.scalar(
                select(Doctor).where(
                    (Doctor.clinic_code == item["clinic_code"])
                    | (Doctor.specialty == item["specialty"])
                )
            )
            if doctor is None:
                doctor = Doctor(
                    telegram_id=None,
                    name=item["name"],
                    specialty=item["specialty"],
                    clinic_code=item["clinic_code"],
                    clinic_name=item["clinic_name"],
                    is_active=True,
                )
                db.add(doctor)
            else:
                # Preserve an optional Telegram ID if one was later assigned.
                doctor.name = item["name"]
                doctor.specialty = item["specialty"]
                doctor.clinic_code = item["clinic_code"]
                doctor.clinic_name = item["clinic_name"]
                doctor.is_active = True
        db.commit()


def _rebuild_slots_table_with_doctor_fk() -> None:
    """
    Link every slot to a doctor/clinic and enforce the FK for existing SQLite DBs.
    Existing slot IDs and booked statuses are preserved.
    """
    if not DB_FILE.exists():
        return

    engine.dispose()
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        if not _sqlite_table_exists(conn, "slots"):
            return

        info = conn.execute("PRAGMA table_info(slots)").fetchall()
        columns = {row["name"]: row for row in info}
        if "doctor_id" not in columns:
            conn.execute("ALTER TABLE slots ADD COLUMN doctor_id INTEGER")
            conn.commit()
            info = conn.execute("PRAGMA table_info(slots)").fetchall()
            columns = {row["name"]: row for row in info}

        # Assign old specialty-only slots to the active doctor representing that clinic.
        conn.execute(
            """
            UPDATE slots
            SET doctor_id = (
                SELECT d.doctor_id
                FROM doctors AS d
                WHERE d.specialty = slots.specialty AND d.is_active = 1
                ORDER BY d.doctor_id
                LIMIT 1
            )
            WHERE doctor_id IS NULL
            """
        )
        gp = conn.execute(
            "SELECT doctor_id FROM doctors WHERE specialty='general_practice' AND is_active=1 ORDER BY doctor_id LIMIT 1"
        ).fetchone()
        if gp:
            conn.execute("UPDATE slots SET doctor_id=? WHERE doctor_id IS NULL", (gp[0],))
        conn.commit()

        null_count = conn.execute("SELECT COUNT(*) FROM slots WHERE doctor_id IS NULL").fetchone()[0]
        if null_count:
            raise RuntimeError("Could not assign every slot to a doctor clinic.")

        fk_rows = conn.execute("PRAGMA foreign_key_list(slots)").fetchall()
        has_doctor_fk = any(row[3] == "doctor_id" and row[2] == "doctors" for row in fk_rows)
        doctor_not_null = bool(columns["doctor_id"]["notnull"])
        if has_doctor_fk and doctor_not_null:
            return

        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE IF EXISTS slots_new")
        conn.execute(
            """
            CREATE TABLE slots_new (
                slot_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER NOT NULL,
                slot_datetime DATETIME NOT NULL,
                specialty VARCHAR(128) NOT NULL,
                priority_class VARCHAR(8),
                status VARCHAR(32) NOT NULL,
                notes TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY(doctor_id) REFERENCES doctors (doctor_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO slots_new (
                slot_id, doctor_id, slot_datetime, specialty, priority_class,
                status, notes, created_at, updated_at
            )
            SELECT
                slot_id, doctor_id, slot_datetime, specialty, priority_class,
                status, notes, created_at, updated_at
            FROM slots
            """
        )
        conn.execute("DROP TABLE slots")
        conn.execute("ALTER TABLE slots_new RENAME TO slots")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_slots_doctor_id ON slots (doctor_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_slots_slot_datetime ON slots (slot_datetime)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_slots_specialty ON slots (specialty)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_slots_status ON slots (status)")
        conn.commit()


def _ensure_fsm_sessions_table() -> None:
    """Create fsm_sessions table on existing SQLite DBs (additive migration)."""
    if not DB_FILE.exists():
        return

    engine.dispose()
    with sqlite3.connect(DB_FILE) as conn:
        if _sqlite_table_exists(conn, "fsm_sessions"):
            return
        conn.execute(
            """
            CREATE TABLE fsm_sessions (
                session_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER NOT NULL,
                role VARCHAR(32) NOT NULL DEFAULT 'patient',
                state VARCHAR(64) NOT NULL,
                data_json JSON,
                slot_options_json JSON,
                slot_index INTEGER NOT NULL DEFAULT 0,
                slot_json JSON,
                priority_json JSON,
                finalized_appointment_id VARCHAR(64),
                updated_at DATETIME,
                UNIQUE (telegram_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS ix_fsm_sessions_telegram_id ON fsm_sessions (telegram_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_fsm_sessions_role ON fsm_sessions (role)")
        conn.commit()


def migrate_sqlite_schema() -> None:
    """Bring the bundled SQLite DB to the V4 schema while preserving its data."""
    if not DB_FILE.exists():
        return

    _rebuild_doctors_table_if_needed()

    # patients
    _sqlite_add_column("patients", "phone", "VARCHAR(64)")
    _sqlite_add_column("patients", "updated_at", "DATETIME")

    # appointments
    _sqlite_add_column("appointments", "slot_id", "INTEGER")
    _sqlite_add_column("appointments", "specialty_ar", "VARCHAR(128)")
    _sqlite_add_column("appointments", "priority_score", "FLOAT")
    _sqlite_add_column("appointments", "complaint_summary", "TEXT")
    _sqlite_add_column("appointments", "time_preference", "JSON")
    _sqlite_add_column("appointments", "updated_at", "DATETIME")

    # slots
    _sqlite_add_column("slots", "created_at", "DATETIME")
    _sqlite_add_column("slots", "updated_at", "DATETIME")

    # sessions
    _sqlite_add_column("sessions", "patient_id", "INTEGER")
    _sqlite_add_column("sessions", "appointment_id", "VARCHAR(64)")

    # patient_profiles
    _sqlite_add_column("patient_profiles", "patient_id", "INTEGER")

    # conversations / message logs linkage
    _sqlite_add_column("conversations", "patient_id", "INTEGER")
    _sqlite_add_column("conversations", "role", "VARCHAR(32) DEFAULT 'patient'")
    _sqlite_add_column("message_logs", "conversation_id", "INTEGER")
    _sqlite_add_column("message_logs", "patient_id", "INTEGER")

    _ensure_fsm_sessions_table()



def _rebuild_appointments_table_with_slot_fk() -> None:
    """Rebuild legacy appointments so patient_id and slot_id are real foreign keys."""
    if not DB_FILE.exists():
        return

    engine.dispose()
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        if not _sqlite_table_exists(conn, "appointments"):
            return

        fk_rows = conn.execute("PRAGMA foreign_key_list(appointments)").fetchall()
        has_patient_fk = any(row[3] == "patient_id" and row[2] == "patients" for row in fk_rows)
        has_slot_fk = any(row[3] == "slot_id" and row[2] == "slots" for row in fk_rows)
        if has_patient_fk and has_slot_fk:
            return

        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            UPDATE appointments
            SET slot_id = NULL
            WHERE slot_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM slots WHERE slots.slot_id = appointments.slot_id)
            """
        )
        conn.execute("DROP TABLE IF EXISTS appointments_new")
        conn.execute(
            """
            CREATE TABLE appointments_new (
                appt_id VARCHAR(64) NOT NULL PRIMARY KEY,
                patient_id INTEGER,
                slot_id INTEGER,
                appt_datetime DATETIME,
                specialty VARCHAR(128),
                specialty_ar VARCHAR(128),
                priority_class VARCHAR(8),
                priority_score FLOAT,
                complaint_summary TEXT,
                time_preference JSON,
                status VARCHAR(32) NOT NULL DEFAULT 'confirmed',
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY(patient_id) REFERENCES patients (patient_id),
                FOREIGN KEY(slot_id) REFERENCES slots (slot_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO appointments_new (
                appt_id, patient_id, slot_id, appt_datetime, specialty, specialty_ar,
                priority_class, priority_score, complaint_summary, time_preference,
                status, created_at, updated_at
            )
            SELECT
                appt_id, patient_id, slot_id, appt_datetime, specialty, specialty_ar,
                priority_class, priority_score, complaint_summary, time_preference,
                status, created_at, updated_at
            FROM appointments
            """
        )
        conn.execute("DROP TABLE appointments")
        conn.execute("ALTER TABLE appointments_new RENAME TO appointments")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_appointments_slot_id ON appointments (slot_id)")
        conn.commit()


def _rebuild_sessions_table_with_all_fks() -> None:
    """Rebuild legacy sessions so doctor, patient and appointment links are enforced."""
    if not DB_FILE.exists():
        return

    engine.dispose()
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        if not _sqlite_table_exists(conn, "sessions"):
            return

        fk_rows = conn.execute("PRAGMA foreign_key_list(sessions)").fetchall()
        has_doctor_fk = any(row[3] == "doctor_id" and row[2] == "doctors" for row in fk_rows)
        has_patient_fk = any(row[3] == "patient_id" and row[2] == "patients" for row in fk_rows)
        has_appointment_fk = any(row[3] == "appointment_id" and row[2] == "appointments" for row in fk_rows)
        if has_doctor_fk and has_patient_fk and has_appointment_fk:
            return

        fallback_doctor = conn.execute(
            "SELECT doctor_id FROM doctors WHERE is_active=1 ORDER BY doctor_id LIMIT 1"
        ).fetchone()
        if fallback_doctor is None:
            raise RuntimeError("Cannot migrate sessions before at least one doctor exists.")

        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "UPDATE sessions SET doctor_id=? WHERE doctor_id IS NULL OR NOT EXISTS "
            "(SELECT 1 FROM doctors WHERE doctors.doctor_id=sessions.doctor_id)",
            (fallback_doctor[0],),
        )
        conn.execute(
            "UPDATE sessions SET patient_id=NULL WHERE patient_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM patients WHERE patients.patient_id=sessions.patient_id)"
        )
        conn.execute(
            "UPDATE sessions SET appointment_id=NULL WHERE appointment_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM appointments WHERE appointments.appt_id=sessions.appointment_id)"
        )
        conn.execute("DROP TABLE IF EXISTS sessions_new")
        conn.execute(
            """
            CREATE TABLE sessions_new (
                session_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER NOT NULL,
                patient_id INTEGER,
                appointment_id VARCHAR(64),
                patient_name VARCHAR(255),
                chief_complaint TEXT,
                diagnosis TEXT,
                medications JSON,
                investigations JSON,
                followup_days INTEGER,
                raw_transcription TEXT,
                session_datetime DATETIME,
                FOREIGN KEY(doctor_id) REFERENCES doctors (doctor_id),
                FOREIGN KEY(patient_id) REFERENCES patients (patient_id),
                FOREIGN KEY(appointment_id) REFERENCES appointments (appt_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions_new (
                session_id, doctor_id, patient_id, appointment_id, patient_name,
                chief_complaint, diagnosis, medications, investigations,
                followup_days, raw_transcription, session_datetime
            )
            SELECT
                session_id, doctor_id, patient_id, appointment_id, patient_name,
                chief_complaint, diagnosis, medications, investigations,
                followup_days, raw_transcription, session_datetime
            FROM sessions
            """
        )
        conn.execute("DROP TABLE sessions")
        conn.execute("ALTER TABLE sessions_new RENAME TO sessions")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sessions_doctor_id ON sessions (doctor_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sessions_patient_id ON sessions (patient_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sessions_appointment_id ON sessions (appointment_id)")
        conn.commit()

def seed_default_slots(days: int = 14) -> None:
    """Ensure each active doctor has a future schedule owned by that doctor."""
    from .models import Doctor, Slot

    slot_plan = [
        (time(9, 0), "P1"),
        (time(9, 30), "P1"),
        (time(10, 0), "P2"),
        (time(10, 30), "P2"),
        (time(11, 0), None),
        (time(11, 30), None),
        (time(12, 0), "P3"),
        (time(12, 30), "P3"),
        (time(13, 0), None),
        (time(13, 30), None),
    ]

    today = date.today()
    end_day = today + timedelta(days=max(days - 1, 0))
    with SessionLocal() as db:
        doctors = db.scalars(
            select(Doctor).where(Doctor.is_active.is_(True)).order_by(Doctor.doctor_id)
        ).all()
        for doctor in doctors:
            existing_datetimes = set(
                db.scalars(
                    select(Slot.slot_datetime).where(
                        Slot.doctor_id == doctor.doctor_id,
                        Slot.slot_datetime >= datetime.combine(today, time.min),
                        Slot.slot_datetime <= datetime.combine(end_day, time.max),
                    )
                ).all()
            )

            for offset in range(days):
                day = today + timedelta(days=offset)
                if day.weekday() == 4:  # Friday off
                    continue
                for slot_time, priority_class in slot_plan:
                    slot_datetime = datetime.combine(day, slot_time)
                    if slot_datetime in existing_datetimes:
                        continue
                    db.add(
                        Slot(
                            doctor_id=doctor.doctor_id,
                            slot_datetime=slot_datetime,
                            specialty=doctor.specialty,
                            priority_class=priority_class,
                            status="available",
                            notes=f"auto-seeded for {doctor.clinic_code}",
                        )
                    )
                    existing_datetimes.add(slot_datetime)
        db.commit()


def init_db() -> None:
    """Create/migrate DB, seed doctors, link slots, then ensure schedules exist."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    migrate_sqlite_schema()
    seed_default_doctors()
    _rebuild_slots_table_with_doctor_fk()
    _rebuild_appointments_table_with_slot_fk()
    _rebuild_sessions_table_with_all_fks()
    seed_default_slots()


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_dependency():
    with get_db() as db:
        yield db
