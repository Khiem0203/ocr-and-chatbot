import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

_TABLE = "chat_history"

MAX_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "3"))
SESSION_TIMEOUT_MINUTES = int(os.getenv("CHAT_SESSION_TIMEOUT_MINUTES", "90"))


def _connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PW"),
        connect_timeout=10,
    )


_schema_ready = False


def ensure_schema() -> None:
    global _schema_ready
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {_TABLE} (
        id          SERIAL PRIMARY KEY,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_{_TABLE}_session ON {_TABLE} (session_id, id);
    ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS doc_bucket TEXT;
    ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS doc_key TEXT;
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    _schema_ready = True


def _ensure_ready() -> None:
    if not _schema_ready:
        ensure_schema()


def get_recent_turns(session_id: str, max_turns: int = MAX_HISTORY_TURNS,
                      timeout_minutes: int = SESSION_TIMEOUT_MINUTES) -> list:
    _ensure_ready()
    sql = f"""
    SELECT role, content FROM {_TABLE}
    WHERE session_id = %s
      AND created_at > now() - (%s * interval '1 minute')
    ORDER BY id DESC
    LIMIT %s
    """
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_id, timeout_minutes, max_turns * 2))
            rows = [dict(r) for r in cur.fetchall()]
    rows.reverse()
    return rows


def cleanup_expired_sessions(timeout_minutes: int = SESSION_TIMEOUT_MINUTES) -> int:
    _ensure_ready()
    sql = f"DELETE FROM {_TABLE} WHERE created_at <= now() - (%s * interval '1 minute')"
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (timeout_minutes,))
            deleted = cur.rowcount
        conn.commit()
    return deleted


def save_turn(session_id: str, question: str, answer: str, doc: tuple = None) -> None:
    _ensure_ready()
    sql = f"INSERT INTO {_TABLE} (session_id, role, content, doc_bucket, doc_key) VALUES (%s, %s, %s, %s, %s)"
    doc_bucket, doc_key = doc if doc else (None, None)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, "user", question, None, None))
            cur.execute(sql, (session_id, "assistant", answer, doc_bucket, doc_key))
        conn.commit()


def get_last_document(session_id: str, timeout_minutes: int = SESSION_TIMEOUT_MINUTES):
    _ensure_ready()
    sql = f"""
    SELECT doc_bucket, doc_key FROM {_TABLE}
    WHERE session_id = %s
      AND doc_key IS NOT NULL
      AND created_at > now() - (%s * interval '1 minute')
    ORDER BY id DESC
    LIMIT 1
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, timeout_minutes))
            row = cur.fetchone()
    return (row[0], row[1]) if row else None


def clear_session(session_id: str) -> None:
    _ensure_ready()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {_TABLE} WHERE session_id = %s", (session_id,))
        conn.commit()
