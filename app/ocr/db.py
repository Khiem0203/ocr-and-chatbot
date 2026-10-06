import logging
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ocr_db")

_TABLE = "ocr_documents"


def _connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PW"),
        connect_timeout=10,
    )


def ensure_schema() -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {_TABLE} (
        id             SERIAL PRIMARY KEY,
        bucket         TEXT NOT NULL,
        object_key     TEXT NOT NULL,
        etag           TEXT,
        status         TEXT NOT NULL DEFAULT 'processing',
        request_id     TEXT,
        document_id    INTEGER,
        creator        TEXT,
        doc_created_at TEXT,
        raw_text       TEXT,
        corrected_text TEXT,
        error          TEXT,
        updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (bucket, object_key)
    );
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def get_tracked_etags(bucket: str) -> dict:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT object_key, etag FROM {_TABLE} WHERE bucket = %s", (bucket,))
            return dict(cur.fetchall())


def mark_processing(bucket: str, object_key: str, etag: str) -> None:
    sql = f"""
    INSERT INTO {_TABLE} (bucket, object_key, etag, status, updated_at)
    VALUES (%s, %s, %s, 'processing', now())
    ON CONFLICT (bucket, object_key) DO UPDATE SET
        etag = EXCLUDED.etag, status = 'processing', updated_at = now()
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (bucket, object_key, etag))
        conn.commit()


def save_result(bucket: str, object_key: str, etag: str, result: dict) -> None:
    sql = f"""
    INSERT INTO {_TABLE} (
        bucket, object_key, etag, status, request_id, document_id,
        creator, doc_created_at, raw_text, corrected_text, error, updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (bucket, object_key) DO UPDATE SET
        etag = EXCLUDED.etag, status = EXCLUDED.status, request_id = EXCLUDED.request_id,
        document_id = EXCLUDED.document_id, creator = EXCLUDED.creator,
        doc_created_at = EXCLUDED.doc_created_at, raw_text = EXCLUDED.raw_text,
        corrected_text = EXCLUDED.corrected_text, error = EXCLUDED.error, updated_at = now()
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                bucket, object_key, etag, result.get("status", "error"),
                result.get("request_id"), result.get("document_id"),
                result.get("creator", ""), result.get("created_at", ""),
                result.get("raw_text", ""), result.get("corrected_text", ""),
                result.get("error"),
            ))
        conn.commit()


def delete_document(bucket: str, object_key: str) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {_TABLE} WHERE bucket = %s AND object_key = %s", (bucket, object_key))
        conn.commit()


def list_documents(bucket: str = None, status: str = None, limit: int = 200) -> list:
    query = (
        f"SELECT bucket, object_key, etag, status, request_id, document_id, "
        f"creator, doc_created_at, corrected_text, error, updated_at FROM {_TABLE}"
    )
    clauses, params = [], []
    if bucket:
        clauses.append("bucket = %s")
        params.append(bucket)
    if status:
        clauses.append("status = %s")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY updated_at DESC LIMIT %s"
    params.append(limit)
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, tuple(params))
            return [dict(r) for r in cur.fetchall()]
