"""Document registry — the catalog of processed documents (vakil.db).

Learning notes:
- Two databases, two roles: vakil.db is the CATALOG (what was processed, from
  which bytes, by which pipeline version); vectors.db is a disposable INDEX,
  rebuildable any time from output/. Delete vectors.db and nothing is lost;
  the registry is the record.
- Dedup key is the sha256 of the PDF BYTES, not the filename — the same
  judgment re-uploaded under any name maps to the already-processed document.
- pipeline_version records WHICH code produced a row. Improve the chunker,
  bump PIPELINE_VERSION in pipeline.py, and every old row becomes detectably
  stale — `python -m app.reprocess` re-runs exactly those.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "vakil.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            doc_id TEXT UNIQUE NOT NULL,
            source_name TEXT NOT NULL,
            content_sha256 TEXT UNIQUE NOT NULL,
            pipeline_version INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'processed',   -- processed | failed
            chunk_count INTEGER,
            total_tokens INTEGER,
            error TEXT,
            processed_at TEXT NOT NULL
        )
        """
    )
    return conn


def find_by_hash(conn: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE content_sha256 = ?", (sha256,)
    ).fetchone()


def find_by_doc_id(conn: sqlite3.Connection, doc_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()


def record_processed(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    source_name: str,
    sha256: str,
    pipeline_version: int,
    chunk_count: int,
    total_tokens: int,
) -> None:
    conn.execute(
        """
        INSERT INTO documents
            (doc_id, source_name, content_sha256, pipeline_version, status,
             chunk_count, total_tokens, error, processed_at)
        VALUES (?, ?, ?, ?, 'processed', ?, ?, NULL, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            source_name = excluded.source_name,
            content_sha256 = excluded.content_sha256,
            pipeline_version = excluded.pipeline_version,
            status = 'processed',
            chunk_count = excluded.chunk_count,
            total_tokens = excluded.total_tokens,
            error = NULL,
            processed_at = excluded.processed_at
        """,
        (doc_id, source_name, sha256, pipeline_version, chunk_count, total_tokens, utc_now()),
    )
    conn.commit()


def record_failed(
    conn: sqlite3.Connection, *, doc_id: str, source_name: str, sha256: str,
    pipeline_version: int, error: str,
) -> None:
    conn.execute(
        """
        INSERT INTO documents
            (doc_id, source_name, content_sha256, pipeline_version, status, error, processed_at)
        VALUES (?, ?, ?, ?, 'failed', ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            status = 'failed', error = excluded.error, processed_at = excluded.processed_at,
            pipeline_version = excluded.pipeline_version
        """,
        (doc_id, source_name, sha256, pipeline_version, error[:500], utc_now()),
    )
    conn.commit()


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM documents ORDER BY processed_at DESC").fetchall()


def stale_documents(conn: sqlite3.Connection, current_version: int) -> list[sqlite3.Row]:
    """Docs processed by an older pipeline (or failed) — the reprocess set."""
    return conn.execute(
        "SELECT * FROM documents WHERE pipeline_version < ? OR status = 'failed' "
        "ORDER BY processed_at",
        (current_version,),
    ).fetchall()
