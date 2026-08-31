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
    # Citation graph (3c). citation_edges = arrows OUT of a doc (what it
    # cites, by normalized ref — the cited case need not be in the corpus).
    # doc_citation_keys = a doc's OWN reporter citations (from its header),
    # the reverse-lookup table that resolves a ref to an in-corpus doc.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS citation_edges (
            citing_doc_id TEXT NOT NULL,
            cited_ref TEXT NOT NULL,
            raw_text TEXT,
            occurrences INTEGER NOT NULL DEFAULT 1,
            UNIQUE(citing_doc_id, cited_ref)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_citation_keys (
            doc_id TEXT NOT NULL,
            ref TEXT NOT NULL,
            raw TEXT,
            UNIQUE(doc_id, ref)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_cited_ref ON citation_edges(cited_ref)"
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


def replace_citation_edges(
    conn: sqlite3.Connection,
    doc_id: str,
    edges: list[tuple[str, str, int]],  # (cited_ref, raw_text, occurrences)
) -> None:
    """Delete + insert = idempotent: re-extracting a doc (backfill, tuned
    regex) converges instead of accreting stale edges."""
    conn.execute("DELETE FROM citation_edges WHERE citing_doc_id = ?", (doc_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO citation_edges "
        "(citing_doc_id, cited_ref, raw_text, occurrences) VALUES (?, ?, ?, ?)",
        [(doc_id, ref, raw, occ) for ref, raw, occ in edges],
    )
    conn.commit()


def replace_doc_citation_keys(
    conn: sqlite3.Connection, doc_id: str, keys: list[tuple[str, str]]  # (ref, raw)
) -> None:
    conn.execute("DELETE FROM doc_citation_keys WHERE doc_id = ?", (doc_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO doc_citation_keys (doc_id, ref, raw) VALUES (?, ?, ?)",
        [(doc_id, ref, raw) for ref, raw in keys],
    )
    conn.commit()


def edges_cited_by(conn: sqlite3.Connection, doc_id: str) -> list[sqlite3.Row]:
    """Arrows out: what this doc cites, with in-corpus resolution when the
    ref matches some doc's own citation key."""
    return conn.execute(
        """
        SELECT e.cited_ref, e.raw_text, e.occurrences, k.doc_id AS resolved_doc_id
        FROM citation_edges e
        LEFT JOIN doc_citation_keys k ON k.ref = e.cited_ref
        WHERE e.citing_doc_id = ?
        ORDER BY e.occurrences DESC, e.cited_ref
        """,
        (doc_id,),
    ).fetchall()


def docs_citing(conn: sqlite3.Connection, refs: list[str]) -> list[sqlite3.Row]:
    """Arrows in: corpus docs whose edges point at any of these refs."""
    if not refs:
        return []
    placeholders = ",".join("?" for _ in refs)
    return conn.execute(
        f"""
        SELECT citing_doc_id, cited_ref, raw_text, occurrences
        FROM citation_edges
        WHERE cited_ref IN ({placeholders})
        ORDER BY occurrences DESC, citing_doc_id
        """,
        refs,
    ).fetchall()


def citation_keys_for_doc(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    return [
        r["ref"]
        for r in conn.execute(
            "SELECT ref FROM doc_citation_keys WHERE doc_id = ?", (doc_id,)
        ).fetchall()
    ]


def stale_documents(conn: sqlite3.Connection, current_version: int) -> list[sqlite3.Row]:
    """Docs processed by an older pipeline (or failed) — the reprocess set."""
    return conn.execute(
        "SELECT * FROM documents WHERE pipeline_version < ? OR status = 'failed' "
        "ORDER BY processed_at",
        (current_version,),
    ).fetchall()
