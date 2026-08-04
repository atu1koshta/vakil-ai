"""Vector store — plain SQLite for payloads + numpy brute-force KNN.

Learning notes:
- Originally built on sqlite-vec, but python.org macOS builds compile sqlite3
  WITHOUT loadable-extension support (`enable_load_extension` missing), and
  this venv runs one. Lesson: extension-based stores depend on the host
  Python's sqlite build, pure-Python fallbacks don't.
- At hundreds-to-tens-of-thousands of chunks, EXACT brute-force cosine in
  numpy is milliseconds — an ANN index (HNSW/sqlite-vec/Qdrant) is a
  millions-of-vectors optimization, not a correctness need. The interface
  here (connect/upsert/search) is store-agnostic so swapping later touches
  only this file.
- Vectors stored as float32 BLOBs; embeddings.py normalizes them, so cosine
  similarity = plain dot product (matrix @ query). Higher = closer.
- Idempotency: key = doc_id:chunk_id is UNIQUE; content_hash lets the indexer
  skip unchanged chunks (free re-runs).
"""
import sqlite3
from pathlib import Path

import numpy as np

from .embeddings import DIM

DB_PATH = Path(__file__).resolve().parent.parent / "vectors.db"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_vectors (
            id INTEGER PRIMARY KEY,
            key TEXT UNIQUE NOT NULL,      -- doc_id:chunk_id
            doc_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            section TEXT,
            case_title TEXT,
            text TEXT NOT NULL,            -- raw chunk text (display)
            content_hash TEXT NOT NULL,    -- hash of the EMBEDDED (enriched) text
            embedding BLOB NOT NULL        -- float32[DIM], L2-normalized
        )
        """
    )
    return conn


def existing_hashes(conn: sqlite3.Connection, doc_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT key, content_hash FROM chunk_vectors WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    return {r["key"]: r["content_hash"] for r in rows}


def upsert_chunk(
    conn: sqlite3.Connection,
    *,
    key: str,
    doc_id: str,
    chunk_id: str,
    section: str,
    case_title: str,
    text: str,
    content_hash: str,
    vector: list[float],
) -> None:
    blob = np.asarray(vector, dtype=np.float32).tobytes()
    conn.execute(
        """
        INSERT INTO chunk_vectors
            (key, doc_id, chunk_id, section, case_title, text, content_hash, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            section = excluded.section,
            case_title = excluded.case_title,
            text = excluded.text,
            content_hash = excluded.content_hash,
            embedding = excluded.embedding
        """,
        (key, doc_id, chunk_id, section, case_title, text, content_hash, blob),
    )


def search(conn: sqlite3.Connection, query_vector: list[float], k: int = 5) -> list[dict]:
    """Exact KNN: one matrix-vector product over all stored vectors.
    Returns dicts with `score` = cosine similarity (higher = closer)."""
    rows = conn.execute(
        "SELECT key, doc_id, chunk_id, section, case_title, text, embedding FROM chunk_vectors"
    ).fetchall()
    if not rows:
        return []
    matrix = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(
        len(rows), DIM
    )
    scores = matrix @ np.asarray(query_vector, dtype=np.float32)  # cosine: all normalized
    top = np.argsort(scores)[::-1][:k]
    return [
        {
            "doc_id": rows[i]["doc_id"],
            "chunk_id": rows[i]["chunk_id"],
            "section": rows[i]["section"],
            "case_title": rows[i]["case_title"],
            "text": rows[i]["text"],
            "score": float(scores[i]),
        }
        for i in top
    ]


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
