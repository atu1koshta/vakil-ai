"""SQLite-backed VectorIndex: plain SQLite for payloads + numpy brute-force KNN.

Learning notes:
- Originally built on sqlite-vec, but python.org macOS builds compile sqlite3
  WITHOUT loadable-extension support (`enable_load_extension` missing), and
  this venv runs one. Lesson: extension-based stores depend on the host
  Python's sqlite build, pure-Python fallbacks don't.
- At hundreds-to-tens-of-thousands of chunks, EXACT brute-force cosine in
  numpy is milliseconds — an ANN index (HNSW/sqlite-vec/Qdrant) is a
  millions-of-vectors optimization, not a correctness need.
- Vectors stored as float32 BLOBs; embedders normalize them, so cosine
  similarity = plain dot product (matrix @ query). Higher = closer.
- Idempotency: key = doc_id:chunk_id is UNIQUE; content_hash lets the indexer
  skip unchanged chunks (free re-runs).
- The index_meta table stamps model/dim/config-fingerprint on first open and
  is verified on every open: pointing a profile at an index built with a
  different model fails loudly instead of returning garbage similarities.
  META_SCHEMA versions the STAMP FORMAT itself — when the fingerprint recipe
  changes (e.g. chunking gained a `strategy` field), stores stamped under the
  old scheme are re-stamped instead of false-alarming a mismatch.
"""
import sqlite3
from pathlib import Path

import numpy as np

from ..config import Profile, get_profile
from .base import IndexConfigMismatch, VectorIndex

# Bump when the fingerprint recipe changes; old stamps get migrated, not rejected.
META_SCHEMA = "2"


class SqliteVectorStore(VectorIndex):
    def __init__(self, profile: Profile | None = None, db_path: Path | None = None):
        self.profile = profile or get_profile()
        self.dim = self.profile.embedding.dim
        self.path = Path(db_path) if db_path else self.profile.resolve_db_path()
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,      -- doc_id:chunk_id
                doc_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                section TEXT,
                case_title TEXT,
                text TEXT NOT NULL,            -- raw chunk text (display)
                content_hash TEXT NOT NULL,    -- hash of the EMBEDDED text
                embedding BLOB NOT NULL        -- float32[dim], L2-normalized
            );
            CREATE TABLE IF NOT EXISTS index_meta (
                meta_key TEXT PRIMARY KEY,
                meta_value TEXT NOT NULL
            );
            """
        )
        self._verify_meta()

    def _verify_meta(self) -> None:
        expected = {
            "schema": META_SCHEMA,
            "embedding_model": self.profile.embedding.model,
            "dim": str(self.dim),
            "fingerprint": self.profile.fingerprint(),
        }
        stored = dict(
            self.conn.execute("SELECT meta_key, meta_value FROM index_meta").fetchall()
        )
        # Fresh store, pre-meta legacy file, or stamp from an older fingerprint
        # scheme: (re-)stamp. Content-hash dedup still protects chunk-level
        # integrity on the next index run.
        if not stored or stored.get("schema") != META_SCHEMA:
            self.conn.execute("DELETE FROM index_meta")
            self.conn.executemany(
                "INSERT INTO index_meta (meta_key, meta_value) VALUES (?, ?)",
                expected.items(),
            )
            self.conn.commit()
            return
        if stored != expected:
            self.conn.close()
            raise IndexConfigMismatch(
                f"index {self.path.name} was built as {stored}, but profile "
                f"'{self.profile.name}' expects {expected}. Delete the db file "
                f"and re-run `python -m app.indexer --profile {self.profile.name}`."
            )

    def existing_hashes(self, doc_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, content_hash FROM chunk_vectors WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        return {r["key"]: r["content_hash"] for r in rows}

    def upsert_chunk(
        self,
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
        self.conn.execute(
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

    def search(self, query_vector: list[float], k: int = 5) -> list[dict]:
        """Exact KNN: one matrix-vector product over all stored vectors.
        Returns dicts with `score` = cosine similarity (higher = closer)."""
        rows = self.conn.execute(
            "SELECT key, doc_id, chunk_id, section, case_title, text, embedding "
            "FROM chunk_vectors"
        ).fetchall()
        if not rows:
            return []
        matrix = np.frombuffer(
            b"".join(r["embedding"] for r in rows), dtype=np.float32
        ).reshape(len(rows), self.dim)
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)  # cosine
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

    def count_chunks(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]

    def count_for_doc(self, doc_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM chunk_vectors WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
