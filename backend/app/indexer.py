"""Index chunked documents into the vector store.

    python -m app.indexer            # index everything under output/
    python -m app.indexer <doc-slug> # index one document

Learning notes:
- Embeds ENRICHED text (case title + section + chunk) so every chunk is
  findable by case name even when the chunk never mentions it; the payload
  stores the RAW text for display. Enrichment costs nothing at index time and
  lifts retrieval quality more than most model choices.
- content_hash makes re-runs free: unchanged chunks are skipped before any
  embedding call happens. Delete vectors.db to force a full rebuild.
- Failure isolation is per document: one bad doc is reported and skipped, the
  rest of the corpus still indexes.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

from .embeddings import embed_documents
from .vector_store import connect, count_chunks, existing_hashes, upsert_chunk

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def enrich(case_title: str, section: str, text: str) -> str:
    header = " — ".join(p for p in (case_title, section) if p)
    return f"{header}\n{text}" if header else text


def index_document(conn, doc_dir: Path) -> tuple[int, int]:
    """Returns (embedded, skipped)."""
    data = json.loads((doc_dir / "chunks.json").read_text())
    meta = json.loads((doc_dir / "metadata.json").read_text())
    doc_id = data["doc_id"]
    case_title = meta.get("case_title") or doc_id

    known = existing_hashes(conn, doc_id)
    pending = []
    for c in data["chunks"]:
        embedded_text = enrich(case_title, c["section"], c["text"])
        content_hash = hashlib.sha256(embedded_text.encode()).hexdigest()
        key = f"{doc_id}:{c['id']}"
        if known.get(key) == content_hash:
            continue  # unchanged — free re-run
        pending.append((key, c, embedded_text, content_hash))

    if pending:
        vectors = embed_documents([p[2] for p in pending])
        for (key, c, _text, content_hash), vector in zip(pending, vectors):
            upsert_chunk(
                conn,
                key=key,
                doc_id=doc_id,
                chunk_id=c["id"],
                section=c["section"],
                case_title=case_title,
                text=c["text"],
                content_hash=content_hash,
                vector=vector,
            )
        conn.commit()
    return len(pending), len(data["chunks"]) - len(pending)


def index_doc_by_id(doc_id: str) -> None:
    """Background-task entrypoint: index one processed document by slug.
    Failures are logged, never raised — a background task has no requester
    to bubble up to; /documents/{doc_id}/index-status exposes the outcome."""
    import logging

    doc_dir = OUTPUT_DIR / doc_id
    conn = connect()
    try:
        embedded, skipped = index_document(conn, doc_dir)
        logging.getLogger("vakil").info(
            "indexed %s: %d embedded, %d skipped", doc_id, embedded, skipped
        )
    except Exception:
        logging.getLogger("vakil").exception("indexing failed for %s", doc_id)
    finally:
        conn.close()


def index_status(doc_id: str) -> dict:
    """Compare chunks on disk vs vectors in store for one document."""
    doc_dir = OUTPUT_DIR / doc_id
    total = 0
    if (doc_dir / "chunks.json").exists():
        total = len(json.loads((doc_dir / "chunks.json").read_text())["chunks"])
    conn = connect()
    try:
        indexed = conn.execute(
            "SELECT COUNT(*) FROM chunk_vectors WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    return {"doc_id": doc_id, "total_chunks": total, "indexed_chunks": indexed,
            "complete": total > 0 and indexed >= total}


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    doc_dirs = [
        d for d in sorted(OUTPUT_DIR.iterdir())
        if d.is_dir() and (d / "chunks.json").exists() and (only in (None, d.name))
    ]
    if not doc_dirs:
        print(f"nothing to index under {OUTPUT_DIR}")
        return

    conn = connect()
    for doc_dir in doc_dirs:
        start = time.perf_counter()
        try:
            embedded, skipped = index_document(conn, doc_dir)
        except Exception as e:  # per-document isolation
            print(f"FAILED {doc_dir.name}: {e}")
            continue
        print(
            f"{doc_dir.name}: {embedded} embedded, {skipped} skipped "
            f"({time.perf_counter() - start:.1f}s)"
        )
    print(f"store total: {count_chunks(conn)} chunks")


if __name__ == "__main__":
    main()
