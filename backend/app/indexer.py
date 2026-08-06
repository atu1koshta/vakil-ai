"""Index processed documents into a profile's vector store.

    python -m app.indexer                          # all docs, active profile
    python -m app.indexer <doc-slug>               # one document
    python -m app.indexer --profile <name>         # all docs, named profile

Learning notes:
- Chunking happens HERE, from output/<doc>/markdown.md, with the profile's
  chunking config — not from chunks.json. Docling parsing (the expensive step)
  runs once per document; every profile re-derives its own chunks in memory
  (milliseconds), so chunking variants need no reprocessing. chunks.json
  remains the studio inspection artifact for the active profile.
- Embeds ENRICHED text (case title + section + chunk) so every chunk is
  findable by case name even when the chunk never mentions it; the payload
  stores the RAW text for display. Toggle via `indexing.enrich` per profile.
- content_hash makes re-runs free: unchanged chunks are skipped before any
  embedding call. Delete the profile's db file to force a full rebuild.
- Failure isolation is per document: one bad doc is reported and skipped, the
  rest of the corpus still indexes.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from .chunker import get_chunker
from .embeddings import get_embedder
from .vector_store import VectorIndex, open_store

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def enrich(case_title: str, section: str, text: str) -> str:
    header = " — ".join(p for p in (case_title, section) if p)
    return f"{header}\n{text}" if header else text


def index_document(store: VectorIndex, doc_dir: Path) -> tuple[int, int]:
    """Returns (embedded, skipped)."""
    profile = store.profile
    markdown = (doc_dir / "markdown.md").read_text()
    meta = json.loads((doc_dir / "metadata.json").read_text())
    doc_id = doc_dir.name
    case_title = meta.get("case_title") or doc_id

    chunks = get_chunker(profile.chunking).chunk_document(markdown).chunks

    known = store.existing_hashes(doc_id)
    pending = []
    for c in chunks:
        embedded_text = (
            enrich(case_title, c.section, c.text) if profile.indexing.enrich else c.text
        )
        content_hash = hashlib.sha256(embedded_text.encode()).hexdigest()
        key = f"{doc_id}:{c.id}"
        if known.get(key) == content_hash:
            continue  # unchanged — free re-run
        pending.append((key, c, embedded_text, content_hash))

    if pending:
        vectors = get_embedder(profile).embed_documents([p[2] for p in pending])
        for (key, c, _text, content_hash), vector in zip(pending, vectors):
            store.upsert_chunk(
                key=key,
                doc_id=doc_id,
                chunk_id=c.id,
                section=c.section,
                case_title=case_title,
                text=c.text,
                content_hash=content_hash,
                vector=vector,
            )
        store.commit()
    return len(pending), len(chunks) - len(pending)


def index_doc_by_id(doc_id: str, profile_name: str | None = None) -> None:
    """Background-task entrypoint: index one processed document by slug.
    Failures are logged, never raised — a background task has no requester
    to bubble up to; /documents/{doc_id}/index-status exposes the outcome."""
    import logging

    from .config import get_profile

    try:
        with open_store(get_profile(profile_name)) as store:
            embedded, skipped = index_document(store, OUTPUT_DIR / doc_id)
        logging.getLogger("vakil").info(
            "indexed %s: %d embedded, %d skipped", doc_id, embedded, skipped
        )
    except Exception:
        logging.getLogger("vakil").exception("indexing failed for %s", doc_id)


def index_status(doc_id: str, profile_name: str | None = None) -> dict:
    """Compare chunks on disk vs vectors in the profile's store for one doc.
    total_chunks comes from chunks.json — the active-profile artifact — so it
    is exact for the active profile and an estimate for chunking variants."""
    from .config import get_profile

    doc_dir = OUTPUT_DIR / doc_id
    total = 0
    if (doc_dir / "chunks.json").exists():
        total = len(json.loads((doc_dir / "chunks.json").read_text())["chunks"])
    with open_store(get_profile(profile_name)) as store:
        indexed = store.count_for_doc(doc_id)
    return {"doc_id": doc_id, "total_chunks": total, "indexed_chunks": indexed,
            "complete": total > 0 and indexed >= total}


def main() -> None:
    from .config import get_profile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_slug", nargs="?", help="index only this document")
    ap.add_argument("--profile", help="profile name (default: active profile)")
    args = ap.parse_args()

    profile = get_profile(args.profile)
    doc_dirs = [
        d for d in sorted(OUTPUT_DIR.iterdir())
        if d.is_dir() and (d / "markdown.md").exists()
        and (args.doc_slug in (None, d.name))
    ]
    if not doc_dirs:
        print(f"nothing to index under {OUTPUT_DIR}")
        return

    print(f"profile: {profile.name} (model={profile.embedding.model}, "
          f"db={profile.resolve_db_path().name})")
    with open_store(profile) as store:
        for doc_dir in doc_dirs:
            start = time.perf_counter()
            try:
                embedded, skipped = index_document(store, doc_dir)
            except Exception as e:  # per-document isolation
                print(f"FAILED {doc_dir.name}: {e}")
                continue
            print(
                f"{doc_dir.name}: {embedded} embedded, {skipped} skipped "
                f"({time.perf_counter() - start:.1f}s)"
            )
        print(f"store total: {store.count_chunks()} chunks")


if __name__ == "__main__":
    main()
