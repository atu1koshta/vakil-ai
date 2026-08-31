"""Backfill citation edges for already-processed documents.

    python -m app.citations.backfill              # every doc under output/
    python -m app.citations.backfill <doc-id>...  # just these

Reads output/<doc>/markdown.md — Docling never runs, PIPELINE_VERSION is
untouched. Re-runnable for free after every extractor tweak (that is the
point of doing 3c outside the pipeline-version mechanism: regex tuning
iterates in seconds, not Docling-minutes per doc).
"""

import sys

from .. import registry
from ..pipeline import OUTPUT_DIR
from .store import extract_and_store


def backfill(doc_ids: list[str] | None = None) -> None:
    if doc_ids:
        doc_dirs = [OUTPUT_DIR / d for d in doc_ids]
    else:
        doc_dirs = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_dir())

    conn = registry.connect()
    done = failed = 0
    try:
        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            md_path = doc_dir / "markdown.md"
            if not md_path.exists():
                print(f"SKIP    {doc_id} — no markdown.md")
                continue
            try:  # per-item failure isolation: one bad doc never stops the run
                own, cited = extract_and_store(
                    conn,
                    doc_id=doc_id,
                    markdown=md_path.read_text(encoding="utf-8"),
                    doc_dir=doc_dir,
                )
                done += 1
                print(f"OK      {doc_id} — {len(cited)} edges, {len(own)} own refs")
            except Exception as e:
                failed += 1
                print(f"FAILED  {doc_id} — {e}")
    finally:
        conn.close()
    print(f"\nbackfill: {done} ok, {failed} failed, {len(doc_dirs)} considered")


if __name__ == "__main__":
    backfill(sys.argv[1:] or None)
