"""Re-run the pipeline for stale documents after improving a processing step.

    python -m app.reprocess            # all docs older than PIPELINE_VERSION (+ failed)
    python -m app.reprocess --all      # everything, regardless of version
    python -m app.reprocess <doc-id>   # one document

Workflow: improve chunker/parser/metadata -> bump PIPELINE_VERSION in
pipeline.py -> run this. Source PDFs are already persisted in output/<id>/
source.pdf, so nothing needs re-uploading. Each reprocessed doc is
re-indexed immediately (embeddings refresh with the new chunks).
"""
import sys
import time

from . import registry
from .indexer import index_document
from .pipeline import OUTPUT_DIR, PIPELINE_VERSION, process_pdf
from .vector_store import connect as vec_connect


def reprocess_one(doc_row) -> str:
    doc_id = doc_row["doc_id"]
    pdf = OUTPUT_DIR / doc_id / "source.pdf"
    if not pdf.exists():
        return f"SKIP {doc_id}: source.pdf missing"
    start = time.perf_counter()
    result = process_pdf(pdf, doc_row["source_name"], force=True)
    embedded, skipped = index_document(vec_connect(), OUTPUT_DIR / result.doc_id)
    return (
        f"OK {doc_id}: v{PIPELINE_VERSION}, {result.stats.total_chunks} chunks, "
        f"{embedded} re-embedded ({time.perf_counter() - start:.0f}s)"
    )


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    reg = registry.connect()
    if arg and arg != "--all":
        row = registry.find_by_doc_id(reg, arg)
        targets = [row] if row else []
        if not targets:
            print(f"unknown doc_id '{arg}'")
            return
    elif arg == "--all":
        targets = registry.list_documents(reg)
    else:
        targets = registry.stale_documents(reg, PIPELINE_VERSION)

    if not targets:
        print(f"nothing stale — all documents at pipeline v{PIPELINE_VERSION}")
        return
    print(f"reprocessing {len(targets)} document(s) with pipeline v{PIPELINE_VERSION}")
    for row in targets:
        try:
            print(reprocess_one(row))
        except Exception as e:  # per-doc isolation
            print(f"FAILED {row['doc_id']}: {e}")


if __name__ == "__main__":
    main()
