"""Process a judgment PDF end-to-end (Phase 1: stops at chunks) and persist artifacts.

Output layout per document:
  output/<doc-slug>/
    source.pdf
    markdown.md
    metadata.json
    chunks.json
    chunks/chunk_000.txt ...
"""

import json
import re
import shutil
from pathlib import Path

from .chunker import chunk_markdown, extract_tables
from .metadata import extract_metadata
from .models import ChunkStats, ProcessResult
from .parser import parse_pdf

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug[:80] or "document"


def process_pdf(pdf_path: Path, original_name: str) -> ProcessResult:
    markdown = parse_pdf(pdf_path)
    # Metadata sees the original text (citation tables help it); chunks don't.
    metadata = extract_metadata(markdown, original_name)
    chunkable_md, tables = extract_tables(markdown)
    chunks = chunk_markdown(chunkable_md)

    token_counts = [c.token_count for c in chunks] or [0]
    stats = ChunkStats(
        total_chunks=len(chunks),
        total_tokens=sum(token_counts),
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        avg_tokens=round(sum(token_counts) / max(len(chunks), 1), 1),
        sections=list(dict.fromkeys(c.section for c in chunks)),
        tables_extracted=len(tables),
    )

    doc_id = slugify(original_name)
    result = ProcessResult(
        doc_id=doc_id, metadata=metadata, markdown=markdown, chunks=chunks, stats=stats
    )
    _persist(result, pdf_path, tables)
    return result


def _persist(result: ProcessResult, pdf_path: Path, tables: list[str]) -> None:
    doc_dir = OUTPUT_DIR / result.doc_id
    chunks_dir = doc_dir / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(pdf_path, doc_dir / "source.pdf")
    (doc_dir / "markdown.md").write_text(result.markdown, encoding="utf-8")
    (doc_dir / "metadata.json").write_text(
        result.metadata.model_dump_json(indent=2), encoding="utf-8"
    )
    (doc_dir / "chunks.json").write_text(
        json.dumps(
            {
                "doc_id": result.doc_id,
                "stats": result.stats.model_dump(),
                "chunks": [c.model_dump() for c in result.chunks],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for chunk in result.chunks:
        (chunks_dir / f"{chunk.id}.txt").write_text(chunk.text, encoding="utf-8")
    if tables:
        (doc_dir / "tables.json").write_text(
            json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def get_source_pdf(doc_id: str) -> Path | None:
    path = OUTPUT_DIR / doc_id / "source.pdf"
    return path if path.exists() else None
