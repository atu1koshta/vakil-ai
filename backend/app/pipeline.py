"""Process a judgment PDF end-to-end and persist artifacts.

Output layout per document:
  output/<doc-slug>/
    source.pdf
    markdown.md
    metadata.json
    chunks.json
    chunks/chunk_000.txt ...

Idempotency (registry.py): the PDF's sha256 is looked up in the documents
catalog first — same bytes already processed by the CURRENT pipeline version
=> cached result loaded from output/, docling never runs. `force=True` or a
PIPELINE_VERSION bump re-runs the real pipeline.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

from . import registry
from .chunker import get_chunker
from .config import get_profile
from .metadata import get_metadata_extractor
from .models import ChunkStats, DocumentMetadata, ProcessResult
from .parser import get_parser

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Bump when any processing step (parser/chunker/metadata) improves —
# previously processed docs become stale; `python -m app.reprocess` redoes them.
PIPELINE_VERSION = 1


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug[:80] or "document"


def load_result(doc_id: str, *, cached: bool, pipeline_version: int) -> ProcessResult:
    """Rebuild a ProcessResult from persisted artifacts — no pipeline run."""
    doc_dir = OUTPUT_DIR / doc_id
    data = json.loads((doc_dir / "chunks.json").read_text())
    return ProcessResult(
        doc_id=doc_id,
        metadata=DocumentMetadata(**json.loads((doc_dir / "metadata.json").read_text())),
        markdown=(doc_dir / "markdown.md").read_text(),
        chunks=data["chunks"],
        stats=ChunkStats(**data["stats"]),
        cached=cached,
        pipeline_version=pipeline_version,
    )


def process_pdf(pdf_path: Path, original_name: str, *, force: bool = False) -> ProcessResult:
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    reg = registry.connect()
    existing = registry.find_by_hash(reg, sha256)
    if (
        existing is not None
        and not force
        and existing["status"] == "processed"
        and existing["pipeline_version"] >= PIPELINE_VERSION
        and (OUTPUT_DIR / existing["doc_id"] / "chunks.json").exists()
    ):
        return load_result(
            existing["doc_id"], cached=True, pipeline_version=existing["pipeline_version"]
        )

    doc_id_for_failure = slugify(original_name)
    try:
        result = _run_pipeline(pdf_path, original_name)
    except Exception as e:
        registry.record_failed(
            reg, doc_id=doc_id_for_failure, source_name=original_name,
            sha256=sha256, pipeline_version=PIPELINE_VERSION, error=str(e),
        )
        raise
    registry.record_processed(
        reg,
        doc_id=result.doc_id,
        source_name=original_name,
        sha256=sha256,
        pipeline_version=PIPELINE_VERSION,
        chunk_count=result.stats.total_chunks,
        total_tokens=result.stats.total_tokens,
    )
    return result


def _run_pipeline(pdf_path: Path, original_name: str) -> ProcessResult:
    markdown = get_parser().parse(pdf_path)
    # Metadata sees the original text (citation tables help it); chunks don't.
    metadata = get_metadata_extractor().extract(markdown, original_name)
    # chunks.json is a studio inspection artifact of the ACTIVE profile's
    # chunking; the indexer re-chunks from markdown.md per profile.
    chunk_result = get_chunker(get_profile().chunking).chunk_document(markdown)
    chunks, tables = chunk_result.chunks, chunk_result.tables

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
        doc_id=doc_id, metadata=metadata, markdown=markdown, chunks=chunks, stats=stats,
        pipeline_version=PIPELINE_VERSION,
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
