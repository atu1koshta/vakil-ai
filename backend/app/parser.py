"""Docling wrapper: PDF -> structured Markdown."""

from pathlib import Path

from docling.document_converter import DocumentConverter

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    # Lazy singleton: Docling loads layout/table models on first use (~seconds),
    # keep one instance for the process lifetime.
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def parse_pdf(path: Path) -> str:
    """Convert a PDF to Markdown, preserving heading structure where possible."""
    result = _get_converter().convert(str(path))
    return result.document.export_to_markdown()
