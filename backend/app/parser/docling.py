"""Docling-backed Parser."""

from pathlib import Path

from .base import Parser


class DoclingParser(Parser):
    def __init__(self) -> None:
        # Lazy: Docling loads layout/table models on first use (~seconds);
        # keep one converter for the parser's lifetime.
        self._converter = None

    def parse(self, pdf_path: Path) -> str:
        """Convert a PDF to Markdown, preserving heading structure where possible."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        result = self._converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
