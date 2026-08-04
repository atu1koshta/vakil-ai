"""Download the selected judgments and extract plain text for eval-question
authoring. Deliberately avoids the real pipeline (docling/chunker/embedder) —
question writing only needs readable text, and gold passages are stored as
raw text spans scored later by fuzzy overlap against whatever chunks exist.

Usage:
    python -m app.evals.fetch_texts

Writes evals/pdfs/<slug>.pdf and evals/text/<slug>.txt. Resumable: existing
non-empty .txt files are skipped.
"""

import json
import shutil
from pathlib import Path

import pypdfium2 as pdfium

from ..connectors import drive
from ..pipeline import slugify

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals"
PDF_DIR = EVALS_DIR / "pdfs"
TEXT_DIR = EVALS_DIR / "text"


def extract_text(pdf_path: Path) -> str:
    doc = pdfium.PdfDocument(pdf_path)
    try:
        pages = []
        for page in doc:
            textpage = page.get_textpage()
            pages.append(textpage.get_text_bounded())
            textpage.close()
            page.close()
        return "\n\n".join(pages)
    finally:
        doc.close()


def main() -> None:
    selection = json.loads((EVALS_DIR / "selection.json").read_text())
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    empty = []
    for i, entry in enumerate(selection, 1):
        slug = slugify(entry["name"])
        txt_path = TEXT_DIR / f"{slug}.txt"
        if txt_path.exists() and txt_path.stat().st_size > 500:
            print(f"[{i}/{len(selection)}] {slug} (cached)", flush=True)
            continue
        tmp_path, _ = drive.download_pdf(entry["file_id"])
        try:
            pdf_path = PDF_DIR / f"{slug}.pdf"
            shutil.copyfile(tmp_path, pdf_path)
            text = extract_text(pdf_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        txt_path.write_text(text, encoding="utf-8")
        if len(text.strip()) < 500:
            empty.append(slug)
        print(f"[{i}/{len(selection)}] {slug}: {len(text)} chars", flush=True)

    if empty:
        print("\nWARNING — little/no text layer (scanned?), need OCR or swap doc:")
        for slug in empty:
            print(f"  {slug}")


if __name__ == "__main__":
    main()
