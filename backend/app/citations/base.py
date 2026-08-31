"""CitationExtractor interface: parsed Markdown -> citation references.

Pipeline-level component (edges are derived from shared parse output, not
from any profile's chunks). Deliberately NOT tied to PIPELINE_VERSION:
extraction reads markdown.md that is already on disk, so improving the
extractor re-runs via `python -m app.citations.backfill` in seconds instead
of re-running Docling on the whole corpus.

Learning notes:
- The normalized `ref` is the graph's join key. "(2017) 10 SCC 1",
  "(2017)10 SCC 1" and "2017 (10) SCC 1" must all collapse to the SAME key
  or the graph silently fragments — same lesson as rewrite.py's FTS5 phrase
  normalization: punctuation must vanish identically on both sides of a
  join. Normalization here mirrors that rule ([A-Za-z0-9]+ tokens), plus
  uppercasing because reporter abbreviations are case-insensitive in the
  wild ("OnLine" vs "ONLINE").
"""

import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One distinct citation reference found in a document."""

    ref: str  # normalized key, e.g. "AIR 1973 SC 1461" — the graph join key
    raw: str  # first raw span as it appeared in the text
    occurrences: int = Field(default=1, ge=1)
    first_offset: int = Field(default=0, ge=0)  # char offset of first match


def normalize_ref(span: str) -> str:
    """Citation span -> canonical join key: bare alphanumeric tokens,
    space-joined, uppercased. Same tokenization rule as rewrite.py."""
    return " ".join(re.findall(r"[A-Za-z0-9]+", span)).upper()


class CitationExtractor(ABC):
    @abstractmethod
    def extract(self, markdown: str) -> list[Citation]:
        """All distinct citation references in the text, in first-appearance
        order, deduped by normalized ref."""
        ...
