"""extract_and_store: one function both write paths share.

Called by pipeline.py for every newly processed doc AND by backfill.py for
docs already on disk — the two entry points, one implementation, so edges
can never drift depending on how a doc got extracted.

Learning notes:
- A doc's OWN citations (how the judgment itself is reported, printed in
  the header block) must not become self-edges — "this case cites itself"
  is noise. Heuristic: refs first seen within OWN_CITE_HEAD_CHARS are
  treated as the doc's own keys, stored in doc_citation_keys (the
  reverse-lookup table) and EXCLUDED from citation_edges. A real cited
  precedent appearing that early is rare; a wrong exclusion costs one edge,
  a wrong inclusion pollutes every get_citing answer.
- citations.json is the studio inspection artifact (like chunks.json);
  vakil.db rows are what tools query. Both written here, atomically enough
  for a single-writer pipeline.
"""

import json
import sqlite3
from pathlib import Path

from .. import registry
from . import get_citation_extractor
from .base import Citation

OWN_CITE_HEAD_CHARS = 2500


def extract_and_store(
    conn: sqlite3.Connection, *, doc_id: str, markdown: str, doc_dir: Path
) -> tuple[list[Citation], list[Citation]]:
    """Extract citations from markdown, persist citations.json + registry
    rows. Returns (own_citations, cited_edges)."""
    citations = get_citation_extractor().extract(markdown)
    own = [c for c in citations if c.first_offset < OWN_CITE_HEAD_CHARS]
    cited = [c for c in citations if c.first_offset >= OWN_CITE_HEAD_CHARS]

    (doc_dir / "citations.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "own": [c.model_dump() for c in own],
                "cited": [c.model_dump() for c in cited],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    registry.replace_doc_citation_keys(conn, doc_id, [(c.ref, c.raw) for c in own])
    registry.replace_citation_edges(
        conn, doc_id, [(c.ref, c.raw, c.occurrences) for c in cited]
    )
    return own, cited
