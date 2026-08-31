"""The agent's tools: schemas (OpenAI function format) + executors.

Plain module, not a base/impl/registry package: the toolset is fixed and
there is no second plausible backend (same reasoning as rewrite.py). The
LOOP is the thing with multiple backends (hand-rolled now, LangGraph in
3b) — that gets the package pattern, tools are shared data + functions.

Learning notes:
- Tools return plain STRINGS, not JSON: prose with provenance headers is
  what 8B-scale models cite reliably; nested JSON invites the model to
  echo structure instead of reasoning over content.
- Every result is CAPPED. The loop runs on num_ctx=8192 local models and
  Ollama truncates the context front SILENTLY on overflow (see
  llm/ollama.py) — an uncapped read_document would evict the system
  prompt. Budget: system+question ~700 tokens; three worst-case tool
  results ~5.5k tokens; the rest is assistant turns + answer headroom.
- User-argument problems return "ERROR: ..." strings — fed back to the
  model as tool output so it can self-correct. Infra failures
  (EmbeddingError, IndexConfigMismatch, ConfigError) propagate: those are
  operator problems (503/400), not model feedback.
- filter_documents matches court/title by SUBSTRING, case-insensitive:
  the regex metadata extractor produces noisy court strings ("High Court
  Of Bombay Under S"), so exact match would mostly miss. The tool
  description tells the model to pass short values ("Bombay").
"""

import json
import re

from ..chunker.section_aware import split_sections
from ..context import count_tokens, format_chunk
from ..pipeline import OUTPUT_DIR
from ..retrieval import retrieve

SEARCH_CAP_CHARS = 6_000
SEARCH_ROW_CHARS = 600
SEARCH_MAX_K = 12
FILTER_MAX_ROWS = 25
FILTER_TITLE_CHARS = 80
READ_CAP_CHARS = 10_000

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "Semantic + keyword search over all judgment chunks. Returns "
                "passages with [doc_id:chunk_id | SECTION] headers — cite "
                "those ids. Start here for most questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (a question or key phrase).",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many passages to return (default 8, max 12).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_documents",
            "description": (
                "List judgments by metadata. court and title_contains match "
                "as case-insensitive SUBSTRINGS — pass short values like "
                "'Bombay' or 'Supreme', not full court names. Use for "
                "'which cases...' / court / year questions. No arguments = "
                "list all documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "court": {
                        "type": "string",
                        "description": "Substring of the court name, e.g. 'Bombay'.",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Earliest judgment year, inclusive.",
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "Latest judgment year, inclusive.",
                    },
                    "title_contains": {
                        "type": "string",
                        "description": "Substring of the case title.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cited",
            "description": (
                "List the earlier cases a judgment cites (its precedent "
                "basis), from the citation graph. Cited cases that exist in "
                "the corpus include their doc_id — follow up with "
                "read_document or get_cited on those to walk the chain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "A doc_id seen in earlier tool results.",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_citing",
            "description": (
                "Reverse citation lookup: which judgments in the corpus cite "
                "a given case. Pass either a doc_id from earlier results or "
                "a reporter citation string like 'AIR 1973 SC 1461' or "
                "'(2017) 10 SCC 1'. Use for 'which cases followed/applied "
                "X?' questions — semantic search cannot enumerate these."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": (
                            "doc_id OR reporter citation, e.g. 'AIR 1973 SC 1461'."
                        ),
                    },
                },
                "required": ["reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read a full section of one judgment — use when a search "
                "chunk cuts off mid-reasoning. Call with doc_id only to list "
                "the document's sections, then call again with one section "
                "name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "A doc_id seen in earlier tool results.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Section name from the section list, e.g. 'Judgment'.",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
]


def run_tool(name: str, args: dict, *, profile: str | None = None) -> str:
    """Dispatch one validated tool call. `profile` is bound by the caller
    (endpoint/CLI), never exposed to the model."""
    # Models send numbers as strings ("k": "12") — coerce, never crash the loop.
    def _int(value, fallback=None):
        try:
            return fallback if value is None else int(value)
        except (TypeError, ValueError):
            return fallback

    if name == "search_chunks":
        return _search_chunks(
            str(args.get("query") or ""), _int(args.get("k"), 8), profile
        )
    if name == "filter_documents":
        return _filter_documents(
            args.get("court"),
            _int(args.get("year_from")),
            _int(args.get("year_to")),
            args.get("title_contains"),
        )
    if name == "read_document":
        return _read_document(str(args.get("doc_id") or ""), args.get("section"))
    if name == "get_cited":
        return _get_cited(str(args.get("doc_id") or ""))
    if name == "get_citing":
        return _get_citing(str(args.get("reference") or ""))
    return (
        f"ERROR: unknown tool '{name}'. Available: search_chunks, "
        "filter_documents, read_document, get_cited, get_citing."
    )


def _search_chunks(query: str, k: int, profile: str | None) -> str:
    if not query.strip():
        return "ERROR: search_chunks needs a non-empty 'query'."
    rows = retrieve(query, k=max(1, min(k, SEARCH_MAX_K)), profile=profile)
    if not rows:
        return "No passages found. Try different phrasing or filter_documents."
    blocks = []
    total = 0
    for row in rows:
        trimmed = dict(row)
        if len(trimmed["text"]) > SEARCH_ROW_CHARS:
            trimmed["text"] = trimmed["text"][:SEARCH_ROW_CHARS] + "…"
        block = format_chunk(trimmed)
        if total + len(block) > SEARCH_CAP_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


_YEAR_RE = re.compile(r"\b(?:1[89]|20)\d{2}\b")


def _year(date_str: str) -> int | None:
    m = _YEAR_RE.search(date_str or "")
    return int(m.group()) if m else None


def _filter_documents(
    court, year_from, year_to, title_contains
) -> str:
    from .. import registry  # local import: keeps module import cheap

    conn = registry.connect()
    try:
        docs = [r for r in registry.list_documents(conn) if r["status"] == "processed"]
    finally:
        conn.close()

    matched = []
    for row in docs:
        meta_path = OUTPUT_DIR / row["doc_id"] / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if court and (court or "").lower() not in (meta.get("court") or "").lower():
            continue
        if title_contains and title_contains.lower() not in (
            meta.get("case_title") or ""
        ).lower():
            continue
        year = _year(meta.get("date") or "")
        if (year_from or year_to) and year is None:
            continue  # can't prove the year matches — exclude from year filters
        if year_from and year is not None and year < int(year_from):
            continue
        if year_to and year is not None and year > int(year_to):
            continue
        matched.append((row["doc_id"], meta))

    if not matched:
        return (
            "No documents matched. court/title match as substrings — try a "
            "shorter value, or call with no arguments to list everything."
        )
    lines = []
    for doc_id, meta in matched[:FILTER_MAX_ROWS]:
        title = (meta.get("case_title") or "?")[:FILTER_TITLE_CHARS]
        lines.append(
            f"{doc_id} — {title} ({meta.get('court') or 'court unknown'}, "
            f"{meta.get('date') or 'date unknown'})"
        )
    if len(matched) > FILTER_MAX_ROWS:
        lines.append(f"...and {len(matched) - FILTER_MAX_ROWS} more")
    return "\n".join(lines)


CITE_MAX_ROWS = 25


def _case_title(doc_id: str) -> str:
    meta_path = OUTPUT_DIR / doc_id / "metadata.json"
    if not meta_path.exists():
        return "?"
    return (json.loads(meta_path.read_text()).get("case_title") or "?")[
        :FILTER_TITLE_CHARS
    ]


def _get_cited(doc_id: str) -> str:
    from .. import registry  # local import: keeps module import cheap

    if not doc_id or not (OUTPUT_DIR / doc_id).exists():
        return (
            f"ERROR: no document '{doc_id}'. Use filter_documents or "
            "search_chunks to find valid doc_ids."
        )
    conn = registry.connect()
    try:
        rows = registry.edges_cited_by(conn, doc_id)
    finally:
        conn.close()
    if not rows:
        return (
            f"No citation edges recorded for {doc_id}. Either it cites no "
            "reported cases or edges were not extracted."
        )
    lines = [f"Cases cited by {doc_id} ({len(rows)} refs, most-cited first):"]
    for row in rows[:CITE_MAX_ROWS]:
        where = (
            f"IN CORPUS: {row['resolved_doc_id']} — {_case_title(row['resolved_doc_id'])}"
            if row["resolved_doc_id"]
            else "not in corpus"
        )
        lines.append(
            f"- {row['raw_text']} (cited {row['occurrences']}x) — {where}"
        )
    if len(rows) > CITE_MAX_ROWS:
        lines.append(f"...and {len(rows) - CITE_MAX_ROWS} more")
    return "\n".join(lines)


def _get_citing(reference: str) -> str:
    from .. import registry  # local import: keeps module import cheap
    from ..citations import normalize_ref

    if not reference.strip():
        return "ERROR: get_citing needs a 'reference' (doc_id or citation string)."
    conn = registry.connect()
    try:
        # doc_id? -> that doc's own reporter citations are the lookup keys.
        if (OUTPUT_DIR / reference).exists():
            refs = registry.citation_keys_for_doc(conn, reference)
            label = f"{reference} — {_case_title(reference)}"
            if not refs:
                return (
                    f"{reference} has no recorded reporter citation of its own, "
                    "so reverse lookup by doc_id is not possible. Try a "
                    "citation string like 'AIR 1973 SC 1461' if you have one."
                )
        else:
            refs = [normalize_ref(reference)]
            label = reference
        # Dedup by citing doc (a doc may cite the target under two of its
        # refs); rows arrive occurrences-DESC, first one wins.
        seen: dict[str, object] = {}
        for r in registry.docs_citing(conn, refs):
            if r["citing_doc_id"] != reference and r["citing_doc_id"] not in seen:
                seen[r["citing_doc_id"]] = r
        rows = list(seen.values())
    finally:
        conn.close()
    if not rows:
        return (
            f"No judgments in the corpus cite {label}. (The graph only "
            "covers indexed documents — absence here is not absence in law.)"
        )
    lines = [f"Judgments citing {label} ({len(rows)}):"]
    for row in rows[:CITE_MAX_ROWS]:
        lines.append(
            f"- {row['citing_doc_id']} — {_case_title(row['citing_doc_id'])} "
            f"(cites it {row['occurrences']}x as {row['raw_text']})"
        )
    if len(rows) > CITE_MAX_ROWS:
        lines.append(f"...and {len(rows) - CITE_MAX_ROWS} more")
    return "\n".join(lines)


def _read_document(doc_id: str, section) -> str:
    md_path = OUTPUT_DIR / doc_id / "markdown.md"
    if not doc_id or not md_path.exists():
        return (
            f"ERROR: no document '{doc_id}'. Use filter_documents or "
            "search_chunks to find valid doc_ids."
        )
    sections = split_sections(md_path.read_text())

    def section_list() -> str:
        seen: dict[str, int] = {}
        for title, text in sections:
            seen[title] = seen.get(title, 0) + count_tokens(text)
        listing = ", ".join(f"{t} (~{n} tokens)" for t, n in seen.items())
        return f"Sections in {doc_id} (call again with one): {listing}"

    if not section:
        return section_list()
    wanted = str(section).strip().lower()
    parts = [text for title, text in sections if title.lower() == wanted]
    if not parts:
        return f"ERROR: no section '{section}'. {section_list()}"
    label = next(t for t, _ in sections if t.lower() == wanted)
    body = "\n\n".join(parts)
    if len(body) > READ_CAP_CHARS:
        body = (
            body[:READ_CAP_CHARS]
            + f"\n[TRUNCATED — {len(body) - READ_CAP_CHARS} more chars]"
        )
    return f"[{doc_id} | {label}]\n\n{body}"
