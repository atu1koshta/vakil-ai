"""Semantic chunker for Indian court judgments.

Strategy (see PLAN.md):
1. Split Markdown into sections by headings; classify against canonical
   judgment sections (Facts / Issues / Arguments / Analysis / Judgment).
2. Sections over MAX_TOKENS are sub-split at paragraph boundaries,
   targeting TARGET_TOKENS per chunk with ~OVERLAP_TOKENS of overlap.
3. Every chunk is prefixed with its section title so it stays
   self-describing after retrieval.
"""

import re

import tiktoken

from .models import Chunk

TARGET_TOKENS = 700
MAX_TOKENS = 900
MIN_TOKENS = 120  # merge tail chunks smaller than this into the previous chunk
OVERLAP_TOKENS = 80

_enc = tiktoken.get_encoding("cl100k_base")

# Canonical judgment sections; matched against heading text.
SECTION_KEYWORDS = {
    "Facts": ["fact", "background", "brief facts", "factual matrix"],
    "Issues": ["issue", "question of law", "points for determination", "questions"],
    "Arguments": ["argument", "submission", "contention", "counsel"],
    "Analysis": ["analysis", "discussion", "consideration", "reasoning", "findings"],
    "Judgment": ["judgment", "conclusion", "order", "held", "decision", "disposition", "relief"],
}

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
# Fallback for documents where Docling found no headings: short ALL-CAPS lines.
CAPS_HEADING_RE = re.compile(r"^([A-Z][A-Z\s&/,'-]{3,60})$", re.MULTILINE)
# Numbered legal paragraphs ("12. The next contention...") — authored semantic
# units in judgments, preferred over blank-line paragraphs as chunk boundaries.
NUMBERED_PARA_RE = re.compile(r"(?m)^(?=\d{1,3}\.\s)")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def is_valid_heading(title: str) -> bool:
    """Reject lines Docling promoted to headings that are really footnotes,
    quoted passages, or hyphenation fragments (common in scanned judgments)."""
    t = title.strip()
    if not t or len(t) > 80:
        return False
    if t[0].islower():  # mid-sentence fragment, e.g. "tial to the community, or"
        return False
    if t.endswith("-"):  # hyphenated mid-word break
        return False
    if re.match(r"""^["'“‘(\[]*\d""", t):  # footnote/citation: "(1) [1917] A.C. 260"
        return False
    if re.search(r",\s*(?:or|and)?$", t, re.IGNORECASE):  # trailing comma / ", or"
        return False
    if re.search(r"\b(?:thus|follows)\s*:$", t, re.IGNORECASE):  # quote introducer
        return False
    return True


def extract_tables(markdown: str) -> tuple[str, list[str]]:
    """Pull markdown pipe-tables (citation reference tables etc.) out of the
    text. They have no retrieval value as prose and their lack of sentence
    punctuation breaks size-splitting."""
    out_lines: list[str] = []
    tables: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if len(current) >= 3:
            tables.append("\n".join(current))
        else:  # too short to be a real table, keep as text
            out_lines.extend(current)
        current.clear()

    for line in markdown.split("\n"):
        if line.lstrip().startswith("|"):
            current.append(line)
        else:
            flush()
            out_lines.append(line)
    flush()
    return "\n".join(out_lines), tables


def classify_section(title: str) -> str:
    lowered = title.lower()
    for canonical, keywords in SECTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return canonical
    return title.strip()[:60] or "Document"


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return [(section_title, section_text)] preserving document order.
    Invalid headings are demoted: their text stays inline in the body."""
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        matches = list(CAPS_HEADING_RE.finditer(markdown))
    matches = [
        m
        for m in matches
        if is_valid_heading(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
    ]

    if not matches:
        body = _demote_headings(markdown).strip()
        return [("Document", body)] if body else []

    sections: list[tuple[str, str]] = []
    preamble = _demote_headings(markdown[: matches[0].start()]).strip()
    if preamble:
        sections.append(("Header", preamble))

    for i, match in enumerate(matches):
        title = match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = _demote_headings(markdown[match.end():end]).strip()
        if body:
            sections.append((classify_section(title.strip()), body))
    return sections


def _demote_headings(text: str) -> str:
    """Strip markdown heading markers from rejected headings left in body text."""
    return re.sub(r"(?m)^#{1,4}\s+", "", text)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_units(text: str) -> list[str]:
    """Numbered legal paragraphs when present (>=3), else blank-line paragraphs."""
    numbered = [u.strip() for u in NUMBERED_PARA_RE.split(text) if u.strip()]
    if len(numbered) >= 3:
        return numbered
    return _split_paragraphs(text)


def _hard_token_split(text: str, limit: int) -> list[str]:
    """Last resort: split on raw token windows so MAX_TOKENS is a guarantee
    (tables/lists without sentence punctuation defeat the sentence splitter)."""
    tokens = _enc.encode(text)
    return [
        _enc.decode(tokens[i : i + limit]).strip()
        for i in range(0, len(tokens), limit)
    ]


def _split_oversized_paragraph(paragraph: str, limit: int) -> list[str]:
    """Sentence-level split for a single paragraph that alone exceeds limit."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, current, current_tokens = [], [], 0
    for sentence in sentences:
        tokens = count_tokens(sentence)
        if current and current_tokens + tokens > limit:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        parts.append(" ".join(current))
    return [
        piece
        for part in parts
        for piece in (
            _hard_token_split(part, limit) if count_tokens(part) > MAX_TOKENS else [part]
        )
    ]


def _sub_split(section_text: str) -> list[str]:
    """Greedy unit packing to TARGET_TOKENS with unit-level overlap. Units are
    numbered legal paragraphs where the document has them, else paragraphs;
    oversized units degrade to sub-paragraphs, sentences, then raw tokens."""
    paragraphs: list[str] = []
    for unit in _split_units(section_text):
        if count_tokens(unit) <= MAX_TOKENS:
            paragraphs.append(unit)
            continue
        # Numbered unit can span several text paragraphs — try those first.
        for paragraph in _split_paragraphs(unit):
            if count_tokens(paragraph) > MAX_TOKENS:
                paragraphs.extend(_split_oversized_paragraph(paragraph, TARGET_TOKENS))
            else:
                paragraphs.append(paragraph)

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        tokens = count_tokens(paragraph)
        if current and current_tokens + tokens > TARGET_TOKENS:
            pieces.append("\n\n".join(current))
            # Overlap: carry trailing paragraphs up to OVERLAP_TOKENS forward.
            overlap: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current):
                prev_tokens = count_tokens(prev)
                if overlap_tokens + prev_tokens > OVERLAP_TOKENS:
                    break
                overlap.insert(0, prev)
                overlap_tokens += prev_tokens
            current = overlap[:]
            current_tokens = overlap_tokens
        current.append(paragraph)
        current_tokens += tokens

    if current:
        piece = "\n\n".join(current)
        # Avoid a tiny tail chunk that is mostly overlap of the previous one.
        if pieces and count_tokens(piece) < MIN_TOKENS:
            pieces[-1] = pieces[-1] + "\n\n" + current[-1]
        else:
            pieces.append(piece)
    # Joins ("\n\n") add tokens beyond the per-unit sums the packer tracked;
    # re-measure so MAX_TOKENS is a hard guarantee on the final text.
    return [
        part
        for piece in pieces
        for part in (
            _hard_token_split(piece, TARGET_TOKENS)
            if count_tokens(piece) > MAX_TOKENS
            else [piece]
        )
    ]


def chunk_markdown(markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_title, section_text in split_sections(markdown):
        if count_tokens(section_text) <= MAX_TOKENS:
            pieces = [section_text]
        else:
            pieces = _sub_split(section_text)
        for piece in pieces:
            text = f"## {section_title}\n\n{piece}"
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"chunk_{index:03d}",
                    index=index,
                    section=section_title,
                    text=text,
                    token_count=count_tokens(text),
                    char_count=len(text),
                )
            )
    return chunks
