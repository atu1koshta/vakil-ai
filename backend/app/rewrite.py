"""Query rewriting — turning the user's question into better search queries.

Plain module, not a component package: citation detection is one regex list
and (2c commit 4) LLM rewrite is one prompt — two functions with no second
implementation on the horizon. The package pattern (base/impl/registry) is
for things with multiple plausible backends; here it would be ceremony.

Learning notes:
- Deterministic vs learned rewriting: legal citations follow rigid reporter
  grammars, and the right tool for a grammar is a regex — same input, same
  output, testable, zero latency, no model to be wrong. Conversational
  dilution has no grammar, so that cure (llm_rewrite) needs a model.
- Ensemble, never replace: everything this module produces becomes an
  ADDITIONAL ranked list in RRF. The raw question always runs untouched, so
  the worst case of any rewrite is graceful degradation to the old behavior
  — a guarantee that comes from structure, not from hoping outputs are good.
- Phrase normalization must mirror the FTS5 tokenizer: a phrase query only
  matches if its tokens equal what unicode61 produced at index time. The
  index stored "(1995)" as the token `1995`, so detected spans are split on
  the same [A-Za-z0-9]+ rule the sanitizer uses — parentheses and brackets
  must vanish on BOTH sides or the phrase silently matches nothing.
"""

import re

from .llm import get_chat_model

# Indian legal reporter formats (extensible list). Each pattern matches one
# citation span in running text; the span is then normalized to bare tokens.
#   AIR 1995 SC 123          — All India Reporter
#   (1995) 2 SCC 7           — Supreme Court Cases (+ Cri/Civ/L&S/Supp)
#   1995 Supp (2) SCC 182    — SCC supplementary volumes
#   1995 (2) SCC 7           — year-first SCC variant
#   [1952] SCR 135 / 1952 SCR 135 / [1952] 2 SCR 135
_YEAR = r"(?:19|20)\d{2}"
_CITATION_PATTERNS = [
    re.compile(rf"\bAIR\s+{_YEAR}\s+[A-Za-z]+\s+\d+", re.IGNORECASE),
    re.compile(
        rf"\(\s*{_YEAR}\s*\)\s*\d+\s*SCC\s*(?:\(\s*(?:Cri|Civ|L&S)\s*\)\s*)?\d+",
        re.IGNORECASE,
    ),
    re.compile(rf"\b{_YEAR}\s+Supp\s*\(\s*\d+\s*\)\s*SCC\s*\d+", re.IGNORECASE),
    re.compile(rf"\b{_YEAR}\s*\(\s*\d+\s*\)\s*SCC\s*\d+", re.IGNORECASE),
    re.compile(rf"\[\s*{_YEAR}\s*\]\s*(?:\d+\s*)?SCR\s*\d+", re.IGNORECASE),
    re.compile(rf"\b{_YEAR}\s+(?:\d+\s+)?SCR\s+\d+", re.IGNORECASE),
]


def detect_citations(question: str) -> list[str]:
    """Citation spans in the question, normalized to FTS5-phrase form.

    Returns e.g. ["1995 2 SCC 7"] for "...reported in (1995) 2 SCC 7?" —
    bare alphanumeric tokens, space-joined, deduped in match order. Empty
    list when the question carries no recognizable citation.
    """
    phrases: list[str] = []
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(question):
            normalized = " ".join(re.findall(r"[A-Za-z0-9]+", match.group()))
            if normalized and normalized not in phrases:
                phrases.append(normalized)
    return phrases


_REWRITE_SYSTEM = (
    "You rewrite questions about Indian court judgments into short keyword "
    "search queries. Keep case names, citations, statute names, section "
    "numbers, and legal doctrines exactly as written. Drop conversational "
    "words that carry no search signal. Output ONLY the rewritten query, "
    "one line, no explanation, no quotes."
)

_MAX_REWRITE_CHARS = 200


def llm_rewrite(question: str, model: str | None = None) -> str | None:
    """Keyword rewrite of the question via the configured chat model.

    Returns None on ANY failure — bad output, timeout, model down. The rule
    behind all the Nones: retrieval must never break because an OPTIONAL
    enhancement had a bad day; a failed rewrite just means the query
    ensemble is one list smaller, which is exactly yesterday's behavior.
    """
    try:
        raw = get_chat_model(model).chat(_REWRITE_SYSTEM, question)
    except Exception:  # GenerationError, ConfigError, network — all -> None
        return None
    rewritten = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if (
        not rewritten
        or len(rewritten) > _MAX_REWRITE_CHARS
        or rewritten.lower() == question.strip().lower()
    ):
        return None
    return rewritten
