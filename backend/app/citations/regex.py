"""Regex CitationExtractor for Indian reporter formats.

Superset of the patterns in rewrite.py (query-side detection) and
metadata/regex.py (header citations): this one scans the WHOLE document,
because cited precedents appear in the reasoning, not the header.

Learning notes:
- Patterns are ordered and matched independently, then deduped by
  normalized ref — so the year-first SCC variant matching a span the
  parenthesised variant already found costs nothing.
- Deterministic vs learned extraction: reporter citations follow rigid
  grammars, so a regex is the right tool (same argument as rewrite.py).
  What regex CANNOT do is resolve "the Kesavananda case" prose references
  — that would be an LLM extractor, a second module + registry entry away.
"""

import re

from .base import Citation, CitationExtractor, normalize_ref

_YEAR = r"(?:18|19|20)\d{2}"

PATTERNS: list[re.Pattern] = [
    # AIR 1973 SC 1461 / AIR 1995 Bom 123
    re.compile(rf"\bAIR\s+{_YEAR}\s+[A-Za-z]+\s+\d+", re.IGNORECASE),
    # (2017) 10 SCC 1  (+ Cri/Civ/L&S sub-series)
    re.compile(
        rf"\(\s*{_YEAR}\s*\)\s*\d+\s*SCC\s*(?:\(\s*(?:Cri|Civ|L&S)\s*\)\s*)?\d+",
        re.IGNORECASE,
    ),
    # 1995 Supp (2) SCC 182
    re.compile(rf"\b{_YEAR}\s+Supp\s*\(\s*\d+\s*\)\s*SCC\s*\d+", re.IGNORECASE),
    # 1995 (2) SCC 7 — year-first variant
    re.compile(rf"\b{_YEAR}\s*\(\s*\d+\s*\)\s*SCC\s*\d+", re.IGNORECASE),
    # 2019 SCC OnLine SC 1234
    re.compile(rf"\b{_YEAR}\s+SCC\s+OnLine\s+[A-Za-z]+\s+\d+", re.IGNORECASE),
    # [1952] SCR 135 / [1952] 2 SCR 135
    re.compile(rf"\[\s*{_YEAR}\s*\]\s*(?:\d+\s*)?SCR\s*\d+", re.IGNORECASE),
    # 1952 SCR 135 / 1952 2 SCR 135
    re.compile(rf"\b{_YEAR}\s+(?:\d+\s+)?SCR\s+\d+", re.IGNORECASE),
    # 2023 INSC 123 — neutral citation
    re.compile(rf"\b{_YEAR}\s+INSC\s+\d+", re.IGNORECASE),
    # (2005) 3 SCALE 134 / 2005 (3) SCALE 134
    re.compile(rf"\(\s*{_YEAR}\s*\)\s*\d+\s*SCALE\s*\d+", re.IGNORECASE),
    re.compile(rf"\b{_YEAR}\s*\(\s*\d+\s*\)\s*SCALE\s*\d+", re.IGNORECASE),
]


class RegexCitationExtractor(CitationExtractor):
    def extract(self, markdown: str) -> list[Citation]:
        found: dict[str, Citation] = {}
        for pattern in PATTERNS:
            for match in pattern.finditer(markdown):
                ref = normalize_ref(match.group())
                if not ref:
                    continue
                existing = found.get(ref)
                if existing is None:
                    found[ref] = Citation(
                        ref=ref,
                        raw=" ".join(match.group().split()),
                        occurrences=1,
                        first_offset=match.start(),
                    )
                else:
                    existing.occurrences += 1
                    if match.start() < existing.first_offset:
                        existing.first_offset = match.start()
        return sorted(found.values(), key=lambda c: c.first_offset)
