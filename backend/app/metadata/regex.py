"""Regex/heuristic MetadataExtractor.

Works on the first portion of the document where Indian judgments carry
their header block: court name, case title (X vs Y), date, coram/judges,
and neutral/reporter citations.
"""

import re

from ..models import DocumentMetadata
from .base import MetadataExtractor

# How much of the document head to scan for header fields.
HEAD_CHARS = 6000

COURT_PATTERNS = [
    r"SUPREME COURT OF INDIA",
    r"HIGH COURT OF JUDICATURE AT \w[\w\s]*",
    r"HIGH COURT OF [\w\s]+?(?=\n|,|\.)",
    r"[\w\s]+ HIGH COURT",
    r"NATIONAL COMPANY LAW APPELLATE TRIBUNAL",
    r"NATIONAL COMPANY LAW TRIBUNAL",
    r"DISTRICT COURT[\w\s,]*",
]

CITATION_PATTERNS = [
    r"AIR\s+\d{4}\s+(?:SC|SCC|[A-Z][a-z]+)\s+\d+",
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
    r"\d{4}\s+SCC\s+OnLine\s+[A-Za-z]+\s+\d+",
    r"\d{4}\s+INSC\s+\d+",
    r"\[\d{4}\]\s+\d+\s+SCR\s+\d+",
]

DATE_PATTERNS = [
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)[,]?\s+\d{4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}[,]?\s+\d{4}\b",
]

VS_PATTERN = re.compile(
    r"^(?P<a>[^\n]{3,120}?)\s+(?:v(?:s)?\.?|versus)\s+(?P<b>[^\n]{3,120})$",
    re.IGNORECASE | re.MULTILINE,
)

JUDGE_LINE_PATTERN = re.compile(
    r"(?:CORAM|BEFORE|BENCH)\s*[:\-]\s*(?P<names>[^\n]+)", re.IGNORECASE
)
JUSTICE_PATTERN = re.compile(
    r"(?:HON'?BLE\s+)?(?:MR\.?|MRS\.?|MS\.?|DR\.?)?\s*JUSTICE\s+([A-Z][A-Za-z.\s]+?)(?=,|\n|$|\s+AND\b)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,:;-*#")


def _extract_case_title(head: str) -> str | None:
    match = VS_PATTERN.search(head)
    if not match:
        return None
    a, b = _clean(match.group("a")), _clean(match.group("b"))
    # Strip party-role suffixes commonly placed on the same line.
    b = re.sub(
        r"\s*(?:\.{2,}|…)?\s*(?:Appellants?|Respondents?|Petitioners?|Defendants?)\s*$",
        "",
        b,
        flags=re.IGNORECASE,
    ).strip()
    if not a or not b:
        return None
    return f"{a} vs {b}"


def _extract_judges(head: str) -> list[str]:
    judges: list[str] = []
    coram = JUDGE_LINE_PATTERN.search(head)
    if coram:
        for part in re.split(r",|\bAND\b|&", coram.group("names"), flags=re.IGNORECASE):
            name = _clean(re.sub(r"\bJJ?\.?$", "", _clean(part)))
            name = _clean(
                re.sub(
                    r"^(?:HON'?BLE\s+)?(?:THE\s+)?(?:MR\.?|MRS\.?|MS\.?|DR\.?)?\s*(?:CHIEF\s+)?JUSTICE\s+",
                    "",
                    name,
                    flags=re.IGNORECASE,
                )
            )
            if len(name) > 3 and name not in judges:
                judges.append(name)
    for match in JUSTICE_PATTERN.finditer(head):
        name = _clean(match.group(1))
        if len(name) > 3 and name not in judges:
            judges.append(name)
    return judges[:10]


class RegexMetadataExtractor(MetadataExtractor):
    def extract(self, markdown: str, source_file: str) -> DocumentMetadata:
        head = markdown[:HEAD_CHARS]

        court = None
        for pattern in COURT_PATTERNS:
            match = re.search(pattern, head, re.IGNORECASE)
            if match:
                court = _clean(match.group(0)).title()
                break

        date = None
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, head, re.IGNORECASE)
            if match:
                date = _clean(match.group(0))
                break

        citations: list[str] = []
        for pattern in CITATION_PATTERNS:
            for match in re.finditer(pattern, markdown):
                value = _clean(match.group(0))
                if value not in citations:
                    citations.append(value)

        return DocumentMetadata(
            case_title=_extract_case_title(head),
            court=court,
            date=date,
            judges=_extract_judges(head),
            citations=citations[:20],
            source_file=source_file,
        )
