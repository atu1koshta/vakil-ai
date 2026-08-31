"""CitationExtractor component package.

base.py = interface + normalization, regex.py = reporter-grammar
implementation, __init__.py = registry + factory. An LLM extractor that
resolves prose references ("the Kesavananda case") would slot in as a
second module and registry entry, selected via
`components.citation_extractor` in config.yaml.
"""

from ..config import ConfigError, get_config
from .base import Citation, CitationExtractor, normalize_ref
from .regex import RegexCitationExtractor

_EXTRACTORS: dict[str, type[CitationExtractor]] = {"regex": RegexCitationExtractor}


def get_citation_extractor(name: str | None = None) -> CitationExtractor:
    name = name or get_config().components.citation_extractor
    if name not in _EXTRACTORS:
        raise ConfigError(
            f"unknown citation_extractor '{name}'; available: {sorted(_EXTRACTORS)}"
        )
    return _EXTRACTORS[name]()


__all__ = [
    "Citation",
    "CitationExtractor",
    "RegexCitationExtractor",
    "get_citation_extractor",
    "normalize_ref",
]
