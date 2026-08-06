"""MetadataExtractor component package.

base.py = interface, regex.py = heuristic implementation, __init__.py =
registry + factory. An LLM-based extractor would slot in as a second module
and registry entry, selected via `components.metadata_extractor` in
config.yaml.
"""

from ..config import ConfigError, get_config
from .base import MetadataExtractor
from .regex import RegexMetadataExtractor

_EXTRACTORS: dict[str, type[MetadataExtractor]] = {"regex": RegexMetadataExtractor}


def get_metadata_extractor(name: str | None = None) -> MetadataExtractor:
    name = name or get_config().components.metadata_extractor
    if name not in _EXTRACTORS:
        raise ConfigError(
            f"unknown metadata_extractor '{name}'; available: {sorted(_EXTRACTORS)}"
        )
    return _EXTRACTORS[name]()


__all__ = ["MetadataExtractor", "RegexMetadataExtractor", "get_metadata_extractor"]
