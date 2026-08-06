"""Chunker component package.

base.py = interface, section_aware.py = the judgment-aware implementation,
__init__.py = registry + factory. A new strategy (fixed-window,
sentence-window...) = new module implementing Chunker + one _CHUNKERS entry,
selected via `chunking.strategy` in a profile. Strategy is part of the
profile fingerprint, so switching it isolates into its own index.

Convention: implementations are constructed with the profile's
ChunkingConfig.
"""

from ..config import ChunkingConfig, ConfigError, get_profile
from .base import Chunker
from .section_aware import SectionAwareChunker

_CHUNKERS: dict[str, type[Chunker]] = {"section-aware": SectionAwareChunker}


def get_chunker(cfg: ChunkingConfig | None = None) -> Chunker:
    cfg = cfg or get_profile().chunking
    if cfg.strategy not in _CHUNKERS:
        raise ConfigError(
            f"unknown chunking strategy '{cfg.strategy}'; available: {sorted(_CHUNKERS)}"
        )
    return _CHUNKERS[cfg.strategy](cfg)


__all__ = ["Chunker", "SectionAwareChunker", "get_chunker"]
