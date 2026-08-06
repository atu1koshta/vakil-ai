"""VectorIndex component package.

base.py = interface + IndexConfigMismatch, sqlite.py = SQLite/numpy
implementation, __init__.py = registry + factory. A Qdrant/ANN backend = new
module implementing VectorIndex + one _STORES entry, selected via a
profile's `store` field. One store per profile — vectors from different
models/chunkers never share an index; stores are disposable and rebuildable
from output/ via `python -m app.indexer --profile <name>`.
"""

from pathlib import Path

from ..config import ConfigError, Profile, get_profile
from .base import IndexConfigMismatch, VectorIndex
from .sqlite import SqliteVectorStore

_STORES: dict[str, type[VectorIndex]] = {"sqlite": SqliteVectorStore}


def open_store(
    profile: Profile | None = None, db_path: Path | None = None
) -> VectorIndex:
    profile = profile or get_profile()
    if profile.store not in _STORES:
        raise ConfigError(
            f"unknown store '{profile.store}'; available: {sorted(_STORES)}"
        )
    return _STORES[profile.store](profile=profile, db_path=db_path)


__all__ = ["VectorIndex", "IndexConfigMismatch", "SqliteVectorStore", "open_store"]
