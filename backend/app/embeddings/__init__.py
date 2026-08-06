"""Embedder component package.

base.py = interface + normalize helper, ollama.py = Ollama backend,
__init__.py = registry + factory. A new backend (OpenAI,
sentence-transformers...) = new module implementing Embedder + one
_EMBEDDERS entry + a profile naming it via `embedding.provider`.

Prefix note: prefix-trained models (nomic, mxbai) need distinct
document/query prefixes for asymmetric retrieval — they are config
(EmbeddingConfig), not code, so each model declares its own.
"""

from ..config import ConfigError, Profile, get_profile
from .base import Embedder, EmbeddingError
from .ollama import OllamaEmbedder

_EMBEDDERS: dict[str, type[Embedder]] = {"ollama": OllamaEmbedder}


def get_embedder(profile: Profile | None = None) -> Embedder:
    profile = profile or get_profile()
    provider = profile.embedding.provider
    if provider not in _EMBEDDERS:
        raise ConfigError(
            f"unknown embedding provider '{provider}'; available: {sorted(_EMBEDDERS)}"
        )
    return _EMBEDDERS[provider](profile.embedding)


__all__ = ["Embedder", "EmbeddingError", "OllamaEmbedder", "get_embedder"]
