"""Reranker component package.

base.py = interface, cross_encoder.py = sentence-transformers backend,
__init__.py = registry + factory. A new backend (LLM-as-judge, Cohere...)
= new module implementing Reranker + one _RERANKERS entry +
`retrieval.rerank.provider` naming it in config.yaml.
"""

from ..config import ConfigError, Profile
from .base import Reranker, RerankError
from .cross_encoder import CrossEncoderReranker

_RERANKERS: dict[str, type[Reranker]] = {
    "cross-encoder": CrossEncoderReranker,
}


def get_reranker(profile: Profile) -> Reranker:
    provider = profile.retrieval.rerank.provider
    if provider not in _RERANKERS:
        raise ConfigError(
            f"unknown rerank provider '{provider}'; available: {sorted(_RERANKERS)}"
        )
    return _RERANKERS[provider](profile)


__all__ = ["Reranker", "RerankError", "CrossEncoderReranker", "get_reranker"]
