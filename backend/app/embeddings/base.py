"""Embedder interface: texts -> L2-normalized vectors.

Document and query paths are separate because prefix-trained models embed
them asymmetrically. PROFILE-level component: model identity (name, dim,
prefixes) is part of the profile fingerprint.
"""

import math
from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    pass


def normalize(vec: list[float]) -> list[float]:
    """L2-normalize at the source so cosine similarity = dot product
    everywhere downstream. Every implementation must return normalized
    vectors."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
