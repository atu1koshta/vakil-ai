"""Chunker interface: Markdown -> retrieval chunks (+ extracted tables).

PROFILE-level component: chunk output feeds the per-profile index, and the
chunking config (including `strategy`) is part of the profile fingerprint —
switching strategies isolates into its own index.
"""

from abc import ABC, abstractmethod

from ..models import ChunkResult


class Chunker(ABC):
    @abstractmethod
    def chunk_document(self, markdown: str) -> ChunkResult: ...
