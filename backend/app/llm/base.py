"""ChatModel interface: (system, user) -> assistant text.

Callers never know which provider or model answers — RAG code depends on
this ABC only. TOP-LEVEL component (like parser/metadata, unlike
embedder): generation only reads retrieved text, so it is outside profile
fingerprints and swapping it never invalidates an index.

chat() is one BLOCKING request/response turn — no SSE, no streaming, no
conversation memory. Streaming and multi-turn state are interface changes
(a generator method / a messages list), added when a consumer needs them,
not speculatively.
"""

from abc import ABC, abstractmethod


class GenerationError(RuntimeError):
    pass


class ChatModel(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        """One chat turn. temperature=None means the configured default."""
