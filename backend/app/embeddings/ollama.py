"""Ollama-backed Embedder.

Learning notes:
- /api/embed accepts an array — one HTTP call per batch_size chunks, not one
  per chunk. Batching is the main throughput lever when indexing a corpus.
- The dimension of every returned vector is checked against the profile's
  declared dim: a wrong `dim` in config fails on the first embed call, not as
  a reshape error deep in search.
"""

import httpx

from ..config import EmbeddingConfig
from .base import Embedder, EmbeddingError, normalize


class OllamaEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/api/embed",
                json={"model": self.cfg.model, "input": texts},
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise EmbeddingError(f"Ollama embed call failed: {e}") from e
        embeddings = resp.json().get("embeddings") or []
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"expected {len(texts)} embeddings, got {len(embeddings)}"
            )
        for v in embeddings:
            if len(v) != self.cfg.dim:
                raise EmbeddingError(
                    f"model {self.cfg.model} returned dim {len(v)}, "
                    f"config declares {self.cfg.dim} — fix `dim` in config.yaml"
                )
        return [normalize(v) for v in embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts for indexing (batched, document prefix)."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = [
                f"{self.cfg.document_prefix}{t}"
                for t in texts[i : i + self.cfg.batch_size]
            ]
            out.extend(self._embed_batch(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Embed a search question (query prefix)."""
        return self._embed_batch([f"{self.cfg.query_prefix}{text}"])[0]
