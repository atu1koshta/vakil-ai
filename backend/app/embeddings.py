"""Ollama embedding client (nomic-embed-text).

Learning notes:
- ONE model embeds both documents and queries — vectors from different models
  live in unrelated spaces, so this module is the single place embeddings are
  produced. Changing MODEL means re-indexing everything.
- nomic-embed-text is prefix-trained for asymmetric retrieval: documents must
  be embedded as "search_document: ..." and queries as "search_query: ...".
  Skipping the prefixes silently degrades retrieval quality.
- Vectors are L2-normalized here, at the source, so cosine similarity and dot
  product become interchangeable downstream.
- /api/embed accepts an array — one HTTP call per BATCH_SIZE chunks, not one
  per chunk. Batching is the main throughput lever when indexing a corpus.
"""
import math

import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL = "nomic-embed-text"
DIM = 768
BATCH_SIZE = 64


class EmbeddingError(RuntimeError):
    pass


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def _embed_batch(texts: list[str]) -> list[list[float]]:
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": MODEL, "input": texts},
            timeout=120.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise EmbeddingError(f"Ollama embed call failed: {e}") from e
    embeddings = resp.json().get("embeddings") or []
    if len(embeddings) != len(texts):
        raise EmbeddingError(f"expected {len(texts)} embeddings, got {len(embeddings)}")
    return [_normalize(v) for v in embeddings]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for indexing (batched, document prefix)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [f"search_document: {t}" for t in texts[i : i + BATCH_SIZE]]
        out.extend(_embed_batch(batch))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a search question (query prefix)."""
    return _embed_batch([f"search_query: {text}"])[0]
