"""Retrieval — the ONE question -> ranked-chunks path.

Every consumer (the /search endpoint, /ask RAG composition, evals, future
agent tools) calls retrieve(); nothing else embeds a query or opens a store
for searching. That makes this module the single seam where retrieval
improves — hybrid BM25+RRF, reranking, query rewriting all land INSIDE
here, and every caller inherits them without changing.

Learning notes:
- Capability vs transport: this is a plain function returning plain dicts.
  HTTP concerns (status codes, rounding for display) stay in main.py; eval
  code measures THIS path — the same one production answers flow through.
- The row shape (doc_id, chunk_id, section, case_title, text, score) is the
  retrieval contract downstream code builds on; extend it, don't break it.
"""

from .config import Profile, get_profile
from .embeddings import get_embedder
from .vector_store import open_store


def retrieve(
    question: str, k: int = 8, profile: Profile | str | None = None
) -> list[dict]:
    """Top-k chunks for a question, best first.

    Raises ConfigError (bad profile), EmbeddingError (Ollama down),
    IndexConfigMismatch (store built by a different profile) — callers map
    these to their own error handling (HTTP codes, eval skips...).
    """
    prof = profile if isinstance(profile, Profile) else get_profile(profile)
    vector = get_embedder(prof).embed_query(question)
    with open_store(prof) as store:
        return store.search(vector, k=k)
