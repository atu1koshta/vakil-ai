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
- Hybrid fuses by RANK (RRF), never by score: dense cosine lives in [0,1],
  BM25 is unbounded and corpus-dependent — averaging them lets whichever
  scale is bigger silently win. Consequence: under hybrid, `score` is an RRF
  sum (~1/rrf_k scale), comparable within one result list but NOT across
  strategies. Nothing downstream may treat it as a similarity.
- Each ranker contributes `candidates` deep (retrieve wide), fusion cuts to
  k (return narrow): a doc absent from dense's top-k can still surface via
  its lexical rank — the whole point of 2b.
"""

from .config import Profile, get_profile
from .embeddings import get_embedder
from .vector_store import open_store


def _rrf(rankings: list[list[dict]], rrf_k: int) -> list[dict]:
    """Reciprocal Rank Fusion: fused(c) = Σ 1/(rrf_k + rank). Consensus
    across rankers beats a high rank in any single one."""
    fused: dict[tuple, float] = {}
    rows: dict[tuple, dict] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            key = (row["doc_id"], row["chunk_id"])
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank)
            rows.setdefault(key, row)
    order = sorted(fused, key=fused.__getitem__, reverse=True)
    return [rows[key] | {"score": fused[key]} for key in order]


def retrieve(
    question: str, k: int = 8, profile: Profile | str | None = None
) -> list[dict]:
    """Top-k chunks for a question, best first.

    Raises ConfigError (bad profile), EmbeddingError (Ollama down),
    IndexConfigMismatch (store built by a different profile) — callers map
    these to their own error handling (HTTP codes, eval skips...).
    """
    prof = profile if isinstance(profile, Profile) else get_profile(profile)
    cfg = prof.retrieval
    vector = get_embedder(prof).embed_query(question)
    with open_store(prof) as store:
        if cfg.strategy == "dense":
            return store.search(vector, k=k)
        dense = store.search(vector, k=cfg.candidates)
        lexical = store.lexical_search(question, k=cfg.candidates)
        return _rrf([dense, lexical], cfg.rrf_k)[:k]
