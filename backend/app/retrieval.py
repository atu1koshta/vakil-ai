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
- Query rewriting ADDS ranked lists to the fusion, never replaces the raw
  question's lists. A citation phrase list is inert (empty, RRF unaffected)
  or decisive (exact match, surfaces at rank 1 of its list) — no middle
  mode where it injects plausible junk.
- `score` now lives on a THIRD scale: cosine (dense), RRF sum (hybrid), or
  cross-encoder logit (reranked — raw, unbounded, possibly negative).
  Comparable only within one result list, never across strategies.
- The reranker is WHY `candidates` can re-widen without paying 2b's
  false-consensus tax: junk that out-fuses gold at RRF rank gets re-scored
  by a judge that actually reads the pairs — false consensus was only
  dangerous when RRF rank was the final word.
"""

from .config import Profile, get_profile
from .embeddings import get_embedder
from .rerank import get_reranker
from .rewrite import detect_citations, llm_rewrite
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
    embedder = get_embedder(prof)
    with open_store(prof) as store:
        if cfg.strategy == "dense":
            # With rerank on, fetch a POOL, not k: reranking k rows would
            # only reorder them, never recover anything below the cut.
            depth = max(k, cfg.rerank.pool) if cfg.rerank.enabled else k
            pool = store.search(embedder.embed_query(question), k=depth)
        else:
            # Query ensemble: the raw question ALWAYS contributes
            # dense+lexical; a successful LLM rewrite adds one more pair of
            # lists. A bad rewrite can only inject extra candidates, never
            # displace the raw lists — failure is bounded by construction.
            queries = [question]
            if cfg.rewrite.llm:
                rewritten = llm_rewrite(question, cfg.rewrite.model or None)
                if rewritten:
                    queries.append(rewritten)
            rankings = []
            for q in queries:
                rankings.append(
                    store.search(embedder.embed_query(q), k=cfg.candidates)
                )
                rankings.append(store.lexical_search(q, k=cfg.candidates))
            if cfg.rewrite.citations:
                phrases = detect_citations(question)
                if phrases:
                    rankings.append(
                        store.lexical_phrase_search(phrases, k=cfg.candidates)
                    )
            pool = _rrf(rankings, cfg.rrf_k)
    if not cfg.rerank.enabled:
        return pool[:k]
    # Precision stage: the cross-encoder reads each (RAW question, chunk)
    # pair — the user's own words are the ground truth of intent, never the
    # rewrite — and keeps the best k of the pool.
    return get_reranker(prof).rerank(question, pool[: cfg.rerank.pool], k)
