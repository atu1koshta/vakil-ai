"""Reranker interface: (question, candidate rows) -> reordered top-k.

PROFILE-level component, but OUTSIDE the fingerprint: a reranker re-scores
rows the index already returned — it never changes what vectors exist, so
flipping it on/off or swapping models must not invalidate any index.

Learning notes:
- This IS a component package (unlike rewrite.py, a plain module): reranker
  backends genuinely vary — cross-encoders, LLM-as-judge, cloud APIs like
  Cohere — so the ABC + registry + factory layout earns its keep.
- rerank() REPLACES each returned row's `score` with the reranker's own
  relevance score. The pool's RRF scores are meaningless once a better
  judge has read the (question, chunk) pairs — carrying them forward would
  invite downstream code to compare incomparable scales.
"""

from abc import ABC, abstractmethod


class RerankError(RuntimeError):
    pass


class Reranker(ABC):
    @abstractmethod
    def rerank(self, question: str, rows: list[dict], k: int) -> list[dict]:
        """Top-k of `rows` for `question`, best first, `score` replaced by
        this reranker's relevance score (scale is implementation-defined,
        comparable only within one result list)."""
