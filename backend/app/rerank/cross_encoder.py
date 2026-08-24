"""Cross-encoder reranker: joint (question, chunk) reading on CPU.

Learning notes:
- A cross-encoder feeds question and chunk through the model TOGETHER,
  attending token-to-token — far more accurate than the bi-encoder's
  independent embeddings, and far too slow for a corpus. We only ever hand
  it the fused candidate pool (~tens of pairs), the standard two-stage
  split: fast-dumb over everything, slow-smart over the shortlist.
- Lazy import: sentence_transformers pulls in torch at import time; a
  profile with rerank disabled must never pay that. ImportError becomes a
  RerankError naming the pip install.
- Module-level model cache keyed by model name: retrieve() constructs a
  reranker per call, and without the cache every /search would reload the
  weights from disk (~80MB for MiniLM).
- 512-token pair limit: the model silently truncates each (question, chunk)
  pair, so evidence in a ~700-token chunk's tail is invisible. Known
  mitigation (sliding-window scoring, max over windows) is out of scope v1.
- v1 scores raw `text` only; scoring title+section-enriched text is a
  future knob to measure, not assume.
"""

from ..config import Profile
from .base import Reranker, RerankError

_MODEL_CACHE: dict[str, object] = {}


def _load_model(name: str):
    if name not in _MODEL_CACHE:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankError(
                "rerank.enabled needs sentence-transformers: "
                "pip install 'sentence-transformers>=3.0'"
            ) from e
        _MODEL_CACHE[name] = CrossEncoder(name)
    return _MODEL_CACHE[name]


class CrossEncoderReranker(Reranker):
    def __init__(self, profile: Profile):
        self.cfg = profile.retrieval.rerank

    def rerank(self, question: str, rows: list[dict], k: int) -> list[dict]:
        if not rows:
            return []
        model = _load_model(self.cfg.model)
        scores = model.predict(
            [(question, r["text"]) for r in rows],
            batch_size=self.cfg.batch_size,
        )
        ranked = sorted(zip(rows, scores), key=lambda p: p[1], reverse=True)
        return [row | {"score": float(s)} for row, s in ranked[:k]]
