"""RAG composition — question -> grounded, cited answer.

The one place retrieval, context assembly, and generation meet:
ask() = retrieve() -> assemble() -> ChatModel.chat(), plus the grounding
prompt that turns a capable-but-unmoored chat model into a witness that
only reports what the excerpts say. /ask and the generation evals (2d)
both call THIS function — same seam pattern as retrieve() and assemble().

The grounding prompt carries three load-bearing clauses:
1. Restriction — answer ONLY from the excerpts; recognizing a case name
   is not permission to use memorized knowledge about it (the parametric-
   leakage failure witnessed in step 1).
2. Escape hatch — "if the excerpts are insufficient, say so". The single
   biggest anti-hallucination lever: without an allowed way out, a model
   ordered to answer will fabricate one.
3. Citation contract — every claim cites [doc_id:chunk_id], the exact ids
   assemble()'s provenance headers put in front of each excerpt. Citations
   are only checkable because the header format and the contract match.

Learning notes:
- Generation params (model name, temperature) are TOP-LEVEL config, not
  profile knobs — see llm/base.py. ask() therefore takes model and profile
  as independent arguments: any chat model over any index.
- retrieve() wider than assemble() keeps (k=12 vs max_chunks=8) is
  deliberate slack: rank order near the tail is noisy, and the budget may
  evict a large mid-ranked chunk — spare rows cost only a slightly larger
  candidate list, never context tokens.
- When nothing is retrievable (empty index) we return a canned refusal
  WITHOUT calling the LLM: sending a bare question to the model is exactly
  the ungrounded mode this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Profile, get_chat_config
from .context import DEFAULT_MAX_CHUNKS, AssembledContext, assemble
from .llm import get_chat_model
from .retrieval import retrieve

# Candidate pool for assemble() to cull from; see module notes on the slack.
DEFAULT_RETRIEVE_K = 12

SYSTEM_PROMPT = """\
You are a legal research assistant answering questions about Indian court \
judgments. You will be given excerpts from these judgments. Each excerpt \
starts with a provenance header of the form [doc_id:chunk_id | SECTION] \
followed by the case title.

Rules:
1. Answer ONLY from the provided excerpts. Do not use outside or memorized \
knowledge about cases, statutes, or doctrine — even when you recognize the \
case name, everything you state must come from the excerpt text.
2. Cite the supporting excerpt for every claim inline, using the exact ids \
from its header in square brackets, e.g. [some-case-1978:chunk_004]. Multiple \
citations for one claim are fine.
3. If the excerpts do not contain enough information to answer the question, \
say so explicitly and state what is missing. Do not guess. An honest "the \
excerpts do not establish this" or a partial answer with citations is always \
better than a complete-sounding answer without support.
4. Be concise. Quote the judgment's own words where the exact wording matters, \
and attribute quotes to their excerpt.\
"""

NO_EVIDENCE_ANSWER = (
    "No indexed excerpts were retrieved for this question, so I cannot give "
    "a grounded answer. Index documents first (python -m app.indexer) or "
    "rephrase the question."
)


@dataclass
class AskResult:
    """Answer + the evidence accounting behind it."""

    question: str
    answer: str
    model: str  # generation config name that answered (e.g. 'llama')
    context: AssembledContext
    grounded: bool  # False = canned refusal, the LLM was never called


def build_user_prompt(context_text: str, question: str) -> str:
    """The user turn: evidence first, question last (recency helps
    instruction-following). Shared so evals can reproduce the exact prompt."""
    return f"Excerpts:\n\n{context_text}\n\nQuestion: {question}"


def ask(
    question: str,
    *,
    k: int = DEFAULT_RETRIEVE_K,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    model: str | None = None,
    profile: Profile | str | None = None,
) -> AskResult:
    """One grounded QA turn over the indexed corpus.

    Raises ConfigError (bad model/profile name), EmbeddingError (Ollama
    down), IndexConfigMismatch (stale index), GenerationError (chat model
    failed) — callers map these to their own error handling.
    """
    cfg = get_chat_config(model)
    rows = retrieve(question, k=k, profile=profile)
    ctx = assemble(
        rows,
        question,
        SYSTEM_PROMPT,
        num_ctx=cfg.num_ctx,
        max_chunks=max_chunks,
    )
    if not ctx.kept:
        return AskResult(question, NO_EVIDENCE_ANSWER, cfg.name, ctx, grounded=False)

    answer = get_chat_model(model).chat(
        SYSTEM_PROMPT, build_user_prompt(ctx.text, question)
    )
    return AskResult(question, answer, cfg.name, ctx, grounded=True)
