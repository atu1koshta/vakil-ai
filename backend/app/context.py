"""Context assembly — ranked rows -> one labeled, budgeted evidence block.

The single seam between retrieve() and the chat model: /ask, evals, and
future agent tools all build their evidence text HERE. Three decisions
live in this module and nowhere else:

1. Provenance labels: every chunk is headed [doc_id:chunk_id | section] —
   the anchor the grounding prompt's citation contract points at. No
   label, no citation.
2. k cap vs token budget: two separate knobs for two separate diseases.
   The budget stops silent front-truncation at num_ctx overflow; the k
   cap stops noise dilution even when the budget has room.
3. Whole-chunk packing: on overflow we drop the lowest-ranked chunks
   entirely, never cut text mid-chunk. A truncated excerpt reads as
   evidence but is missing its content — worse than absent.

Learning notes:
- Token counts use tiktoken cl100k_base, an APPROXIMATION of whatever
  tokenizer the active chat model uses (llama, deepseek...). Exactness is
  not the goal — staying safely under num_ctx is, which is what
  answer_headroom plus the count being conservative-enough buys. The
  alternative (loading each model's real tokenizer) couples this module
  to providers for no measurable win.
- Rows are packed in retrieve() rank order, so the budget evicts from the
  tail: the worst-ranked chunks go first. This mirrors what a human would
  cut and is the opposite of Ollama's overflow behavior (which eats the
  PROMPT FRONT — system instructions — silently).
- The return value keeps the kept/dropped split, not just the final
  string, so /ask can report which chunks made the cut and the break-it
  lab (step 5) can assert on eviction behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken

# Tokens reserved for the model's answer. Generous on purpose: it also
# absorbs the cl100k-vs-real-tokenizer approximation error.
DEFAULT_ANSWER_HEADROOM = 1024

# Noise-dilution cap, independent of the token budget (see module notes).
DEFAULT_MAX_CHUNKS = 8

_ENCODING = "cl100k_base"


@lru_cache(maxsize=4)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str) -> int:
    return len(_encoding(_ENCODING).encode(text))


def format_chunk(row: dict) -> str:
    """One evidence block: provenance header + raw chunk text.

    The header is the citation anchor — [doc_id:chunk_id | section] — and
    the case title gives the model attribution context without it having
    to infer which judgment the text belongs to.
    """
    header = f"[{row['doc_id']}:{row['chunk_id']} | {row['section']}]"
    return f"{header}\n{row['case_title']}\n\n{row['text']}"


@dataclass
class AssembledContext:
    """Evidence block + the accounting that produced it."""

    text: str  # joined evidence blocks, "" when nothing fit
    kept: list[dict] = field(default_factory=list)  # rows in packed order
    dropped: list[dict] = field(default_factory=list)  # rows evicted (budget or cap)
    context_tokens: int = 0  # tokens of `text`
    budget_tokens: int = 0  # what was available for evidence

    @property
    def kept_ids(self) -> list[str]:
        return [f"{r['doc_id']}:{r['chunk_id']}" for r in self.kept]


def assemble(
    rows: list[dict],
    question: str,
    system: str,
    *,
    num_ctx: int,
    answer_headroom: int = DEFAULT_ANSWER_HEADROOM,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> AssembledContext:
    """Pack retrieve() rows into one evidence block that fits the model.

    budget = num_ctx - system - question - answer_headroom. Rows are taken
    in given (rank) order; a row that doesn't fit whole is dropped along
    with everything after it — later rows are worse-ranked by contract, so
    packing them around a gap would trade a better chunk for worse ones.
    """
    budget = (
        num_ctx - count_tokens(system) - count_tokens(question) - answer_headroom
    )
    if budget <= 0:
        raise ValueError(
            f"no room for evidence: num_ctx={num_ctx} minus system/question/"
            f"headroom leaves {budget} tokens"
        )

    separator = "\n\n---\n\n"
    separator_cost = count_tokens(separator)

    kept: list[dict] = []
    dropped: list[dict] = []
    blocks: list[str] = []
    used = 0

    for i, row in enumerate(rows):
        if len(kept) >= max_chunks:
            dropped.extend(rows[i:])
            break
        block = format_chunk(row)
        cost = count_tokens(block) + (separator_cost if blocks else 0)
        if used + cost > budget:
            dropped.extend(rows[i:])
            break
        blocks.append(block)
        kept.append(row)
        used += cost

    return AssembledContext(
        text=separator.join(blocks),
        kept=kept,
        dropped=dropped,
        context_tokens=used,
        budget_tokens=budget,
    )
