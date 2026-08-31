"""The same three tools, declared the LangChain way: annotations, not dicts.

Learning notes:
- tools.py writes OpenAI JSON schemas BY HAND; here @tool derives the same
  schema FROM the function — type hints become parameter types, Annotated
  strings become parameter descriptions, the docstring becomes the tool
  description. One source of truth, no hand-sync. Compare search_chunks
  here against its TOOL_SCHEMAS entry — identical wire format.
- Tools are built per run by a factory, not at module level, for two
  bindings the model must never control: `profile` (which vector index)
  and `trace` (the AgentStep list the UI renders). Closures carry both;
  the model only ever sees the annotated parameters.
- Bodies delegate to tools.run_tool — executors, caps and ERROR-string
  self-correction semantics stay identical across both agent loops.
"""

import time
from typing import Annotated

from langchain_core.tools import BaseTool, tool

from .base import AgentStep
from .tools import run_tool

PREVIEW_CHARS = 400


def build_tools(profile: str | None, trace: list[AgentStep]) -> list[BaseTool]:
    """Three annotated tools bound to one run's profile + trace."""

    def _record(name: str, args: dict) -> str:
        started = time.perf_counter()
        result = run_tool(name, args, profile=profile)
        trace.append(
            AgentStep(
                tool=name,
                args=args,
                result_preview=result[:PREVIEW_CHARS],
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=result.startswith("ERROR:"),
            )
        )
        return result

    @tool
    def search_chunks(
        query: Annotated[str, "What to search for (a question or key phrase)."],
        k: Annotated[int, "How many passages to return (default 8, max 12)."] = 8,
    ) -> str:
        """Semantic + keyword search over all judgment chunks. Returns
        passages with [doc_id:chunk_id | SECTION] headers — cite those ids.
        Start here for most questions."""
        return _record("search_chunks", {"query": query, "k": k})

    @tool
    def filter_documents(
        court: Annotated[str | None, "Substring of the court name, e.g. 'Bombay'."] = None,
        year_from: Annotated[int | None, "Earliest judgment year, inclusive."] = None,
        year_to: Annotated[int | None, "Latest judgment year, inclusive."] = None,
        title_contains: Annotated[str | None, "Substring of the case title."] = None,
    ) -> str:
        """List judgments by metadata. court and title_contains match as
        case-insensitive SUBSTRINGS — pass short values like 'Bombay' or
        'Supreme', not full court names. Use for 'which cases...' / court /
        year questions. No arguments = list all documents."""
        return _record(
            "filter_documents",
            {
                "court": court,
                "year_from": year_from,
                "year_to": year_to,
                "title_contains": title_contains,
            },
        )

    @tool
    def read_document(
        doc_id: Annotated[str, "A doc_id seen in earlier tool results."],
        section: Annotated[
            str | None, "Section name from the section list, e.g. 'Judgment'."
        ] = None,
    ) -> str:
        """Read a full section of one judgment — use when a search chunk cuts
        off mid-reasoning. Call with doc_id only to list the document's
        sections, then call again with one section name."""
        return _record("read_document", {"doc_id": doc_id, "section": section})

    return [search_chunks, filter_documents, read_document]
