"""The hand-rolled agent loop — the whole trick, with nothing hidden.

Learning notes:
- The loop IS the agent: call the model with tool schemas; if it returns
  tool_calls, execute them, append results, repeat; if it returns text,
  that's the answer. Everything a framework adds (graphs, checkpoints,
  retries) layers onto this — built raw first so 3b's LangGraph port is a
  translation, not a mystery.
- Three stop conditions, in order of preference: the model answers on its
  own; an identical tool call repeats (result served from cache with a
  NOTE, nudging it to move on); max_steps exhausts, after which ONE final
  call with tools disabled forces an answer from gathered evidence.
- Bad tool arguments and unknown tool names are fed BACK to the model as
  "ERROR: ..." tool results, not raised — the model can self-correct;
  crashing the loop teaches it nothing.
"""

import json
import time

from ..config import get_chat_config
from ..llm import get_chat_model
from .base import Agent, AgentError, AgentResult, AgentStep
from .tools import TOOL_SCHEMAS, run_tool

MAX_STEPS_CAP = 12  # guard: ?max_steps=999 would blow the context window
PREVIEW_CHARS = 400

AGENT_SYSTEM_PROMPT = """You are a legal research assistant answering questions about Indian court judgments, using tools to gather evidence.

Rules:
- Answer ONLY from tool results. Never use outside knowledge of cases or law.
- Every factual claim must cite its source id exactly as it appeared in tool results: [doc_id:chunk_id] for search results, [doc_id:SECTION] for read_document text. Never invent doc_ids or chunk_ids.
- Start with search_chunks for most questions. Use filter_documents for court/year/"which cases" questions. Use read_document when a passage cuts off mid-reasoning and you need the full section.
- Call tools ONLY through the tool-calling mechanism. Never write tool-call JSON in your reply text.
- When the evidence already answers the question, reply with your final answer WITHOUT calling more tools.
- If the evidence is insufficient, say exactly what is missing instead of guessing."""

def _looks_like_tool_json(content: str) -> bool:
    """True when the reply is (or ends with) a bare {"name": ..., "parameters"/
    "arguments"/...} blob — a tool call leaked into text, not an answer."""
    text = content.strip()
    if not text.endswith("}"):
        return False
    # Outermost parse wins: scan "{" positions left to right (rfind would
    # land on a NESTED object like {"q": ...} and miss the "name" wrapper).
    start = text.find("{")
    while start != -1:
        try:
            blob = json.loads(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return isinstance(blob, dict) and "name" in blob
    return False


FORCED_ANSWER_PROMPT = (
    "You have no tool calls left. Give your final answer now from the "
    "evidence above, with [doc_id:chunk_id] citations; if the evidence is "
    "insufficient, say exactly what is missing."
)


class HandRolledAgent(Agent):
    def run(
        self,
        question: str,
        *,
        model: str | None = None,
        profile: str | None = None,
        max_steps: int = 6,
        history: list[dict] | None = None,
    ) -> AgentResult:
        max_steps = max(1, min(max_steps, MAX_STEPS_CAP))
        cfg = get_chat_config(model)
        chat = get_chat_model(model)
        # Prior turns sit between system and the new question, so the model
        # can resolve references like "the earlier case" — the whole point
        # of chat. They contain only user/assistant text (sessions.py).
        prior = list(history or [])
        messages: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *prior,
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []
        seen: dict[tuple[str, str], str] = {}  # (tool, canonical args) -> result
        iterations = 0

        for _ in range(max_steps):
            result = chat.chat_tools(messages, TOOL_SCHEMAS)
            iterations += 1
            if not result.tool_calls:
                # Witnessed llama3.1 failure: after a tool error it sometimes
                # writes the NEXT call as JSON text instead of a structured
                # tool call. Don't accept that as an answer — nudge and retry.
                if _looks_like_tool_json(result.content):
                    messages.append({"role": "assistant", "content": result.content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That was tool-call JSON in plain text, which "
                                "does nothing. Use the tool-calling mechanism, "
                                "or give your final answer as prose."
                            ),
                        }
                    )
                    continue
                return AgentResult(
                    question=question,
                    answer=result.content,
                    model=cfg.name,
                    steps=steps,
                    iterations=iterations,
                    history=self._next_history(prior, question, result.content),
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in result.tool_calls
                    ],
                }
            )
            for tc in result.tool_calls:
                args, output = self._execute(tc.name, tc.arguments, seen, profile)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": output[1],
                    }
                )
                steps.append(
                    AgentStep(
                        tool=tc.name,
                        args=args,
                        result_preview=output[1][:PREVIEW_CHARS],
                        duration_ms=output[0],
                        error=output[1].startswith("ERROR:"),
                    )
                )

        # Out of tool budget: force a final answer with tools disabled.
        messages.append({"role": "user", "content": FORCED_ANSWER_PROMPT})
        final = chat.chat_tools(messages, tools=[])
        iterations += 1
        if not final.content:
            raise AgentError("model produced no final answer after max_steps")
        return AgentResult(
            question=question,
            answer=final.content,
            model=cfg.name,
            steps=steps,
            iterations=iterations,
            exhausted=True,
            history=self._next_history(prior, question, final.content),
        )

    @staticmethod
    def _next_history(prior: list[dict], question: str, answer: str) -> list[dict]:
        """History to persist: prior turns + this one, tool noise dropped."""
        return [
            *prior,
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    @staticmethod
    def _execute(
        name: str,
        raw_arguments: str,
        seen: dict[tuple[str, str], str],
        profile: str | None,
    ) -> tuple[dict, tuple[int, str]]:
        """Parse, dedupe, run. Returns (parsed_args, (duration_ms, result))."""
        try:
            args = json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return {"_raw": (raw_arguments or "")[:200]}, (
                0,
                f"ERROR: tool arguments were not valid JSON: {e}. "
                "Resend as a valid JSON object.",
            )
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            return args, (
                0,
                "NOTE: identical call already made; same result follows.\n"
                + seen[key],
            )
        started = time.perf_counter()
        result = run_tool(name, args, profile=profile)
        duration_ms = int((time.perf_counter() - started) * 1000)
        seen[key] = result
        return args, (duration_ms, result)
