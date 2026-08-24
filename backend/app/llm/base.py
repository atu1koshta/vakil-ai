"""ChatModel interface: (system, user) -> assistant text, plus tool turns.

Callers never know which provider or model answers — RAG code depends on
this ABC only. TOP-LEVEL component (like parser/metadata, unlike
embedder): generation only reads retrieved text, so it is outside profile
fingerprints and swapping it never invalidates an index.

chat() is one BLOCKING request/response turn — no SSE, no streaming, no
conversation memory. chat_tools() is the messages-list extension the agent
loop needed (added when a consumer needed it, per the original note here).

Learning notes:
- The NEUTRAL message format is OpenAI-style dicts; providers translate at
  the wire, callers never see provider quirks:
    {"role": "system"|"user", "content": str}
    {"role": "assistant", "content": str,
     "tool_calls": [{"id": str, "name": str, "arguments": <json str>}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
  `arguments` is ALWAYS a raw JSON string in neutral form (OpenAI's native
  shape); Ollama wants a dict on the wire, so its adapter converts both
  directions. The agent loop, not the provider, parses/validates it.
- chat_tools() is non-abstract with a raising default so providers that
  can't do tool calling aren't forced to stub it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class GenerationError(RuntimeError):
    pass


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str  # provider id; Ollama has none -> adapter synthesizes "call_<i>"
    name: str
    arguments: str  # raw JSON string — parsed by the agent, not the provider


@dataclass
class ToolChatResult:
    """One assistant turn in a tool conversation."""

    content: str  # may be "" when the model only calls tools
    tool_calls: list[ToolCall] = field(default_factory=list)  # empty = final answer


class ChatModel(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        """One chat turn. temperature=None means the configured default."""

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        temperature: float | None = None,
    ) -> ToolChatResult:
        """One assistant turn given a full message history and tool schemas
        (OpenAI function format). Empty `tools` = final-answer turn (adapters
        omit the key on the wire). Raises GenerationError if unsupported."""
        raise GenerationError(
            f"{type(self).__name__} does not implement tool calling"
        )
