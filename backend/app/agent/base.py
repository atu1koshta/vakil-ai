"""Agent interface: question in, cited answer + tool-call trace out.

An agent = LLM + tools + loop + state. The interface hides which loop runs
underneath — hand-rolled today, a LangGraph port in step 3b — the same way
ChatModel hides the provider. The trace (steps) is part of the contract:
an agent answer without its tool trail is unauditable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class AgentError(RuntimeError):
    pass


@dataclass
class AgentStep:
    """One executed tool call, as shown to the user."""

    tool: str
    args: dict  # parsed arguments; {"_raw": "..."} when JSON parsing failed
    result_preview: str  # first 400 chars of what the model saw
    duration_ms: int
    error: bool = False  # result was an ERROR: message fed back to the model


@dataclass
class AgentResult:
    question: str
    answer: str
    model: str  # generation config NAME (llama/deepseek-api...), like AskResult
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0  # LLM calls made
    exhausted: bool = False  # True = answer was forced after max_steps


class Agent(ABC):
    @abstractmethod
    def run(
        self,
        question: str,
        *,
        model: str | None = None,
        profile: str | None = None,
        max_steps: int = 6,
    ) -> AgentResult:
        """Answer one question, deciding tool use internally. model/profile
        resolve like everywhere else (explicit > env > config active)."""
