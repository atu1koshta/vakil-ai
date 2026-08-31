"""Agent component package.

base.py = interface + trace dataclasses, hand_rolled.py = the raw loop,
langgraph_agent.py = the 3b graph port, tools.py = the shared toolset
(lg_tools.py = the same tools as LangChain annotations), sessions.py =
chat history for the raw loop, __init__.py = registry + factory — callers
import get_agent() only.

Registry values are LOADERS, not classes: importing langgraph_agent pulls
langgraph+langchain, which the hand-rolled path must never pay for. The
selected kind resolves like every other switch: explicit > VAKIL_AGENT env
> config.yaml agent.kind.
"""

from typing import Callable

from ..config import ConfigError, get_agent_kind
from .base import Agent, AgentError, AgentResult, AgentStep


def _load_hand_rolled() -> type[Agent]:
    from .hand_rolled import HandRolledAgent

    return HandRolledAgent


def _load_langgraph() -> type[Agent]:
    from .langgraph_agent import LangGraphAgent

    return LangGraphAgent


_AGENTS: dict[str, Callable[[], type[Agent]]] = {
    "hand-rolled": _load_hand_rolled,
    "langgraph": _load_langgraph,
}


def get_agent(kind: str | None = None) -> Agent:
    resolved = get_agent_kind(kind)
    if resolved not in _AGENTS:
        raise ConfigError(f"unknown agent '{resolved}'; available: {sorted(_AGENTS)}")
    return _AGENTS[resolved]()()


__all__ = [
    "Agent",
    "AgentError",
    "AgentResult",
    "AgentStep",
    "get_agent",
]
