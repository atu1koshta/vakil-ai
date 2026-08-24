"""Agent component package.

base.py = interface + trace dataclasses, hand_rolled.py = the raw loop,
tools.py = the shared toolset, __init__.py = registry + factory. Step 3b
adds a "langgraph" entry implementing the same Agent ABC — callers keep
importing get_agent() only.
"""

from ..config import ConfigError
from .base import Agent, AgentError, AgentResult, AgentStep
from .hand_rolled import HandRolledAgent

_AGENTS: dict[str, type[Agent]] = {
    "hand-rolled": HandRolledAgent,
}


def get_agent(kind: str = "hand-rolled") -> Agent:
    if kind not in _AGENTS:
        raise ConfigError(f"unknown agent '{kind}'; available: {sorted(_AGENTS)}")
    return _AGENTS[kind]()


__all__ = [
    "Agent",
    "AgentError",
    "AgentResult",
    "AgentStep",
    "HandRolledAgent",
    "get_agent",
]
