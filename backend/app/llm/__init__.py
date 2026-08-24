"""ChatModel component package.

base.py = interface, ollama.py = Ollama backend, __init__.py = registry +
factory. A new backend (llama.cpp, OpenAI-compatible server...) = new
module implementing ChatModel + one _CHAT_MODELS entry + `generation.
provider` naming it in config.yaml. RAG code imports get_chat_model()
only — it never learns which model answered.
"""

from ..config import ConfigError, get_chat_config
from .base import ChatModel, GenerationError, ToolCall, ToolChatResult
from .ollama import OllamaChat
from .openai_compat import OpenAICompatChat

_CHAT_MODELS: dict[str, type[ChatModel]] = {
    "ollama": OllamaChat,
    "openai-compatible": OpenAICompatChat,
}


def get_chat_model(name: str | None = None) -> ChatModel:
    """Resolve by config name (generation.models key), not provider:
    explicit name > VAKIL_CHAT_MODEL env > generation.active."""
    cfg = get_chat_config(name)
    if cfg.provider not in _CHAT_MODELS:
        raise ConfigError(
            f"unknown generation provider '{cfg.provider}'; "
            f"available: {sorted(_CHAT_MODELS)}"
        )
    return _CHAT_MODELS[cfg.provider](cfg)


__all__ = [
    "ChatModel",
    "GenerationError",
    "OllamaChat",
    "OpenAICompatChat",
    "ToolCall",
    "ToolChatResult",
    "get_chat_model",
]
