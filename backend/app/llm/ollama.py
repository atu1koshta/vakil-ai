"""Ollama-backed ChatModel.

Learning notes:
- stream=False: Ollama buffers the full completion into one JSON response.
  With stream=true it would send incremental NDJSON lines (not SSE — SSE is
  a browser-facing framing we'd add at OUR API layer if the UI wants
  token-by-token display).
- num_ctx fails SILENTLY: Ollama's default context window is small (2-4k
  tokens). A RAG prompt with 8 x 700-token chunks overflows it, and Ollama
  just TRUNCATES the front — system prompt and first chunks vanish, the
  model answers from whatever survived. Size num_ctx to system + context +
  question + answer headroom.
- temperature=0 default for grounded QA: most probable answer given the
  evidence, not creative variety. Sampling randomness is where "same
  question, different citation" flakiness comes from.
"""

import re

import httpx

from ..config import GenerationConfig
from .base import ChatModel, GenerationError

# Reasoning models (deepseek-r1...) emit chain-of-thought before the answer.
# Newer Ollama versions put it in a separate `thinking` field; older ones
# leave <think>...</think> inline in content. Handle both — callers get the
# ANSWER only; thinking is a provider artifact, not part of the interface.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OllamaChat(ChatModel):
    def __init__(self, cfg: GenerationConfig) -> None:
        self.cfg = cfg

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": (
                            self.cfg.temperature if temperature is None else temperature
                        ),
                        "num_ctx": self.cfg.num_ctx,
                    },
                },
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise GenerationError(f"Ollama chat call failed: {e}") from e
        content = (resp.json().get("message") or {}).get("content") or ""
        content = _THINK_RE.sub("", content).strip()
        if not content:
            raise GenerationError("Ollama returned an empty message.")
        return content
