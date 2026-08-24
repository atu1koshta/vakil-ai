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

import json
import re

import httpx

from ..config import GenerationConfig
from .base import ChatModel, GenerationError, ToolCall, ToolChatResult

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

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        temperature: float | None = None,
    ) -> ToolChatResult:
        body: dict = {
            "model": self.cfg.model,
            "messages": [self._to_wire(m) for m in messages],
            "stream": False,
            "options": {
                "temperature": (
                    self.cfg.temperature if temperature is None else temperature
                ),
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if tools:  # empty list = forced final answer; Ollama key must be absent
            body["tools"] = tools
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/api/chat",
                json=body,
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and "does not support tools" in (
                e.response.text or ""
            ):
                raise GenerationError(
                    f"model '{self.cfg.model}' does not support tool calling — "
                    "use a tool-capable config, e.g. model=llama or "
                    "model=deepseek-api"
                ) from e
            raise GenerationError(
                f"Ollama chat call failed ({e.response.status_code}): "
                f"{e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise GenerationError(f"Ollama chat call failed: {e}") from e
        message = resp.json().get("message") or {}
        tool_calls = [
            ToolCall(
                id=f"call_{i}",  # Ollama sends no ids; matching is positional
                name=(tc.get("function") or {}).get("name") or "",
                # Ollama arguments come as a dict — neutral form is a JSON string
                arguments=json.dumps((tc.get("function") or {}).get("arguments") or {}),
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        content = _THINK_RE.sub("", message.get("content") or "").strip()
        if not content and not tool_calls:
            raise GenerationError("Ollama returned neither content nor tool calls.")
        return ToolChatResult(content=content, tool_calls=tool_calls)

    @staticmethod
    def _to_wire(msg: dict) -> dict:
        """Neutral message -> Ollama wire shape. Assistant tool_calls carry
        arguments as a dict; tool results have no tool_call_id (positional)."""
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc["name"],
                            "arguments": json.loads(tc["arguments"]),
                        }
                    }
                    for tc in msg["tool_calls"]
                ],
            }
        if msg["role"] == "tool":
            return {"role": "tool", "content": msg["content"]}
        return {"role": msg["role"], "content": msg["content"]}
