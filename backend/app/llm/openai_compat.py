"""ChatModel over any OpenAI-compatible /v1/chat/completions API.

Covers DeepSeek's hosted API (api.deepseek.com), OpenAI itself, Groq,
Together, vLLM servers... — one client, many vendors, because the wire
format became a de-facto standard.

Learning notes:
- Secrets discipline: config.yaml is committed, so it stores the NAME of an
  env var (api_key_env), never the key. Missing env var fails at first call
  with a message naming exactly what to export.
- num_ctx is Ollama-specific (local models need their context window sized
  by the caller); hosted APIs manage context server-side, so it's ignored
  here. Provider-specific knobs stay in provider code — the interface stays
  clean.
- deepseek-reasoner returns chain-of-thought in a separate
  `reasoning_content` field; `content` is already the clean answer. The
  <think> strip is kept anyway — harmless, and other vendors inline it.
"""

import os
import re

import httpx

from ..config import GenerationConfig
from .base import ChatModel, GenerationError, ToolCall, ToolChatResult

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OpenAICompatChat(ChatModel):
    def __init__(self, cfg: GenerationConfig) -> None:
        self.cfg = cfg

    def _api_key(self) -> str:
        if not self.cfg.api_key_env:
            raise GenerationError(
                f"chat model '{self.cfg.name}' uses an API provider but sets "
                "no api_key_env in config.yaml"
            )
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            raise GenerationError(
                f"env var {self.cfg.api_key_env} is not set — "
                f"export {self.cfg.api_key_env}=<your key> before starting"
            )
        return key

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key()}"},
                json={
                    "model": self.cfg.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "temperature": (
                        self.cfg.temperature if temperature is None else temperature
                    ),
                },
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            raise GenerationError(
                f"{self.cfg.name} API call failed ({e.response.status_code}): {body}"
            ) from e
        except httpx.HTTPError as e:
            raise GenerationError(f"{self.cfg.name} API call failed: {e}") from e
        choices = resp.json().get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        content = _THINK_RE.sub("", content or "").strip()
        if not content:
            raise GenerationError(f"{self.cfg.name} returned an empty message.")
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
            "temperature": (
                self.cfg.temperature if temperature is None else temperature
            ),
        }
        if tools:  # OpenAI-compatible APIs 400 on tools: []
            body["tools"] = tools
        try:
            resp = httpx.post(
                f"{self.cfg.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key()}"},
                json=body,
                timeout=self.cfg.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GenerationError(
                f"{self.cfg.name} API call failed ({e.response.status_code}): "
                f"{e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise GenerationError(f"{self.cfg.name} API call failed: {e}") from e
        choices = resp.json().get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=(tc.get("function") or {}).get("name") or "",
                # already a JSON string in the OpenAI wire format
                arguments=(tc.get("function") or {}).get("arguments") or "{}",
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        content = _THINK_RE.sub("", message.get("content") or "").strip()
        if not content and not tool_calls:
            raise GenerationError(
                f"{self.cfg.name} returned neither content nor tool calls."
            )
        return ToolChatResult(content=content, tool_calls=tool_calls)

    @staticmethod
    def _to_wire(msg: dict) -> dict:
        """Neutral message -> OpenAI wire shape. tool_call_id is REQUIRED on
        tool results; arguments stay JSON strings (native form)."""
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in msg["tool_calls"]
                ],
            }
        if msg["role"] == "tool":
            return {
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"],
            }
        return {"role": msg["role"], "content": msg["content"]}
