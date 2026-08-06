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
from .base import ChatModel, GenerationError

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
