"""Hosted Claude models through the Anthropic API."""
from __future__ import annotations

from typing import Iterator

import anthropic

from .base import LLM, LLMError

MAX_TOKENS = 16000

# If Claude's safety system declines a request, the API can re-run it on another
# model inside the same call. These models support that server-side fallback.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str = "", base_url: str = "", effort: str = "medium") -> None:
        super().__init__(model)
        self._effort = effort
        try:
            # Empty values fall back to ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL or a saved login.
            self._client = anthropic.Anthropic(api_key=api_key or None, base_url=base_url or None)
        except anthropic.AnthropicError as exc:
            raise LLMError("Anthropic is selected but no API key or login was found.") from exc

    def _request(self, prompt: str, system: str | None) -> dict:
        request: dict = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        if not self.model.startswith("claude-haiku"):  # Haiku 4.5 does not take an effort setting
            request["output_config"] = {"effort": self._effort}
        if self.model in _FALLBACK_MODELS:
            request["betas"] = [_FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        try:
            with self._client.beta.messages.stream(**self._request(prompt, system)) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Anthropic does not know the model '{self.model}'.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached. Try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic returned an error ({exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not connect to Anthropic.") from exc

        if final.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        if final.stop_reason == "max_tokens":
            raise LLMError("The answer was cut off because it reached the length limit.")
