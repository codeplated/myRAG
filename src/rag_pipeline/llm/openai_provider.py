"""Hosted models through the OpenAI API (or any server that speaks the same API)."""
from __future__ import annotations

from typing import Iterator

import openai

from .base import LLM, LLMError


class OpenAILLM(LLM):
    provider = "openai"

    def __init__(self, model: str, api_key: str = "", base_url: str = "") -> None:
        super().__init__(model)
        try:
            # Empty values fall back to the OPENAI_API_KEY / OPENAI_BASE_URL environment variables.
            self._client = openai.OpenAI(api_key=api_key or None, base_url=base_url or None)
        except openai.OpenAIError as exc:
            raise LLMError("OpenAI is selected but OPENAI_API_KEY is not set.") from exc

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            chunks = self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key.") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"OpenAI does not know the model '{self.model}'.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenAI rate limit reached. Try again shortly.") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI returned an error ({exc.status_code}).") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not connect to OpenAI.") from exc
