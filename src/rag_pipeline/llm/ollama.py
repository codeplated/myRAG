"""Local models served by Ollama."""
from __future__ import annotations

import json
from typing import Iterator

import requests

from .base import LLM, LLMError


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        host: str,
        think: bool | None = False,
        num_ctx: int | None = None,
        timeout: int = 600,
    ) -> None:
        """
        think: False switches a model's thinking mode off, which makes short tasks such as
               reranking several times faster. None leaves the model's own default.
        num_ctx: context window in tokens. Ollama's default (4096) silently cuts off long prompts.
        """
        super().__init__(model)
        self._url = host.rstrip("/") + "/api/chat"
        self._host = host
        self._think = think
        self._num_ctx = num_ctx
        self._timeout = timeout

    def _payload(self, prompt: str, system: str | None) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {"model": self.model, "messages": messages, "stream": True}
        if self._think is not None:
            payload["think"] = self._think
        if self._num_ctx:
            payload["options"] = {"num_ctx": self._num_ctx}
        return payload

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        try:
            resp = requests.post(self._url, json=self._payload(prompt, system), stream=True, timeout=self._timeout)
            if resp.status_code == 400 and self._think is not None and "think" in resp.text:
                # This model has no thinking mode. Remember that and ask again without the setting.
                resp.close()
                self._think = None
                resp = requests.post(
                    self._url, json=self._payload(prompt, system), stream=True, timeout=self._timeout
                )
            with resp:
                if resp.status_code == 404:
                    raise LLMError(
                        f"Model '{self.model}' is not available in Ollama. Run: ollama pull {self.model}"
                    )
                resp.raise_for_status()
                # Ollama streams one JSON object per line.
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise LLMError(f"Ollama error: {data['error']}")
                    text = (data.get("message") or {}).get("content")
                    if text:
                        yield text
        except requests.ConnectionError as exc:
            raise LLMError(f"Ollama is not reachable at {self._host}. Is it running?") from exc
        except requests.Timeout as exc:
            raise LLMError(f"Ollama did not answer within {self._timeout} seconds.") from exc
        except requests.HTTPError as exc:
            raise LLMError(f"Ollama returned an error: {exc}") from exc
