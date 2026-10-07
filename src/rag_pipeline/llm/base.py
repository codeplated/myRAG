"""Common interface for the language models that write answers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMError(RuntimeError):
    """A language model call failed. The message is safe to show to the user."""


class LLM(ABC):
    """One text-in, text-out model. Every provider implements `stream`."""

    provider: str = ""

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the answer piece by piece as the model writes it."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the complete answer as one string."""
        return "".join(self.stream(prompt, system)).strip()
