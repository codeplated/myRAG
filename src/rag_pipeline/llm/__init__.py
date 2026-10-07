"""Pick the language model named in the settings.

LLM_PROVIDER decides which service answers: a local Ollama model, or a hosted
model from OpenAI or Anthropic. The rest of the pipeline only sees the `LLM`
interface, so switching provider is a one-line change in `.env`.
"""
from __future__ import annotations

from ..config import Settings
from .base import LLM, LLMError

__all__ = ["LLM", "LLMError", "get_llm"]


def get_llm(settings: Settings) -> LLM:
    # Providers are imported here so that only the selected SDK has to be installed.
    if settings.llm_provider == "openai":
        from .openai_provider import OpenAILLM

        return OpenAILLM(settings.openai_model, settings.openai_api_key, settings.openai_base_url)
    if settings.llm_provider == "anthropic":
        from .anthropic_provider import AnthropicLLM

        return AnthropicLLM(
            settings.anthropic_model, settings.anthropic_api_key, settings.anthropic_base_url
        )
    from .ollama import OllamaLLM

    think = {"true": True, "false": False}.get(settings.ollama_think)
    return OllamaLLM(settings.llm_model, settings.ollama_host, think=think, num_ctx=settings.ollama_num_ctx)
