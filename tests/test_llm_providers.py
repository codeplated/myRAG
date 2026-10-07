"""Run the real provider clients against the fake API server from conftest."""
import pytest

from rag_pipeline.embeddings_ollama import EmbeddingError, embed_query
from rag_pipeline.llm import LLMError, get_llm
from rag_pipeline.llm.anthropic_provider import AnthropicLLM
from rag_pipeline.llm.ollama import OllamaLLM
from rag_pipeline.llm.openai_provider import OpenAILLM

from .conftest import make_settings

EXPECTED = "Hello from the fake API"


# --- Ollama ---------------------------------------------------------------


def test_ollama_streams_and_joins_the_answer(fake_api):
    llm = OllamaLLM("qwen3:8b", fake_api.url)

    assert list(llm.stream("Hi")) == fake_api.words
    assert llm.generate("Hi", system="Be brief.") == EXPECTED

    body = fake_api.last["body"]
    assert fake_api.last["path"] == "/api/chat"
    assert body["model"] == "qwen3:8b"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]


def test_ollama_switches_thinking_off_and_sets_the_context_size(fake_api):
    OllamaLLM("qwen3:8b", fake_api.url, think=False, num_ctx=8192).generate("Hi")

    assert fake_api.last["body"]["think"] is False
    assert fake_api.last["body"]["options"] == {"num_ctx": 8192}


def test_ollama_leaves_the_model_default_when_thinking_is_not_configured(fake_api):
    OllamaLLM("qwen3:8b", fake_api.url, think=None).generate("Hi")

    assert "think" not in fake_api.last["body"]
    assert "options" not in fake_api.last["body"]


def test_ollama_retries_without_the_setting_for_models_that_cannot_think(fake_api):
    fake_api.rejects_think = True
    llm = OllamaLLM("llama3", fake_api.url, think=False)

    assert llm.generate("Hi") == EXPECTED
    assert len(fake_api.requests) == 2, "one rejected call, one retry"

    llm.generate("Hi again")
    assert len(fake_api.requests) == 3, "the model is remembered, so later calls need no retry"


def test_ollama_missing_model_tells_the_user_how_to_fix_it(fake_api):
    fake_api.status = 404
    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        OllamaLLM("qwen3:8b", fake_api.url).generate("Hi")


def test_ollama_not_running_is_reported_clearly():
    with pytest.raises(LLMError, match="not reachable"):
        OllamaLLM("qwen3:8b", "http://127.0.0.1:1").generate("Hi")


# --- OpenAI ---------------------------------------------------------------


def test_openai_streams_and_joins_the_answer(fake_api):
    llm = OpenAILLM("gpt-test", api_key="test-key", base_url=fake_api.url + "/v1")

    assert llm.generate("Hi", system="Be brief.") == EXPECTED

    request = fake_api.last
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer test-key"
    assert request["body"]["model"] == "gpt-test"
    assert request["body"]["stream"] is True
    assert request["body"]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert request["body"]["messages"][-1] == {"role": "user", "content": "Hi"}


def test_openai_bad_key_is_reported_clearly(fake_api):
    fake_api.status = 401
    llm = OpenAILLM("gpt-test", api_key="wrong", base_url=fake_api.url + "/v1")
    with pytest.raises(LLMError, match="rejected the API key"):
        llm.generate("Hi")


def test_openai_unknown_model_is_reported_clearly(fake_api):
    fake_api.status = 404
    llm = OpenAILLM("gpt-nope", api_key="test-key", base_url=fake_api.url + "/v1")
    with pytest.raises(LLMError, match="gpt-nope"):
        llm.generate("Hi")


def test_openai_without_a_key_fails_at_start(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAILLM("gpt-test")


# --- Anthropic ------------------------------------------------------------


def test_anthropic_streams_and_joins_the_answer(fake_api):
    llm = AnthropicLLM("claude-opus-5-5", api_key="test-key", base_url=fake_api.url)

    assert list(llm.stream("Hi")) == fake_api.words
    assert llm.generate("Hi", system="Be brief.") == EXPECTED

    request = fake_api.last
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == "test-key"
    body = request["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "Be brief."
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["stream"] is True
    assert body["output_config"] == {"effort": "medium"}
    assert "temperature" not in body, "current Claude models reject sampling parameters"


def test_anthropic_enables_the_refusal_fallback_on_models_that_support_it(fake_api):
    AnthropicLLM("claude-opus-5-5", api_key="k", base_url=fake_api.url).generate("Hi")

    assert fake_api.last["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in fake_api.last["headers"]["anthropic-beta"]


def test_anthropic_sends_no_effort_or_fallback_to_haiku(fake_api):
    AnthropicLLM("claude-haiku-4-5", api_key="k", base_url=fake_api.url).generate("Hi")

    body = fake_api.last["body"]
    assert "output_config" not in body
    assert "fallbacks" not in body


def test_anthropic_refusal_is_reported_instead_of_returned_as_an_answer(fake_api):
    fake_api.stop_reason = "refusal"
    llm = AnthropicLLM("claude-opus-5-5", api_key="k", base_url=fake_api.url)
    with pytest.raises(LLMError, match="declined"):
        llm.generate("Hi")


def test_anthropic_bad_key_is_reported_clearly(fake_api):
    fake_api.status = 401
    llm = AnthropicLLM("claude-opus-5-5", api_key="wrong", base_url=fake_api.url)
    with pytest.raises(LLMError, match="rejected the API key"):
        llm.generate("Hi")


# --- Provider selection ---------------------------------------------------


@pytest.mark.parametrize(
    "provider, expected_class, expected_model",
    [
        ("ollama", OllamaLLM, "qwen3:8b"),
        ("openai", OpenAILLM, "gpt-test"),
        ("anthropic", AnthropicLLM, "claude-opus-5-5"),
    ],
)
def test_provider_is_chosen_from_settings(tmp_path, provider, expected_class, expected_model):
    settings = make_settings(tmp_path, llm_provider=provider)
    llm = get_llm(settings)

    assert isinstance(llm, expected_class)
    assert llm.model == expected_model
    assert settings.active_llm_model == expected_model


# --- Embeddings -----------------------------------------------------------


def test_embed_query_returns_one_vector(tmp_path, fake_api):
    settings = make_settings(tmp_path, ollama_host=fake_api.url)

    assert embed_query(settings, "what is the fuel limit?") == [0.1, 0.2, 0.3, 0.4]
    assert fake_api.last["body"] == {"model": "bge-m3", "input": ["what is the fuel limit?"]}


def test_wrong_embedding_size_is_caught_before_it_reaches_the_database(tmp_path, fake_api):
    settings = make_settings(tmp_path, ollama_host=fake_api.url, embedding_dim=1024)
    with pytest.raises(EmbeddingError, match="EMBEDDING_DIM"):
        embed_query(settings, "question")


def test_missing_embedding_model_tells_the_user_how_to_fix_it(tmp_path, fake_api):
    fake_api.status = 404
    settings = make_settings(tmp_path, ollama_host=fake_api.url)
    with pytest.raises(EmbeddingError, match="ollama pull bge-m3"):
        embed_query(settings, "question")
