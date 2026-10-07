import pytest

from rag_pipeline.config import get_settings


def test_defaults(monkeypatch):
    for name in ("LLM_PROVIDER", "SEARCH_MODE", "RERANK_MODE", "CHUNK_SIZE", "BOOKS_DIR"):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.llm_provider == "ollama"
    assert settings.search_mode == "hybrid"
    assert settings.rerank_mode == "llm"
    assert settings.chunk_size == 1000
    assert settings.books_dir.is_absolute()
    assert settings.books_dir.parts[-2:] == ("data", "books")


def test_values_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("CHUNK_SIZE", "500")

    settings = get_settings()

    assert settings.llm_provider == "anthropic"
    assert settings.active_llm_model == "claude-haiku-4-5"
    assert settings.chunk_size == 500


def test_number_that_is_not_a_number_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "big")
    assert get_settings().chunk_size == 1000


@pytest.mark.parametrize("name", ["LLM_PROVIDER", "SEARCH_MODE", "RERANK_MODE"])
def test_unknown_choice_is_rejected_with_the_setting_name(monkeypatch, name):
    monkeypatch.setenv(name, "something-else")
    with pytest.raises(ValueError, match=name):
        get_settings()
