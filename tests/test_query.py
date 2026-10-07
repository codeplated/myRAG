import pytest

from rag_pipeline import query as query_module
from rag_pipeline.prompts import format_prompt, get_template
from rag_pipeline.query import NO_RESULTS_ANSWER, build_context, build_sources, run_query, stream_query

from .conftest import FakeLLM, make_chunk


@pytest.fixture
def found(monkeypatch):
    """Make retrieval return three fixed chunks."""
    chunks = [
        make_chunk(1, "The fuel limit is 500 kg.", book_id="manual", page=12),
        make_chunk(2, "Flaps are set to 15 degrees.", book_id="manual", page=14),
        make_chunk(3, "The Wright brothers flew in 1903.", book_id="history", page=3),
    ]
    monkeypatch.setattr(query_module, "retrieve", lambda settings, query, mode=None: list(chunks))
    return chunks


def test_context_numbers_each_source_with_document_and_page(found):
    context = build_context(found)

    assert "[1] manual, page 12\nThe fuel limit is 500 kg." in context
    assert "[3] history, page 3" in context


def test_context_shows_a_page_range_for_chunks_that_span_pages():
    chunk = make_chunk(1, "text")
    chunk.page_end = 2
    assert "[1] book, pages 1-2" in build_context([chunk])


def test_sources_carry_the_citation_number(found):
    sources = build_sources(found)

    assert [s["number"] for s in sources] == [1, 2, 3]
    assert sources[0]["book_id"] == "manual"
    assert sources[0]["page_start"] == 12
    assert sources[0]["content_preview"].startswith("The fuel limit")


def test_run_query_returns_answer_sources_and_timings(settings, found):
    # First reply is the reranker's order, second is the answer.
    llm = FakeLLM("[2, 1, 3]", "Flaps are set to 15 degrees [1].")

    result = run_query(settings, "What is the flap setting?", llm=llm)

    assert result["answer"] == "Flaps are set to 15 degrees [1]."
    assert [s["page_start"] for s in result["sources"]] == [14, 12, 3], "sources follow the reranked order"
    assert set(result["timings"]) == {"retrieval_seconds", "rerank_seconds", "generation_seconds"}


def test_answer_prompt_contains_question_sources_and_citation_rule(settings, found):
    llm = FakeLLM("[1, 2, 3]", "answer")
    run_query(settings, "What is the fuel limit?", llm=llm)

    answer_prompt = llm.prompts[-1]
    assert "Question: What is the fuel limit?" in answer_prompt
    assert "[1] manual, page 12" in answer_prompt
    assert settings.citation_instruction in answer_prompt


def test_stream_sends_sources_first_then_tokens_then_done(settings, found):
    llm = FakeLLM("[1, 2, 3]", "The limit is 500 kg [1].")

    events = list(stream_query(settings, "fuel?", llm=llm))

    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) > 1, "the answer arrives in several pieces"
    assert "".join(tokens) == "The limit is 500 kg [1]."


def test_no_results_gives_a_fixed_answer_without_calling_the_model(settings, monkeypatch):
    monkeypatch.setattr(query_module, "retrieve", lambda settings, query, mode=None: [])
    llm = FakeLLM("should not be used")

    result = run_query(settings, "anything", llm=llm)

    assert result["answer"] == NO_RESULTS_ANSWER
    assert result["sources"] == []
    assert llm.prompts == []


def test_search_mode_is_passed_to_retrieval(settings, monkeypatch):
    seen = {}

    def fake_retrieve(settings, query, mode=None):
        seen["mode"] = mode
        return []

    monkeypatch.setattr(query_module, "retrieve", fake_retrieve)
    run_query(settings, "q", llm=FakeLLM(), mode="keyword")

    assert seen["mode"] == "keyword"


def test_template_is_loaded_by_name_and_filled(settings):
    template = get_template(settings)
    prompt = format_prompt(template, context="CTX", query="Q?", citation_instruction="CITE")

    assert "CTX" in prompt and "Question: Q?" in prompt and "CITE" in prompt


def test_missing_template_is_reported(settings):
    with pytest.raises(FileNotFoundError):
        get_template(settings, "does_not_exist")


def test_braces_in_document_text_do_not_break_the_prompt(settings):
    template = get_template(settings)
    prompt = format_prompt(template, context="code: {x: 1}", query="what is {x}?")

    assert "{x: 1}" in prompt
