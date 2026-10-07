from typing import Iterator

from rag_pipeline.llm import LLMError
from rag_pipeline.reranker import build_rerank_prompt, parse_ranking, rerank

from .conftest import FakeLLM, make_chunk


def ids(chunks):
    return [c.id for c in chunks]


class BrokenLLM(FakeLLM):
    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        raise LLMError("model is down")
        yield ""


def test_parse_plain_array():
    assert parse_ranking("[3, 1, 2]", 5) == [2, 0, 1]


def test_parse_takes_the_last_array_when_the_model_explains_first():
    reply = "Passage [2] mentions fuel. Passage [4] is closest.\nFinal answer: [4, 2]"
    assert parse_ranking(reply, 5) == [3, 1]


def test_parse_drops_duplicates_and_numbers_out_of_range():
    assert parse_ranking("[2, 2, 9, 0, 1]", 3) == [1, 0]


def test_parse_returns_nothing_for_text_without_an_array():
    assert parse_ranking("I cannot decide.", 3) == []
    assert parse_ranking("", 3) == []


def test_prompt_numbers_the_passages_and_shortens_long_ones():
    chunks = [make_chunk(1, "short text"), make_chunk(2, "long " * 500)]
    prompt = build_rerank_prompt("What is the fuel limit?", chunks, top_n=1)

    assert "What is the fuel limit?" in prompt
    assert "[1] short text" in prompt
    assert "[2] long long" in prompt
    assert len(prompt) < 1500
    assert prompt.count("What is the fuel limit?") == 2, "the question is asked before and after the passages"
    assert prompt.index("What is the fuel limit?") < prompt.index("[1] short text")


def test_rerank_uses_the_model_order(settings):
    chunks = [make_chunk(i) for i in range(1, 6)]
    llm = FakeLLM("[4, 2, 5]")

    assert ids(rerank(settings, "q", chunks, llm=llm)) == [4, 2, 5]
    assert len(llm.prompts) == 1, "all candidates are ranked in a single call"


def test_rerank_fills_up_with_retrieval_order_when_the_model_returns_too_few(settings):
    chunks = [make_chunk(i) for i in range(1, 6)]
    assert ids(rerank(settings, "q", chunks, llm=FakeLLM("[5]"))) == [5, 1, 2]


def test_rerank_only_looks_at_the_first_candidates(settings):
    settings.rerank_candidates = 3
    chunks = [make_chunk(i) for i in range(1, 9)]
    llm = FakeLLM("[3, 1, 2]")

    assert ids(rerank(settings, "q", chunks, llm=llm)) == [3, 1, 2]
    assert "content of chunk 4" not in llm.prompts[0]


def test_rerank_keeps_retrieval_order_when_the_reply_is_unreadable(settings):
    chunks = [make_chunk(i) for i in range(1, 6)]
    assert ids(rerank(settings, "q", chunks, llm=FakeLLM("no idea"))) == [1, 2, 3]


def test_rerank_keeps_retrieval_order_when_the_model_fails(settings):
    chunks = [make_chunk(i) for i in range(1, 6)]
    assert ids(rerank(settings, "q", chunks, llm=BrokenLLM())) == [1, 2, 3]


def test_rerank_mode_none_makes_no_model_call(settings):
    settings.rerank_mode = "none"
    chunks = [make_chunk(i) for i in range(1, 6)]
    llm = FakeLLM("[5, 4, 3]")

    assert ids(rerank(settings, "q", chunks, llm=llm)) == [1, 2, 3]
    assert llm.prompts == []


def test_rerank_of_nothing_is_nothing(settings):
    assert rerank(settings, "q", [], llm=FakeLLM()) == []
