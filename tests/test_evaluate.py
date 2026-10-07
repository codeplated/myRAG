import json

import pytest

from rag_pipeline import evaluate
from rag_pipeline.evaluate import (
    EvalQuestion,
    evaluate_config,
    first_relevant_rank,
    format_table,
    is_relevant,
    load_questions,
    parse_judgement,
    retrieval_metrics,
)

from .conftest import FakeLLM, make_chunk

QUESTION = EvalQuestion(
    id="q1",
    question="What was the probable cause?",
    book_ids=["report"],
    evidence=["loss of thrust in both engines"],
)


def test_chunk_is_relevant_when_it_contains_an_evidence_phrase():
    chunk = make_chunk(1, "The probable cause was the Loss of  thrust\nin both engines.", book_id="report")
    assert is_relevant(chunk, QUESTION), "matching ignores case, line breaks and extra spaces"


def test_chunk_from_another_document_is_not_relevant():
    chunk = make_chunk(1, "loss of thrust in both engines", book_id="other")
    assert not is_relevant(chunk, QUESTION)


def test_question_can_accept_more_than_one_document():
    both = EvalQuestion(id="q2", question="?", book_ids=["report", "summary"], evidence=["loss of thrust"])

    assert is_relevant(make_chunk(1, "a loss of thrust occurred", book_id="summary"), both)
    assert not is_relevant(make_chunk(1, "a loss of thrust occurred", book_id="news"), both)


def test_first_relevant_rank_only_counts_the_top_k():
    chunks = [
        make_chunk(1, "unrelated", book_id="report"),
        make_chunk(2, "unrelated", book_id="report"),
        make_chunk(3, "loss of thrust in both engines", book_id="report"),
    ]

    assert first_relevant_rank(chunks, QUESTION, k=5) == 3
    assert first_relevant_rank(chunks, QUESTION, k=2) is None


def test_hit_rate_and_mrr():
    # Four questions: found at rank 1, rank 2, rank 4, and not found.
    metrics = retrieval_metrics([1, 2, 4, None])

    assert metrics["hit_rate"] == pytest.approx(0.75)
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 0.25 + 0) / 4)


def test_metrics_of_no_questions_are_zero():
    assert retrieval_metrics([]) == {"hit_rate": 0.0, "mrr": 0.0}


def test_judgement_is_the_share_of_supported_claims():
    claims = [
        {"claim": "a", "supported": True},
        {"claim": "b", "supported": True},
        {"claim": "c", "supported": False},
    ]
    reply = json.dumps({"claims": claims})
    assert parse_judgement(reply) == pytest.approx(2 / 3)


def test_judgement_is_read_from_a_reply_with_extra_text():
    reply = 'Here is my check:\n```json\n{"claims": [{"claim": "a", "supported": false}]}\n```'
    assert parse_judgement(reply) == 0.0


def test_judgement_survives_broken_json():
    reply = '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false},'
    assert parse_judgement(reply) == 0.5


def test_judgement_without_claims_is_not_scored():
    assert parse_judgement('{"claims": []}') is None
    assert parse_judgement("no json here") is None


def test_questions_are_loaded_from_jsonl(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "# comment line\n"
        '{"id": 1, "question": "Q?", "book_id": "report", "evidence": ["phrase"], "reference": "A"}\n'
        "\n",
        encoding="utf-8",
    )

    questions = load_questions(path)

    assert len(questions) == 1
    assert questions[0].id == "1"
    assert questions[0].book_ids == ["report"]
    assert questions[0].evidence == ["phrase"]
    assert questions[0].reference == "A"


def test_question_without_evidence_is_rejected(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id": 1, "question": "Q?", "book_id": "report", "evidence": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no evidence"):
        load_questions(path)


def test_evaluate_config_scores_retrieval_and_faithfulness(settings, monkeypatch):
    relevant = make_chunk(2, "loss of thrust in both engines", book_id="report")
    monkeypatch.setattr(
        evaluate, "retrieve", lambda cfg, query, mode=None: [make_chunk(1, "unrelated", book_id="report"), relevant]
    )
    judge_reply = json.dumps({"claims": [{"claim": "x", "supported": True}, {"claim": "y", "supported": False}]})
    # Replies in order: reranker, answer, judge.
    llm = FakeLLM("[2, 1]", "Both engines lost thrust [1].", judge_reply)

    run = evaluate_config(settings, [QUESTION], "hybrid+rerank", k=5, llm=llm, with_answers=True)

    assert run["metrics"]["hit_rate"] == 1.0
    assert run["metrics"]["mrr"] == 1.0, "the reranker moved the relevant chunk to rank 1"
    assert run["metrics"]["faithfulness"] == 0.5
    assert run["questions"][0]["answer"] == "Both engines lost thrust [1]."


def test_evaluate_config_without_reranker_keeps_retrieval_order(settings, monkeypatch):
    relevant = make_chunk(2, "loss of thrust in both engines", book_id="report")
    monkeypatch.setattr(
        evaluate, "retrieve", lambda cfg, query, mode=None: [make_chunk(1, "unrelated", book_id="report"), relevant]
    )
    llm = FakeLLM("[2, 1]")

    run = evaluate_config(settings, [QUESTION], "hybrid", k=5, llm=llm, with_answers=False)

    assert run["metrics"]["mrr"] == 0.5
    assert "faithfulness" not in run["metrics"]
    assert llm.prompts == [], "no model call without reranking or answers"


def test_table_lists_every_configuration():
    runs = [
        {"config": "vector", "metrics": {"hit_rate": 0.8, "mrr": 0.6, "avg_search_seconds": 0.05}},
        {
            "config": "hybrid+rerank",
            "metrics": {"hit_rate": 0.9, "mrr": 0.75, "faithfulness": 0.95, "avg_search_seconds": 2.0},
        },
    ]
    table = format_table(runs, k=5)

    assert "| vector | 0.80 | 0.60 | not measured | 0.05 s |" in table
    assert "| hybrid+rerank | 0.90 | 0.75 | 0.95 | 2.00 s |" in table
    assert "Hit rate@5" in table
