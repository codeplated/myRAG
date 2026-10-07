from contextlib import contextmanager

import pytest

from rag_pipeline import retrieval
from rag_pipeline.retrieval import retrieve, rrf_merge

from .conftest import make_chunk


def ids(chunks):
    return [c.id for c in chunks]


def test_rrf_puts_chunks_found_by_both_searches_first():
    dense = [make_chunk(1), make_chunk(2), make_chunk(3)]
    sparse = [make_chunk(3), make_chunk(4)]

    merged = rrf_merge(dense, sparse)

    assert merged[0].id == 3, "chunk 3 is in both lists and should win"
    assert sorted(ids(merged)) == [1, 2, 3, 4], "every chunk appears exactly once"


def test_rrf_keeps_rank_order_within_one_list():
    merged = rrf_merge([make_chunk(1), make_chunk(2), make_chunk(3)], [])
    assert ids(merged) == [1, 2, 3]


def test_rrf_weight_decides_which_search_wins_a_tie():
    dense, sparse = [make_chunk(1)], [make_chunk(2)]

    assert ids(rrf_merge(dense, sparse, sparse_weight=0.3)) == [1, 2]
    assert ids(rrf_merge(dense, sparse, sparse_weight=3.0)) == [2, 1]


def test_low_keyword_weight_still_lifts_chunks_found_by_both_searches():
    dense = [make_chunk(1), make_chunk(2), make_chunk(3)]
    sparse = [make_chunk(3)]

    assert ids(rrf_merge(dense, sparse, sparse_weight=0.3))[0] == 3


def test_rrf_handles_empty_input():
    assert rrf_merge([], []) == []


@pytest.fixture
def fake_search(monkeypatch):
    """Replace the database and the embedding model with fixed results."""
    calls = {"embed": 0, "dense": 0, "sparse": 0}

    @contextmanager
    def no_db(_settings):
        yield object()

    def embed(_settings, _query):
        calls["embed"] += 1
        return [0.0, 0.0, 0.0, 1.0]

    def dense(_conn, _settings, _embedding, top_k):
        calls["dense"] += 1
        return [make_chunk(1), make_chunk(2)]

    def sparse(_conn, _settings, _query, top_k):
        calls["sparse"] += 1
        return [make_chunk(2), make_chunk(3)]

    monkeypatch.setattr(retrieval, "db_connection", no_db)
    monkeypatch.setattr(retrieval, "embed_query", embed)
    monkeypatch.setattr(retrieval, "dense_search", dense)
    monkeypatch.setattr(retrieval, "sparse_search", sparse)
    return calls


def test_hybrid_mode_merges_both_searches(settings, fake_search):
    result = retrieve(settings, "question", mode="hybrid")

    assert ids(result)[0] == 2
    assert sorted(ids(result)) == [1, 2, 3]


def test_vector_mode_skips_keyword_search(settings, fake_search):
    assert ids(retrieve(settings, "question", mode="vector")) == [1, 2]
    assert fake_search["sparse"] == 0


def test_keyword_mode_needs_no_embedding(settings, fake_search):
    assert ids(retrieve(settings, "question", mode="keyword")) == [2, 3]
    assert fake_search["embed"] == 0 and fake_search["dense"] == 0


def test_default_mode_comes_from_settings(settings, fake_search):
    settings.search_mode = "keyword"
    assert ids(retrieve(settings, "question")) == [2, 3]


def test_result_is_limited_to_top_k(settings, fake_search):
    settings.retrieval_top_k = 2
    assert len(retrieve(settings, "question", mode="hybrid")) == 2


def test_unknown_mode_is_rejected(settings, fake_search):
    with pytest.raises(ValueError):
        retrieve(settings, "question", mode="magic")
