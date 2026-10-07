"""Tests against a real pgvector database.

They are skipped unless RAG_TEST_DATABASE=1 is set. They create and drop the
`documents` table, so they only ever run in a separate throwaway database
(RAG_TEST_PGDATABASE, default "rag_test"), never in the one that holds real data.

    docker compose up -d pgvector
    docker compose exec pgvector createdb -U postgres rag_test
    RAG_TEST_DATABASE=1 pytest -m integration
"""
import os
from pathlib import Path

import pytest

from rag_pipeline import retrieval
from rag_pipeline.chunking import Chunk
from rag_pipeline.db_pgvector import (
    SchemaMismatchError,
    db_connection,
    delete_by_book_id,
    ensure_schema,
    insert_embeddings,
    list_books,
    reset_schema,
)
from rag_pipeline.embeddings_ollama import EmbeddedChunk
from rag_pipeline.retrieval import dense_search, retrieve, sparse_search

from .conftest import make_settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RAG_TEST_DATABASE") != "1", reason="set RAG_TEST_DATABASE=1 to run"),
]

# (book, page, text, embedding). The vectors are made up so that the nearest neighbour is obvious.
ROWS = [
    ("manual", 1, "The maximum fuel load is 500 kilograms.", [1.0, 0.0, 0.0, 0.0]),
    ("manual", 2, "Set the flaps to 15 degrees for landing.", [0.0, 1.0, 0.0, 0.0]),
    ("history", 1, "The Wright brothers first flew in 1903.", [0.0, 0.0, 1.0, 0.0]),
]


@pytest.fixture
def db(tmp_path):
    settings = make_settings(
        tmp_path,
        pg_host=os.getenv("PGHOST", "localhost"),
        pg_port=int(os.getenv("PGPORT", "5432")),
        pg_user=os.getenv("PGUSER", "postgres"),
        pg_password=os.getenv("PGPASSWORD", "postgres"),
        pg_database=os.getenv("RAG_TEST_PGDATABASE", "rag_test"),
        embedding_dim=4,
    )
    with db_connection(settings) as conn:
        reset_schema(conn)
        ensure_schema(conn, settings)
        chunks = [
            EmbeddedChunk(Chunk(book, i, text, page, page, i), vector)
            for i, (book, page, text, vector) in enumerate(ROWS, start=1)
        ]
        insert_embeddings(conn, chunks)
        yield settings, conn
        reset_schema(conn)


def test_schema_can_be_created_twice(db):
    settings, conn = db
    ensure_schema(conn, settings)


def test_vector_search_returns_the_nearest_chunk_first(db):
    settings, conn = db
    result = dense_search(conn, settings, [0.1, 0.9, 0.0, 0.0], top_k=3)

    assert "flaps" in result[0].content
    assert len(result) == 3


def test_keyword_search_matches_a_whole_question(db):
    settings, conn = db
    # Most words of the question are not in the text. An AND query would find nothing.
    result = sparse_search(conn, settings, "How many kilograms of fuel can the aircraft carry?", top_k=3)

    assert result, "the words are joined with OR, so partial matches are found"
    assert "fuel" in result[0].content


def test_keyword_search_uses_word_stems(db):
    settings, conn = db
    result = sparse_search(conn, settings, "flying brother", top_k=3)

    assert [c.book_id for c in result] == ["history"]


def test_keyword_search_with_only_stop_words_finds_nothing(db):
    settings, conn = db
    assert sparse_search(conn, settings, "the of and", top_k=3) == []


def test_hybrid_retrieve_end_to_end(db, monkeypatch):
    settings, _conn = db
    monkeypatch.setattr(retrieval, "embed_query", lambda s, q: [0.9, 0.1, 0.0, 0.0])

    result = retrieve(settings, "fuel load", mode="hybrid")

    assert "fuel" in result[0].content, "found by both searches, so it ranks first"


def test_documents_are_listed_with_chunk_and_page_counts(db):
    _settings, conn = db
    assert list_books(conn) == [
        {"book_id": "history", "chunks": 1, "pages": 1},
        {"book_id": "manual", "chunks": 2, "pages": 2},
    ]


def test_deleting_a_document_removes_only_its_chunks(db):
    _settings, conn = db

    assert delete_by_book_id(conn, "manual") == 2
    assert [b["book_id"] for b in list_books(conn)] == ["history"]


def test_changed_embedding_size_is_detected(db):
    settings, conn = db
    other = make_settings(Path(settings.books_dir).parent, pg_database=settings.pg_database, embedding_dim=8)
    other.pg_host, other.pg_port = settings.pg_host, settings.pg_port
    other.pg_user, other.pg_password = settings.pg_user, settings.pg_password

    with pytest.raises(SchemaMismatchError, match="--reset"):
        ensure_schema(conn, other)


def test_keyword_search_also_matches_the_document_title(db):
    settings, conn = db
    # "manual" is only in the file name, not in any chunk text.
    result = sparse_search(conn, settings, "manual", top_k=5)

    assert sorted(c.book_id for c in result) == ["manual", "manual"]
