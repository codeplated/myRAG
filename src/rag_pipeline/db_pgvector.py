from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, List

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from .chunking import document_title, text_for_search
from .config import Settings
from .embeddings_ollama import EmbeddedChunk


@dataclass
class DbStats:
    inserted_rows: int


class SchemaMismatchError(RuntimeError):
    """The documents table was created for a different embedding size."""


def get_connection(settings: Settings):
    return psycopg2.connect(settings.pg_dsn, connect_timeout=5)


@contextmanager
def db_connection(settings: Settings) -> Iterator:
    """Open a connection and always close it again."""
    conn = get_connection(settings)
    try:
        yield conn
    finally:
        conn.close()


def _stored_embedding_dim(cur) -> int | None:
    """Vector size of the existing documents table, or None if there is no table yet."""
    cur.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        WHERE a.attrelid = to_regclass('documents') AND a.attname = 'embedding'
        """
    )
    row = cur.fetchone()
    return row[0] if row else None


def ensure_schema(conn, settings: Settings) -> None:
    dim = settings.embedding_dim
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        stored_dim = _stored_embedding_dim(cur)
        if stored_dim is not None and stored_dim != dim:
            conn.rollback()
            raise SchemaMismatchError(
                f"The documents table stores {stored_dim}-dimensional vectors but EMBEDDING_DIM is {dim}. "
                "Fix EMBEDDING_DIM, or re-create the table with: "
                "python -m rag_pipeline.ingest_books --reset"
            )

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                book_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                book_title TEXT,
                page_start INTEGER,
                page_end INTEGER,
                position INTEGER,
                content TEXT NOT NULL,
                embedding VECTOR({dim}) NOT NULL,
                content_tsv TSVECTOR,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR;")
        cur.execute(
            """
            UPDATE documents SET content_tsv = to_tsvector('english', COALESCE(content, ''))
            WHERE content_tsv IS NULL AND content IS NOT NULL;
            """
        )
        # Keyword search index.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_tsv ON documents USING GIN(content_tsv);"
        )
        # Approximate nearest-neighbour index for vector search (cosine distance).
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents USING hnsw (embedding vector_cosine_ops);
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_book_id ON documents (book_id);")
    conn.commit()


def reset_schema(conn) -> None:
    """Drop the documents table. All stored chunks are lost and must be ingested again."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS documents;")
    conn.commit()


def delete_by_book_id(conn, book_id: str) -> int:
    """Delete all rows for the given book_id. Returns number of rows deleted."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE book_id = %s", (book_id,))
        deleted = cur.rowcount
    conn.commit()
    return deleted


def list_books(conn) -> List[dict]:
    """One row per ingested document: id, number of chunks and number of pages."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT to_regclass('documents') IS NOT NULL AS exists")
        if not cur.fetchone()["exists"]:
            return []
        cur.execute(
            """
            SELECT book_id, COUNT(*) AS chunks, MAX(page_end) AS pages
            FROM documents
            GROUP BY book_id
            ORDER BY book_id
            """
        )
        return [dict(r) for r in cur.fetchall()]


def insert_embeddings(conn, embedded_chunks: Iterable[EmbeddedChunk]) -> DbStats:
    rows: List[tuple] = []
    for ec in embedded_chunks:
        c = ec.chunk
        rows.append(
            (
                c.book_id,
                c.chunk_id,
                document_title(c.book_id),
                c.page_start,
                c.page_end,
                c.position,
                c.text,
                ec.embedding,
                text_for_search(c),  # keyword index covers the title as well
            )
        )

    if not rows:
        return DbStats(inserted_rows=0)

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO documents (
                book_id,
                chunk_id,
                book_title,
                page_start,
                page_end,
                position,
                content,
                embedding,
                content_tsv
            )
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s::vector, to_tsvector('english', %s))",
        )
    conn.commit()
    return DbStats(inserted_rows=len(rows))
