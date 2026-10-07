"""Retrieval: vector search (pgvector) and keyword search (Postgres full-text), merged with RRF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from psycopg2.extras import RealDictCursor

from .config import SEARCH_MODES, Settings
from .db_pgvector import db_connection
from .embeddings_ollama import embed_query


@dataclass
class RetrievedChunk:
    id: int
    book_id: str
    chunk_id: int
    book_title: str | None
    page_start: int | None
    page_end: int | None
    content: str

    def to_source(self, content_preview_len: int = 300) -> dict:
        return {
            "book_id": self.book_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content_preview": (self.content or "")[:content_preview_len],
        }


def _row_to_chunk(row: dict) -> RetrievedChunk:
    return RetrievedChunk(
        id=row["id"],
        book_id=row["book_id"],
        chunk_id=row["chunk_id"],
        book_title=row.get("book_title"),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        content=row.get("content") or "",
    )


def _embedding_to_vector_str(embedding: List[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def dense_search(conn, settings: Settings, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
    """Vector search: the chunks whose embeddings are closest to the query embedding."""
    vec_str = _embedding_to_vector_str(query_embedding)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # The HNSW index returns at most ef_search rows, so it must not be below top_k.
        cur.execute(f"SET LOCAL hnsw.ef_search = {min(max(int(top_k), 40), 1000)}")
        cur.execute(
            """
            SELECT id, book_id, chunk_id, book_title, page_start, page_end, content
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec_str, top_k),
        )
        rows = cur.fetchall()
    return [_row_to_chunk(dict(r)) for r in rows]


def sparse_search(conn, settings: Settings, query: str, top_k: int) -> List[RetrievedChunk]:
    """Keyword search with Postgres full-text search.

    plainto_tsquery joins the query words with AND, so a normal question would
    only match chunks that contain every word. The words are joined with OR
    instead and ts_rank puts the chunks that match the most words first.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH q AS (
                SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS tsq
            )
            SELECT id, book_id, chunk_id, book_title, page_start, page_end, content
            FROM documents, q
            WHERE content_tsv @@ q.tsq
            ORDER BY ts_rank(content_tsv, q.tsq, 1) DESC, id
            LIMIT %s
            """,
            (query, top_k),
        )
        rows = cur.fetchall()
    return [_row_to_chunk(dict(r)) for r in rows]


def rrf_merge(
    dense: List[RetrievedChunk],
    sparse: List[RetrievedChunk],
    k: int = 60,
    sparse_weight: float = 1.0,
) -> List[RetrievedChunk]:
    """Reciprocal Rank Fusion. Returns merged list sorted by RRF score (desc).

    A chunk scores 1 / (k + rank) in each list it appears in. Chunks found by
    both searches rise to the top without having to compare the two different
    score scales. sparse_weight scales the keyword list: below 1.0 a keyword
    hit counts less than a vector hit at the same rank.
    """
    scores: dict[int, float] = {}
    id_to_chunk: dict[int, RetrievedChunk] = {}

    for ranked, weight in ((dense, 1.0), (sparse, sparse_weight)):
        for rank, c in enumerate(ranked):
            scores[c.id] = scores.get(c.id, 0.0) + weight / (k + rank + 1)
            id_to_chunk[c.id] = c

    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [id_to_chunk[i] for i in sorted_ids]


def retrieve(settings: Settings, query: str, mode: str | None = None) -> List[RetrievedChunk]:
    """Return the top RETRIEVAL_TOP_K chunks for a query.

    mode is "hybrid" (default from SEARCH_MODE), "vector" or "keyword".
    """
    mode = mode or settings.search_mode
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unknown search mode '{mode}'. Use one of {', '.join(SEARCH_MODES)}.")
    top_k = settings.retrieval_top_k

    dense: List[RetrievedChunk] = []
    sparse: List[RetrievedChunk] = []
    query_embedding = embed_query(settings, query) if mode in ("hybrid", "vector") else None

    with db_connection(settings) as conn:
        if query_embedding is not None:
            dense = dense_search(conn, settings, query_embedding, top_k=top_k)
        if mode in ("hybrid", "keyword"):
            sparse = sparse_search(conn, settings, query, top_k=top_k)

    if mode == "vector":
        return dense
    if mode == "keyword":
        return sparse
    return rrf_merge(dense, sparse, sparse_weight=settings.keyword_weight)[:top_k]


def hybrid_retrieve(settings: Settings, query: str) -> List[RetrievedChunk]:
    """Vector plus keyword search, merged with RRF."""
    return retrieve(settings, query, mode="hybrid")
