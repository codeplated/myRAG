"""Query pipeline: retrieve -> rerank -> build prompt -> LLM -> answer with numbered sources."""
from __future__ import annotations

import time
from typing import Any, Iterator, List

from .config import Settings
from .llm import LLM, get_llm
from .prompts import format_prompt, get_template
from .reranker import rerank
from .retrieval import RetrievedChunk, retrieve

SYSTEM_PROMPT = (
    "You answer questions about the user's documents. Use only the sources you are given, "
    "cite them, and say so plainly when the sources do not contain the answer."
)

NO_RESULTS_ANSWER = "I could not find anything about this in the ingested documents."


def page_label(chunk: RetrievedChunk) -> str:
    if chunk.page_start == chunk.page_end:
        return f"page {chunk.page_start}"
    return f"pages {chunk.page_start}-{chunk.page_end}"


def build_context(chunks: List[RetrievedChunk]) -> str:
    """Number the chunks so the model can cite them as [1], [2], ..."""
    parts = []
    for number, c in enumerate(chunks, start=1):
        parts.append(f"[{number}] {c.book_id}, {page_label(c)}\n{c.content}")
    return "\n\n---\n\n".join(parts)


def build_sources(chunks: List[RetrievedChunk]) -> List[dict]:
    """Source list for the API response. `number` matches the citation in the answer."""
    return [{"number": number, **c.to_source()} for number, c in enumerate(chunks, start=1)]


def find_chunks(
    settings: Settings, query: str, llm: LLM, mode: str | None = None
) -> tuple[List[RetrievedChunk], dict[str, float]]:
    """Retrieve and rerank. Returns the final chunks and how long each step took."""
    timings: dict[str, float] = {}
    start = time.perf_counter()
    candidates = retrieve(settings, query, mode=mode)
    timings["retrieval_seconds"] = round(time.perf_counter() - start, 3)

    start = time.perf_counter()
    chunks = rerank(settings, query, candidates, llm=llm)
    timings["rerank_seconds"] = round(time.perf_counter() - start, 3)
    return chunks, timings


def build_prompt(settings: Settings, query: str, chunks: List[RetrievedChunk]) -> str:
    return format_prompt(
        get_template(settings),
        context=build_context(chunks),
        query=query,
        citation_instruction=settings.citation_instruction,
    )


def stream_query(
    settings: Settings, query: str, llm: LLM | None = None, mode: str | None = None
) -> Iterator[dict[str, Any]]:
    """Run the pipeline and yield events as they happen.

    Events: {"type": "sources", "sources": [...]} once, then {"type": "token",
    "text": "..."} for each piece of the answer, then {"type": "done", "timings": {...}}.
    """
    llm = llm or get_llm(settings)
    chunks, timings = find_chunks(settings, query, llm, mode=mode)
    yield {"type": "sources", "sources": build_sources(chunks)}

    start = time.perf_counter()
    if not chunks:
        # Nothing was retrieved, so there is nothing to ground an answer on.
        yield {"type": "token", "text": NO_RESULTS_ANSWER}
    else:
        for text in llm.stream(build_prompt(settings, query, chunks), system=SYSTEM_PROMPT):
            yield {"type": "token", "text": text}
    timings["generation_seconds"] = round(time.perf_counter() - start, 3)
    yield {"type": "done", "timings": timings}


def run_query(
    settings: Settings, query: str, llm: LLM | None = None, mode: str | None = None
) -> dict[str, Any]:
    """Run the full pipeline and return {"answer", "sources", "timings"}."""
    answer_parts: List[str] = []
    result: dict[str, Any] = {"sources": [], "timings": {}}
    for event in stream_query(settings, query, llm=llm, mode=mode):
        if event["type"] == "sources":
            result["sources"] = event["sources"]
        elif event["type"] == "token":
            answer_parts.append(event["text"])
        elif event["type"] == "done":
            result["timings"] = event["timings"]
    result["answer"] = "".join(answer_parts).strip()
    return result
