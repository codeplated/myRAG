"""Reranker: one LLM call puts the retrieved chunks in order of relevance.

The first version scored every chunk in its own LLM call, which meant 50 calls
per question. Here the model sees all candidates at once and returns their
order, so reranking costs a single call.
"""
from __future__ import annotations

import logging
import re
from typing import List

from .config import Settings
from .llm import LLM, LLMError, get_llm
from .retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Each passage is shortened in the prompt so that all candidates fit comfortably.
RERANK_DOC_MAX_CHARS = 700

_SYSTEM = (
    "You rank text passages by how well they answer a question. "
    "Reply with a JSON array of passage numbers only, most relevant first."
)


def build_rerank_prompt(query: str, chunks: List[RetrievedChunk], top_n: int) -> str:
    passages = []
    for number, chunk in enumerate(chunks, start=1):
        snippet = " ".join((chunk.content or "").split())[:RERANK_DOC_MAX_CHARS]
        passages.append(f"[{number}] {snippet}")
    # The question stands before the passages, so the model reads them knowing what to look for,
    # and again after them. In the evaluation, asking only at the end ranked clearly worse.
    return (
        f"Question: {query}\n\n"
        "Passages:\n" + "\n\n".join(passages) + "\n\n"
        f"Question again: {query}\n\n"
        f"Return the numbers of the {top_n} passages that best answer the question, "
        "most relevant first, as a JSON array such as [3, 1, 7]. Return only the array."
    )


def parse_ranking(response_text: str, num_passages: int) -> List[int]:
    """Read the model's ranking and return valid, unique, zero-based passage indexes."""
    arrays = re.findall(r"\[([\d,\s]+)\]", response_text or "")
    # The answer is the last array in the text; anything before it is the model thinking aloud.
    numbers = re.findall(r"\d+", arrays[-1]) if arrays else []
    order: List[int] = []
    for raw in numbers:
        index = int(raw) - 1
        if 0 <= index < num_passages and index not in order:
            order.append(index)
    return order


def rerank(
    settings: Settings,
    query: str,
    chunks: List[RetrievedChunk],
    llm: LLM | None = None,
) -> List[RetrievedChunk]:
    """Return the top RERANK_TOP_K chunks.

    With RERANK_MODE=none the retrieval order is kept. If the model call fails
    or its reply cannot be read, the retrieval order is used as well, so a
    reranker problem never breaks a query.
    """
    top_n = min(settings.rerank_top_k, len(chunks))
    if settings.rerank_mode == "none" or len(chunks) <= 1:
        return chunks[:top_n]

    candidates = chunks[: settings.rerank_candidates]
    try:
        reply = (llm or get_llm(settings)).generate(
            build_rerank_prompt(query, candidates, top_n), system=_SYSTEM
        )
    except LLMError as exc:
        logger.warning("Reranker call failed, keeping retrieval order: %s", exc)
        return candidates[:top_n]

    order = parse_ranking(reply, len(candidates))
    if not order:
        logger.warning("Could not read a ranking from the reranker reply, keeping retrieval order")
    ranked = [candidates[i] for i in order]
    ranked += [c for i, c in enumerate(candidates) if i not in order]
    return ranked[:top_n]
