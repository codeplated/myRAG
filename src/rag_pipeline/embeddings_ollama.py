from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

import requests

from .chunking import Chunk, text_for_search
from .config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """The embedding model could not be reached or returned something unexpected."""


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: List[float]


def _embed(settings: Settings, inputs: List[str], timeout: int) -> List[List[float]]:
    """Send texts to Ollama's /api/embed endpoint and return one vector per text."""
    url = settings.ollama_host.rstrip("/") + "/api/embed"
    model = settings.embedding_model
    try:
        resp = requests.post(url, json={"model": model, "input": inputs}, timeout=timeout)
    except requests.RequestException as exc:
        raise EmbeddingError(f"Ollama is not reachable at {settings.ollama_host}. Is it running?") from exc
    if resp.status_code == 404:
        raise EmbeddingError(f"Embedding model '{model}' is not available. Run: ollama pull {model}")
    resp.raise_for_status()

    data = resp.json()
    vectors = data.get("embeddings") or data.get("data")
    if vectors is None or len(vectors) != len(inputs):
        raise EmbeddingError(f"Unexpected embeddings response for {len(inputs)} inputs: {str(data)[:200]}")
    if vectors and len(vectors[0]) != settings.embedding_dim:
        raise EmbeddingError(
            f"Model '{model}' returns {len(vectors[0])}-dimensional vectors "
            f"but EMBEDDING_DIM is {settings.embedding_dim}."
        )
    return [list(v) for v in vectors]


def embed_chunks(chunks: List[Chunk], settings: Settings, batch_size: int = 8) -> List[EmbeddedChunk]:
    embedded: List[EmbeddedChunk] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        start = time.time()
        vectors = _embed(settings, [text_for_search(c) for c in batch], timeout=600)
        embedded.extend(EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(batch, vectors, strict=True))
        logger.info(
            "Embedded batch %d with %d chunks in %.2fs",
            i // batch_size + 1,
            len(batch),
            time.time() - start,
        )

    return embedded


def embed_query(settings: Settings, query: str) -> List[float]:
    """Embed a single query string. Returns one embedding vector."""
    return _embed(settings, [query], timeout=120)[0]
