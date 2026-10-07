"""FastAPI app: upload PDFs, ask questions, serve the web page.

Run from the project root with PYTHONPATH=src (see the Makefile):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator, Literal

import psycopg2
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_pipeline.config import get_settings
from rag_pipeline.db_pgvector import SchemaMismatchError, db_connection, list_books
from rag_pipeline.embeddings_ollama import EmbeddingError
from rag_pipeline.ingest_books import ingest_single_file
from rag_pipeline.llm import LLMError
from rag_pipeline.query import run_query, stream_query

logger = logging.getLogger("myrag.api")

app = FastAPI(title="myRAG", version="0.2.0", description="Question answering over your PDF documents.")

_TEMPLATES = Path(__file__).resolve().parent / "templates"
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["hybrid", "vector", "keyword"] | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    timings: dict[str, float] = {}


def safe_pdf_name(filename: str | None) -> str | None:
    """Return a file name that is safe to store, or None if it is not a usable PDF name.

    The name comes from the browser and must not be trusted: "../../etc/x.pdf"
    would otherwise be written outside the books folder.
    """
    name = Path((filename or "").replace("\\", "/")).name
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    if not name.lower().endswith(".pdf") or len(name) <= len(".pdf"):
        return None
    return name


def _user_message(exc: Exception) -> tuple[int, str]:
    """Translate an internal error into a status code and a message that is safe to show."""
    if isinstance(exc, LLMError):
        return 502, str(exc)
    if isinstance(exc, (EmbeddingError, SchemaMismatchError)):
        return 503, str(exc)
    if isinstance(exc, psycopg2.OperationalError):
        return 503, "The database is not reachable. Is the pgvector container running?"
    if isinstance(exc, psycopg2.errors.UndefinedTable):
        return 409, "No documents have been ingested yet. Upload a PDF first."
    logger.exception("Unexpected error")
    return 500, "Something went wrong while handling the request."


@app.exception_handler(LLMError)
@app.exception_handler(EmbeddingError)
@app.exception_handler(SchemaMismatchError)
@app.exception_handler(psycopg2.OperationalError)
@app.exception_handler(psycopg2.errors.UndefinedTable)
def handle_known_error(_request: Request, exc: Exception) -> JSONResponse:
    """Known failures get a clear message. Anything else stays a plain 500 without details."""
    status, message = _user_message(exc)
    return JSONResponse(status_code=status, content={"detail": message})


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_TEMPLATES / "index.html", media_type="text/html")


@app.get("/health")
def health():
    """Report whether the database answers, and which model is configured."""
    settings = get_settings()
    database = "ok"
    try:
        with db_connection(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except psycopg2.Error:
        database = "unreachable"
    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "embedding_model": settings.embedding_model,
        "search_mode": settings.search_mode,
        "rerank_mode": settings.rerank_mode,
    }


@app.get("/documents")
def documents():
    """List the ingested documents."""
    settings = get_settings()
    with db_connection(settings) as conn:
        return {"documents": list_books(conn)}


@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    settings = get_settings()
    settings.books_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    results = []
    for f in files:
        shown_name = f.filename or "(unknown)"
        name = safe_pdf_name(f.filename)
        if name is None:
            results.append({"filename": shown_name, "ok": False, "error": "Only PDF files are accepted"})
            continue
        content = f.file.read(max_bytes + 1)
        if len(content) > max_bytes:
            results.append(
                {"filename": shown_name, "ok": False, "error": f"File is larger than {settings.max_upload_mb} MB"}
            )
            continue
        if not content.startswith(b"%PDF-"):
            results.append({"filename": shown_name, "ok": False, "error": "The file is not a valid PDF"})
            continue

        dest = settings.books_dir / name
        dest.write_bytes(content)
        result = ingest_single_file(dest)
        results.append(
            {
                "filename": name,
                "ok": result["ok"],
                "book_id": result["book_id"],
                "chunks": result["chunks"],
                "error": result.get("error"),
            }
        )
    return {"results": results}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    out = run_query(get_settings(), body.query, mode=body.mode)
    return QueryResponse(answer=out["answer"], sources=out["sources"], timings=out["timings"])


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
def query_stream_endpoint(body: QueryRequest):
    """Same as /query, but sends the answer as server-sent events while it is written."""

    def events() -> Iterator[str]:
        try:
            for event in stream_query(get_settings(), body.query, mode=body.mode):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - the stream has started, so report the error in-band
            _status, message = _user_message(exc)
            yield _sse({"type": "error", "message": message})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
