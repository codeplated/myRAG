import json

import psycopg2
import pytest
from fastapi.testclient import TestClient

from app import main
from rag_pipeline.llm import LLMError

PDF_BYTES = b"%PDF-1.4\n% a tiny fake pdf\n"


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(main.app)


@pytest.fixture
def ingested(monkeypatch):
    """Record the files that would be ingested instead of really ingesting them."""
    paths = []

    def fake_ingest(path):
        paths.append(path)
        return {"ok": True, "book_id": path.stem, "chunks": 7, "error": None}

    monkeypatch.setattr(main, "ingest_single_file", fake_ingest)
    return paths


# --- file names -----------------------------------------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        ("report.pdf", "report.pdf"),
        ("Flight Manual v2.PDF", "Flight Manual v2.PDF"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\evil.pdf", "evil.pdf"),
        ("/absolute/path/doc.pdf", "doc.pdf"),
        ("we<ird>:na|me?.pdf", "we_ird__na_me_.pdf"),
        ("notes.txt", None),
        (".pdf", None),
        ("", None),
        (None, None),
    ],
)
def test_safe_pdf_name(given, expected):
    assert main.safe_pdf_name(given) == expected


# --- upload ---------------------------------------------------------------


def test_upload_stores_and_ingests_a_pdf(client, settings, ingested):
    response = client.post("/upload", files={"files": ("manual.pdf", PDF_BYTES, "application/pdf")})

    assert response.status_code == 200
    assert response.json()["results"] == [
        {"filename": "manual.pdf", "ok": True, "book_id": "manual", "chunks": 7, "error": None}
    ]
    assert (settings.books_dir / "manual.pdf").read_bytes() == PDF_BYTES
    assert ingested == [settings.books_dir / "manual.pdf"]


def test_upload_cannot_write_outside_the_books_folder(client, settings, ingested, tmp_path):
    response = client.post("/upload", files={"files": ("../../escaped.pdf", PDF_BYTES, "application/pdf")})

    assert response.json()["results"][0]["filename"] == "escaped.pdf"
    assert (settings.books_dir / "escaped.pdf").exists()
    assert not (tmp_path.parent / "escaped.pdf").exists()
    assert not (tmp_path / "escaped.pdf").exists()


def test_upload_rejects_files_that_are_not_pdfs(client, ingested):
    by_name = client.post("/upload", files={"files": ("notes.txt", b"hello", "text/plain")})
    by_content = client.post("/upload", files={"files": ("fake.pdf", b"<html>not a pdf</html>", "application/pdf")})

    assert by_name.json()["results"][0]["error"] == "Only PDF files are accepted"
    assert by_content.json()["results"][0]["error"] == "The file is not a valid PDF"
    assert ingested == []


def test_upload_rejects_files_over_the_size_limit(client, settings, ingested):
    too_big = PDF_BYTES + b"0" * (settings.max_upload_mb * 1024 * 1024)
    response = client.post("/upload", files={"files": ("big.pdf", too_big, "application/pdf")})

    assert "larger than 1 MB" in response.json()["results"][0]["error"]
    assert ingested == []


def test_upload_reports_each_file_separately(client, ingested):
    response = client.post(
        "/upload",
        files=[
            ("files", ("good.pdf", PDF_BYTES, "application/pdf")),
            ("files", ("bad.txt", b"x", "text/plain")),
        ],
    )

    assert [r["ok"] for r in response.json()["results"]] == [True, False]


# --- query ----------------------------------------------------------------


def test_query_returns_answer_sources_and_timings(client, monkeypatch):
    def fake_run(settings, query, mode=None):
        return {
            "answer": f"Answer to: {query} ({mode})",
            "sources": [{"number": 1, "book_id": "manual", "page_start": 3, "page_end": 3, "content_preview": "x"}],
            "timings": {"retrieval_seconds": 0.1},
        }

    monkeypatch.setattr(main, "run_query", fake_run)
    response = client.post("/query", json={"query": "What is the fuel limit?", "mode": "keyword"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer to: What is the fuel limit? (keyword)"
    assert body["sources"][0]["number"] == 1
    assert body["timings"] == {"retrieval_seconds": 0.1}


@pytest.mark.parametrize("payload", [{"query": ""}, {}, {"query": "x", "mode": "magic"}, {"query": "x" * 2001}])
def test_query_rejects_invalid_input(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_model_failure_becomes_a_clear_502(client, monkeypatch):
    def failing(settings, query, mode=None):
        raise LLMError("Ollama is not reachable at http://localhost:11434. Is it running?")

    monkeypatch.setattr(main, "run_query", failing)
    response = client.post("/query", json={"query": "hi"})

    assert response.status_code == 502
    assert "Ollama is not reachable" in response.json()["detail"]


def test_database_failure_becomes_a_clear_503_without_internal_details(client, monkeypatch):
    def failing(settings, query, mode=None):
        raise psycopg2.OperationalError("connection to server at 10.0.0.5 failed: password=secret")

    monkeypatch.setattr(main, "run_query", failing)
    response = client.post("/query", json={"query": "hi"})

    assert response.status_code == 503
    assert "database is not reachable" in response.json()["detail"]
    assert "secret" not in response.text


def read_events(response) -> list[dict]:
    return [json.loads(block[6:]) for block in response.text.split("\n\n") if block.startswith("data: ")]


def test_stream_sends_server_sent_events(client, monkeypatch):
    def fake_stream(settings, query, mode=None):
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": "Hel"}
        yield {"type": "token", "text": "lo"}
        yield {"type": "done", "timings": {}}

    monkeypatch.setattr(main, "stream_query", fake_stream)
    response = client.post("/query/stream", json={"query": "hi"})

    assert response.headers["content-type"].startswith("text/event-stream")
    events = read_events(response)
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello"


def test_stream_reports_a_failure_as_an_error_event(client, monkeypatch):
    def failing_stream(settings, query, mode=None):
        yield {"type": "sources", "sources": []}
        raise LLMError("The model declined to answer this request.")

    monkeypatch.setattr(main, "stream_query", failing_stream)
    events = read_events(client.post("/query/stream", json={"query": "hi"}))

    assert events[-1] == {"type": "error", "message": "The model declined to answer this request."}


# --- status ---------------------------------------------------------------


def test_health_reports_an_unreachable_database_without_failing(client, monkeypatch):
    def no_database(_settings):
        raise psycopg2.OperationalError("down")

    monkeypatch.setattr(main, "db_connection", no_database)
    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["database"] == "unreachable"
    assert body["llm_provider"] == "ollama"
    assert body["llm_model"] == "qwen3:8b"


def test_index_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "myRAG" in response.text
