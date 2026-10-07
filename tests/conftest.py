"""Shared test helpers: default settings, a fake LLM, and a fake API server.

The fake server speaks just enough of the Ollama, OpenAI and Anthropic HTTP
APIs to run the real client code against it. No test talks to a real service.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.llm import LLM
from rag_pipeline.retrieval import RetrievedChunk


def make_settings(tmp_path, **overrides) -> Settings:
    values = dict(
        pg_host="localhost",
        pg_port=5432,
        pg_user="postgres",
        pg_password="postgres",
        pg_database="vector_db",
        ollama_host="http://localhost:11434",
        embedding_model="bge-m3",
        llm_model="qwen3:8b",
        ollama_think="false",
        ollama_num_ctx=8192,
        llm_provider="ollama",
        openai_model="gpt-test",
        openai_api_key="test-key",
        openai_base_url="",
        anthropic_model="claude-opus-5-5",
        anthropic_api_key="test-key",
        anthropic_base_url="",
        books_dir=tmp_path / "books",
        logs_dir=tmp_path / "logs",
        prompts_dir=tmp_path / "prompts",
        chunk_size=200,
        chunk_overlap=40,
        embedding_dim=4,
        search_mode="hybrid",
        keyword_weight=1.0,
        rerank_mode="llm",
        retrieval_top_k=10,
        rerank_candidates=5,
        rerank_top_k=3,
        prompt_template="default",
        citation_instruction="Cite sources as [1].",
        max_upload_mb=1,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = make_settings(tmp_path)
    s.prompts_dir.mkdir(parents=True)
    (s.prompts_dir / "default.txt").write_text(
        "{citation_instruction}\n\nSources:\n{context}\n\nQuestion: {query}\n", encoding="utf-8"
    )
    return s


def make_chunk(id: int, content: str = "", book_id: str = "book", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        id=id,
        book_id=book_id,
        chunk_id=id,
        book_title=book_id,
        page_start=page,
        page_end=page,
        content=content or f"content of chunk {id}",
    )


class FakeLLM(LLM):
    """Returns prepared replies in order and remembers the prompts it was given."""

    provider = "fake"

    def __init__(self, *replies: str) -> None:
        super().__init__("fake-model")
        self.replies = list(replies)
        self.prompts: list[str] = []

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else ""
        # Split the reply so that callers have to join several pieces.
        middle = len(reply) // 2
        yield reply[:middle]
        yield reply[middle:]


def _sse(events: list[tuple[str | None, dict | str]]) -> bytes:
    out = ""
    for name, data in events:
        if name:
            out += f"event: {name}\n"
        out += "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"
    return out.encode()


def _anthropic_stream(words: list[str], stop_reason: str) -> bytes:
    events: list[tuple[str | None, dict | str]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    for word in words:
        events.append(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": word}},
            )
        )
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return _sse(events)


def _openai_stream(words: list[str]) -> bytes:
    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    events: list[tuple[str | None, dict | str]] = [(None, chunk({"role": "assistant", "content": ""}))]
    events += [(None, chunk({"content": word})) for word in words]
    events += [(None, chunk({}, "stop")), (None, "[DONE]")]
    return _sse(events)


class FakeAPI:
    """State shared between a test and the fake server."""

    def __init__(self) -> None:
        self.words = ["Hello", " from", " the", " fake", " API"]
        self.status = 200  # set to 401 or 404 to make the next calls fail
        self.rejects_think = False  # behave like a local model that has no thinking mode
        self.stop_reason = "end_turn"
        self.embedding = [0.1, 0.2, 0.3, 0.4]
        self.requests: list[dict] = []
        self.url = ""

    @property
    def last(self) -> dict:
        return self.requests[-1]


@pytest.fixture
def fake_api() -> Iterator[FakeAPI]:
    state = FakeAPI()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep test output quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            path = self.path.split("?")[0]
            headers = {name.lower(): value for name, value in self.headers.items()}
            state.requests.append({"path": path, "headers": headers, "body": body})

            if state.rejects_think and path == "/api/chat" and "think" in body:
                error = {"error": f"\"{body['model']}\" does not support thinking"}
                self._send(400, json.dumps(error).encode(), "application/json")
            elif state.status != 200:
                error = {"type": "error", "error": {"type": "api_error", "message": "fake failure"}}
                self._send(state.status, json.dumps(error).encode(), "application/json")
            elif path == "/v1/messages":
                self._send(200, _anthropic_stream(state.words, state.stop_reason), "text/event-stream")
            elif path == "/v1/chat/completions":
                self._send(200, _openai_stream(state.words), "text/event-stream")
            elif path == "/api/chat":
                lines = [{"message": {"role": "assistant", "content": w}, "done": False} for w in state.words]
                lines.append({"message": {"role": "assistant", "content": ""}, "done": True})
                self._send(200, "\n".join(json.dumps(x) for x in lines).encode(), "application/x-ndjson")
            elif path == "/api/embed":
                vectors = [state.embedding for _ in body.get("input", [])]
                self._send(200, json.dumps({"embeddings": vectors}).encode(), "application/json")
            else:
                self._send(404, b"{}", "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
