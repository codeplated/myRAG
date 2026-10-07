from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load values from .env if present

SEARCH_MODES = ("hybrid", "vector", "keyword")
RERANK_MODES = ("llm", "none")
LLM_PROVIDERS = ("ollama", "openai", "anthropic")


@dataclass
class Settings:
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_database: str

    ollama_host: str
    embedding_model: str
    llm_model: str
    # Thinking mode of local models: "false" switches it off (much faster), "true" on, "" leaves the model default.
    ollama_think: str
    # Context window for local models in tokens. Ollama's default of 4096 is too small for reranking.
    ollama_num_ctx: int

    # Which service answers questions: ollama (local), openai or anthropic (hosted).
    llm_provider: str
    openai_model: str
    openai_api_key: str
    openai_base_url: str
    anthropic_model: str
    anthropic_api_key: str
    anthropic_base_url: str

    books_dir: Path
    logs_dir: Path
    prompts_dir: Path

    chunk_size: int
    chunk_overlap: int
    embedding_dim: int

    search_mode: str
    # Weight of keyword search in the hybrid merge. Vector search always counts 1.0.
    keyword_weight: float
    rerank_mode: str
    retrieval_top_k: int
    rerank_candidates: int
    rerank_top_k: int
    prompt_template: str
    citation_instruction: str

    max_upload_mb: int

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_database} "
            f"user={self.pg_user} password={self.pg_password}"
        )

    @property
    def active_llm_model(self) -> str:
        """Model name used by the selected provider."""
        return {
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
        }.get(self.llm_provider, self.llm_model)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def _get_env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = os.getenv(name)
    value = (default if raw is None else raw).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)} (got '{value}')")
    return value


def _resolve_dir(root: Path, env_name: str, default: str) -> Path:
    path = Path(os.getenv(env_name, default))
    return path if path.is_absolute() else root / path


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]

    return Settings(
        pg_host=os.getenv("PGHOST", "localhost"),
        pg_port=_get_env_int("PGPORT", 5432),
        pg_user=os.getenv("PGUSER", "postgres"),
        pg_password=os.getenv("PGPASSWORD", "postgres"),
        pg_database=os.getenv("PGDATABASE", "vector_db"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
        llm_model=os.getenv("LLM_MODEL", "qwen3:8b"),
        ollama_think=_get_env_choice("OLLAMA_THINK", "false", ("true", "false", "")),
        ollama_num_ctx=_get_env_int("OLLAMA_NUM_CTX", 6144),
        llm_provider=_get_env_choice("LLM_PROVIDER", "ollama", LLM_PROVIDERS),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
        books_dir=_resolve_dir(root, "BOOKS_DIR", "data/books"),
        logs_dir=_resolve_dir(root, "LOGS_DIR", "logs"),
        prompts_dir=_resolve_dir(root, "PROMPTS_DIR", "config/prompts"),
        chunk_size=_get_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_env_int("CHUNK_OVERLAP", 200),
        embedding_dim=_get_env_int("EMBEDDING_DIM", 1024),
        search_mode=_get_env_choice("SEARCH_MODE", "hybrid", SEARCH_MODES),
        keyword_weight=_get_env_float("KEYWORD_WEIGHT", 0.3),
        rerank_mode=_get_env_choice("RERANK_MODE", "llm", RERANK_MODES),
        retrieval_top_k=_get_env_int("RETRIEVAL_TOP_K", 50),
        rerank_candidates=_get_env_int("RERANK_CANDIDATES", 20),
        rerank_top_k=_get_env_int("RERANK_TOP_K", 5),
        prompt_template=os.getenv("PROMPT_TEMPLATE", "default"),
        citation_instruction=os.getenv(
            "CITATION_INSTRUCTION",
            "Cite the sources you use with their numbers in square brackets, for example [1] or [2][3].",
        ),
        max_upload_mb=_get_env_int("MAX_UPLOAD_MB", 50),
    )
