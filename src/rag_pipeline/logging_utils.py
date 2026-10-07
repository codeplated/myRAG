from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import Settings


@dataclass
class LogContext:
    run_id: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_run_loggers(settings: Settings) -> tuple[logging.Logger, Path, Path, LogContext]:
    """Create text and JSONL loggers for a single run."""
    _ensure_dir(settings.logs_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = timestamp

    text_log_path = settings.logs_dir / f"ingest_{timestamp}.log"
    jsonl_log_path = settings.logs_dir / f"ingest_{timestamp}.jsonl"

    logger = logging.getLogger(f"rag_pipeline_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        text_handler = logging.FileHandler(text_log_path, encoding="utf-8")
        text_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(text_handler)
        logger.addHandler(stream_handler)

    context = LogContext(run_id=run_id)

    # JSONL logger writes manually; return path for convenience.
    return logger, text_log_path, jsonl_log_path, context


def write_jsonl_event(path: Path, event: Dict[str, Any], context: LogContext) -> None:
    enriched = {"run_id": context.run_id, **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

