from __future__ import annotations

import argparse
import time
from pathlib import Path

from .chunking import chunk_book
from .config import get_settings
from .db_pgvector import (
    DbStats,
    db_connection,
    delete_by_book_id,
    ensure_schema,
    insert_embeddings,
    reset_schema,
)
from .embeddings_ollama import embed_chunks
from .logging_utils import LogContext, setup_run_loggers, write_jsonl_event
from .pdf_loader import LoadedBook, iter_pdf_files, load_book
from .text_cleaning import clean_book_pages


def process_book(
    loaded: LoadedBook,
    log_ctx: LogContext,
    jsonl_path,
    settings,
    logger,
) -> DbStats:
    start = time.time()
    pages = loaded.pages
    book_id = loaded.meta.id

    logger.info("Processing book %s with %d pages", book_id, len(pages))
    write_jsonl_event(
        jsonl_path,
        {
            "event_type": "book_start",
            "book_id": book_id,
            "file_path": str(loaded.meta.file_path),
            "num_pages": len(pages),
        },
        log_ctx,
    )

    clean_pages = clean_book_pages(pages)
    chunks = chunk_book(clean_pages, loaded.meta, settings)
    logger.info("Book %s produced %d chunks", book_id, len(chunks))

    if not chunks:
        write_jsonl_event(
            jsonl_path,
            {
                "event_type": "book_empty",
                "book_id": book_id,
            },
            log_ctx,
        )
        return DbStats(inserted_rows=0)

    embedded = embed_chunks(chunks, settings=settings)
    with db_connection(settings) as conn:
        ensure_schema(conn, settings)
        # Ingesting the same file again replaces its chunks instead of duplicating them.
        deleted = delete_by_book_id(conn, book_id)
        if deleted:
            logger.info("Removed %d existing chunks for book %s (dedup)", deleted, book_id)
        db_stats = insert_embeddings(conn, embedded)

    elapsed = time.time() - start
    logger.info(
        "Finished book %s: %d chunks, %d rows inserted in %.2fs",
        book_id,
        len(chunks),
        db_stats.inserted_rows,
        elapsed,
    )
    write_jsonl_event(
        jsonl_path,
        {
            "event_type": "book_end",
            "book_id": book_id,
            "num_chunks": len(chunks),
            "rows_inserted": db_stats.inserted_rows,
            "elapsed_seconds": elapsed,
        },
        log_ctx,
    )
    return db_stats


def ingest_single_file(file_path: Path) -> dict:
    """
    Ingest a single PDF file. Returns {"ok": bool, "book_id": str, "chunks": int, "error": str|None}.
    """
    settings = get_settings()
    logger, _text_log_path, jsonl_path, log_ctx = setup_run_loggers(settings)
    try:
        loaded = load_book(file_path)
        stats = process_book(
            loaded=loaded,
            log_ctx=log_ctx,
            jsonl_path=jsonl_path,
            settings=settings,
            logger=logger,
        )
        if stats.inserted_rows == 0:
            error = "No text could be extracted. Scanned PDFs need OCR first."
            return {"ok": False, "book_id": loaded.meta.id, "chunks": 0, "error": error}
        return {"ok": True, "book_id": loaded.meta.id, "chunks": stats.inserted_rows, "error": None}
    except Exception as e:  # noqa: BLE001 - the caller shows the error to the user
        logger.exception("Ingest failed for %s: %s", file_path, e)
        write_jsonl_event(
            jsonl_path,
            {"event_type": "book_error", "book_id": file_path.stem, "file_path": str(file_path), "error": str(e)},
            log_ctx,
        )
        return {"ok": False, "book_id": file_path.stem, "chunks": 0, "error": str(e)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF books into pgvector via Ollama embeddings.")
    parser.add_argument(
        "--books-dir",
        type=str,
        default=None,
        help="Override books directory (defaults to BOOKS_DIR env var).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the documents table first. Needed after changing the embedding model or EMBEDDING_DIM.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.books_dir:
        settings.books_dir = Path(args.books_dir).resolve()

    logger, _text_log_path, jsonl_path, log_ctx = setup_run_loggers(settings)

    if args.reset:
        with db_connection(settings) as conn:
            reset_schema(conn)
        logger.info("Dropped the documents table (--reset)")

    logger.info("Starting ingestion run. Books directory: %s", settings.books_dir)

    total_books = 0
    total_rows = 0

    for path in iter_pdf_files(settings):
        total_books += 1
        try:
            loaded = load_book(path)
            stats = process_book(
                loaded=loaded,
                log_ctx=log_ctx,
                jsonl_path=jsonl_path,
                settings=settings,
                logger=logger,
            )
            total_rows += stats.inserted_rows
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing book %s: %s", path.name, exc)
            write_jsonl_event(
                jsonl_path,
                {
                    "event_type": "book_error",
                    "book_id": path.stem,
                    "file_path": str(path),
                    "error": str(exc),
                },
                log_ctx,
            )

    logger.info(
        "Ingestion complete. Books processed: %d, total rows inserted: %d",
        total_books,
        total_rows,
    )
    write_jsonl_event(
        jsonl_path,
        {
            "event_type": "run_summary",
            "books_processed": total_books,
            "rows_inserted": total_rows,
        },
        log_ctx,
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

