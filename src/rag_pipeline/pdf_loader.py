from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

from .config import Settings


@dataclass
class BookMeta:
    id: str
    file_path: Path


@dataclass
class LoadedBook:
    meta: BookMeta
    pages: List[str]


def iter_pdf_files(settings: Settings) -> Iterable[Path]:
    books_dir = settings.books_dir
    for path in sorted(books_dir.glob("*.pdf")):
        if path.is_file():
            yield path


class PdfError(ValueError):
    """The PDF cannot be read. The message is safe to show to the user."""


def load_book(path: Path) -> LoadedBook:
    meta = BookMeta(id=path.stem, file_path=path)
    reader = PdfReader(str(path))
    # Many official PDFs are encrypted only to restrict editing and open with an empty password.
    if reader.is_encrypted and not reader.decrypt(""):
        raise PdfError("The PDF is password protected and cannot be read.")
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return LoadedBook(meta=meta, pages=pages)

