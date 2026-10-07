from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .config import Settings
from .pdf_loader import BookMeta

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    book_id: str
    chunk_id: int
    text: str
    page_start: int
    page_end: int
    position: int


def document_title(book_id: str) -> str:
    """Readable title from the file name: "ntsb-aar-10-03-us-airways-1549" -> "ntsb aar 10 03 us airways 1549"."""
    return re.sub(r"[-_]+", " ", book_id).strip()


def text_for_search(chunk: Chunk) -> str:
    """Chunk text with the document title in front, used for the embedding and the keyword index.

    A chunk from the middle of a report rarely repeats what the report is about.
    With the title attached, a question such as "what caused the flight 1549
    accident" can still find it. The stored chunk text itself stays unchanged.
    """
    return f"{document_title(chunk.book_id)}\n\n{chunk.text}"


@dataclass
class _Unit:
    """Smallest piece of text that is never split: a paragraph or a sentence."""

    text: str
    page_idx: int
    starts_paragraph: bool


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """Split text that is longer than max_chars at word boundaries."""
    pieces: List[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > max_chars:  # a single "word" longer than a chunk, e.g. a URL
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def _split_into_units(pages: List[str], max_chars: int) -> List[_Unit]:
    """Split pages into paragraphs, and paragraphs that are too long into sentences.

    PDF text often has no blank lines, so a whole page arrives as one paragraph.
    Without the sentence fallback such a page would become one oversized chunk.
    """
    units: List[_Unit] = []
    for page_idx, page in enumerate(pages):
        for paragraph in page.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= max_chars:
                units.append(_Unit(paragraph, page_idx, starts_paragraph=True))
                continue
            first = True
            for sentence in _SENTENCE_END.split(paragraph):
                for piece in _split_long_text(sentence.strip(), max_chars):
                    if piece:
                        units.append(_Unit(piece, page_idx, starts_paragraph=first))
                        first = False
    return units


def _join(units: List[_Unit]) -> str:
    text = ""
    for i, unit in enumerate(units):
        if i:
            text += "\n\n" if unit.starts_paragraph else " "
        text += unit.text
    return text


def _length(units: List[_Unit]) -> int:
    return len(_join(units))


def _overlap_tail(units: List[_Unit], overlap: int) -> List[_Unit]:
    """Trailing units of a chunk, up to `overlap` characters, to repeat in the next chunk."""
    tail: List[_Unit] = []
    for unit in reversed(units):
        if _length([unit] + tail) > overlap:
            break
        tail.insert(0, unit)
    return tail


def chunk_book(clean_pages: List[str], meta: BookMeta, settings: Settings) -> List[Chunk]:
    """Pack paragraphs and sentences into chunks of at most CHUNK_SIZE characters.

    Consecutive chunks share up to CHUNK_OVERLAP characters so that an answer
    sitting on a chunk border is still found. Each chunk records the pages it
    covers, which is what makes page citations possible later.
    """
    max_chars = max(settings.chunk_size, 1)
    overlap = max(min(settings.chunk_overlap, max_chars // 2), 0)

    chunks: List[Chunk] = []
    current: List[_Unit] = []

    def flush() -> None:
        text = _join(current)
        chunks.append(
            Chunk(
                book_id=meta.id,
                chunk_id=len(chunks) + 1,
                text=text,
                page_start=min(u.page_idx for u in current) + 1,
                page_end=max(u.page_idx for u in current) + 1,
                position=len(chunks),
            )
        )

    for unit in _split_into_units(clean_pages, max_chars):
        if current and _length(current + [unit]) > max_chars:
            flush()
            current = _overlap_tail(current, overlap)
            if current and _length(current + [unit]) > max_chars:
                current = []  # the overlap does not fit next to this unit
        current.append(unit)

    if current:
        flush()
    return chunks
