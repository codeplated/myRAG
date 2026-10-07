from __future__ import annotations

import re
from typing import List


def clean_page_text(text: str) -> str:
    """Turn raw PDF page text into clean paragraphs.

    PDF extraction breaks lines at the page width, so a line break usually sits
    in the middle of a sentence. Blank lines are kept as paragraph breaks and
    all other line breaks become spaces.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    # Re-join words that were hyphenated at the end of a line: "avia-\ntion" -> "aviation".
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


_DOT_LEADER = re.compile(r"(?:\.\s?){8,}")


def is_table_of_contents(text: str) -> bool:
    """True for pages that are mostly "Section title ........ 12" lines.

    Such pages contain the words of every section title, so they match almost
    any question while answering none.
    """
    return len(_DOT_LEADER.findall(text)) >= 5


def clean_book_pages(pages: List[str]) -> List[str]:
    """Clean every page. Table-of-contents pages become empty but keep their place,
    so page numbers of the following pages stay correct."""
    return ["" if is_table_of_contents(p) else clean_page_text(p) for p in pages]
