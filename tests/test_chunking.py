from pathlib import Path

from rag_pipeline.chunking import Chunk, chunk_book, document_title, text_for_search
from rag_pipeline.pdf_loader import BookMeta
from rag_pipeline.text_cleaning import clean_book_pages, clean_page_text, is_table_of_contents

from .conftest import make_settings

META = BookMeta(id="manual", file_path=Path("manual.pdf"))


def sentences(count: int, start: int = 0) -> str:
    return " ".join(f"Sentence number {i} talks about flight rules." for i in range(start, start + count))


def test_page_without_blank_lines_is_split_into_small_chunks(tmp_path):
    # PDF pages usually arrive as one long paragraph. This used to become one oversized chunk.
    settings = make_settings(tmp_path, chunk_size=200, chunk_overlap=40)
    chunks = chunk_book([sentences(30)], META, settings)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_no_text_is_lost(tmp_path):
    settings = make_settings(tmp_path, chunk_size=200, chunk_overlap=0)
    page = sentences(30)
    chunks = chunk_book([page], META, settings)

    assert " ".join(c.text for c in chunks) == page


def test_consecutive_chunks_overlap(tmp_path):
    settings = make_settings(tmp_path, chunk_size=200, chunk_overlap=60)
    chunks = chunk_book([sentences(30)], META, settings)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = current.text.split(". ")[0]
        assert first_sentence in previous.text


def test_chunks_record_the_pages_they_cover(tmp_path):
    settings = make_settings(tmp_path, chunk_size=400, chunk_overlap=0)
    # Each page is about 170 characters, so the first two fit into one chunk together.
    pages = [sentences(4, start=0), sentences(4, start=10), sentences(4, start=20)]
    chunks = chunk_book(pages, META, settings)

    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    assert any(c.page_start != c.page_end for c in chunks), "a chunk should be able to span two pages"
    assert all(c.page_start <= c.page_end for c in chunks)


def test_chunk_ids_and_positions_are_sequential(tmp_path):
    settings = make_settings(tmp_path)
    chunks = chunk_book([sentences(30)], META, settings)

    assert [c.chunk_id for c in chunks] == list(range(1, len(chunks) + 1))
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.book_id == "manual" for c in chunks)


def test_text_without_any_sentence_end_is_still_split(tmp_path):
    settings = make_settings(tmp_path, chunk_size=100, chunk_overlap=0)
    chunks = chunk_book(["word " * 200], META, settings)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_single_word_longer_than_a_chunk_is_cut(tmp_path):
    settings = make_settings(tmp_path, chunk_size=50, chunk_overlap=0)
    chunks = chunk_book(["x" * 180], META, settings)

    assert [len(c.text) for c in chunks] == [50, 50, 50, 30]


def test_short_paragraphs_stay_whole(tmp_path):
    settings = make_settings(tmp_path, chunk_size=200, chunk_overlap=0)
    chunks = chunk_book(["First paragraph.\n\nSecond paragraph."], META, settings)

    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_empty_pages_produce_no_chunks(tmp_path):
    assert chunk_book(["", "   "], META, make_settings(tmp_path)) == []


def test_cleaning_joins_broken_lines_and_keeps_paragraphs():
    raw = "The aircraft was cleared\r\nfor take-off at 0932.\n\n\n\nThe crew con-\nfirmed the clearance."
    assert clean_page_text(raw) == (
        "The aircraft was cleared for take-off at 0932.\n\nThe crew confirmed the clearance."
    )


def test_cleaning_collapses_spaces():
    assert clean_page_text("  too    many\t\tspaces  ") == "too many spaces"


def test_search_text_puts_the_document_title_in_front_of_the_chunk():
    chunk = Chunk("ntsb-aar-10-03_us-airways-1549", 1, "The captain started the APU.", 19, 19, 0)

    assert document_title(chunk.book_id) == "ntsb aar 10 03 us airways 1549"
    assert text_for_search(chunk) == "ntsb aar 10 03 us airways 1549\n\nThe captain started the APU."
    assert chunk.text == "The captain started the APU.", "the stored text is not changed"


TOC_PAGE = "\n".join(f"1.{i} Section title number {i} {'.' * 40} {i + 10}" for i in range(12))


def test_table_of_contents_pages_are_recognised():
    assert is_table_of_contents(TOC_PAGE)
    assert is_table_of_contents(TOC_PAGE.replace("....", ". . . . "))
    assert not is_table_of_contents("A normal sentence. Another one... and a third.")
    assert not is_table_of_contents("One reference only ................................ 5")


def test_table_of_contents_pages_are_dropped_but_page_numbers_stay(tmp_path):
    settings = make_settings(tmp_path)
    pages = clean_book_pages(["First page text.", TOC_PAGE, "Third page text."])
    chunks = chunk_book(pages, META, settings)

    assert pages[1] == ""
    assert "Section title" not in " ".join(c.text for c in chunks)
    assert chunks[-1].page_end == 3, "the page after the table of contents is still page 3"
