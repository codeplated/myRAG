"""Download the sample documents used for the demo and the evaluation.

All of them are in the public domain:
  - three early aviation books from Project Gutenberg (plain text, turned into PDFs here)
  - two accident reports from the US National Transportation Safety Board (US government works)

The files go to BOOKS_DIR (data/books by default) and are not committed to the
repository. Run it once, then ingest:

    python scripts/fetch_sample_data.py
    python -m rag_pipeline.ingest_books
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import requests
from fpdf import FPDF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_pipeline.config import get_settings  # noqa: E402

USER_AGENT = "myRAG sample data fetcher (https://github.com/codeplated/myRAG)"

# file name (without .pdf) -> Project Gutenberg book id
GUTENBERG_BOOKS = {
    "wright-early-history-of-the-airplane": 25420,
    "jackman-flying-machines-construction-and-operation": 907,
    "vivian-a-history-of-aeronautics": 874,
}

# file name (without .pdf) -> report number
NTSB_REPORTS = {
    "ntsb-aar-10-03-us-airways-1549-hudson-river": "AAR1003",
    "ntsb-aar-14-01-asiana-214-san-francisco": "AAR1401",
}


def download(url: str) -> bytes:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    return response.content


def gutenberg_body(raw: str) -> tuple[str, str]:
    """Return (title, text) without the Project Gutenberg header and licence footer."""
    title_match = re.search(r"^Title:\s*(.+)$", raw, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF.*?\*\*\*", raw)
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    return title, body.strip()


def to_latin1(text: str) -> str:
    """The built-in PDF fonts only cover Latin-1, so map typographic characters to plain ones."""
    replacements = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKC", text)
    return text.encode("latin-1", "replace").decode("latin-1")


def text_to_pdf(title: str, body: str, dest: Path) -> int:
    """Write the text as a simple paginated PDF. Returns the number of pages."""
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(22, 20, 22)
    pdf.add_page()
    pdf.set_font("Times", "B", 18)
    pdf.multi_cell(0, 9, to_latin1(title))
    pdf.ln(4)
    pdf.set_font("Times", "I", 10)
    pdf.multi_cell(0, 5, "Public domain text from Project Gutenberg (gutenberg.org).")
    pdf.ln(6)
    pdf.set_font("Times", "", 11)
    # Gutenberg text is hard-wrapped; blank lines separate paragraphs.
    for paragraph in re.split(r"\n\s*\n", body.replace("\r\n", "\n")):
        paragraph = " ".join(line.strip() for line in paragraph.splitlines()).strip()
        if paragraph:
            pdf.multi_cell(0, 5.5, to_latin1(paragraph))
            pdf.ln(2.5)
    pdf.output(str(dest))
    return pdf.page_no()


def main() -> int:
    books_dir = get_settings().books_dir
    books_dir.mkdir(parents=True, exist_ok=True)

    for name, book_id in GUTENBERG_BOOKS.items():
        dest = books_dir / f"{name}.pdf"
        if dest.exists():
            print(f"exists   {dest.name}")
            continue
        raw = download(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt").decode("utf-8-sig")
        title, body = gutenberg_body(raw)
        pages = text_to_pdf(title, body, dest)
        print(f"created  {dest.name} ({pages} pages)")

    for name, report in NTSB_REPORTS.items():
        dest = books_dir / f"{name}.pdf"
        if dest.exists():
            print(f"exists   {dest.name}")
            continue
        dest.write_bytes(download(f"https://www.ntsb.gov/investigations/AccidentReports/Reports/{report}.pdf"))
        print(f"fetched  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

    print(f"\nDocuments are in {books_dir}. Next: python -m rag_pipeline.ingest_books")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
