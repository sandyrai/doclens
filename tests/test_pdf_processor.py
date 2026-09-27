"""Tests for the PDF processor module.

What these tests cover:
  1. Text chunking — splitting text into overlapping chunks.
  2. Chunk metadata — each chunk carries source info.
  3. Edge cases — empty text, tiny text, exact boundaries.

Why NOT test extraction and embeddings here?
  - PDF extraction needs a real PDF file → tested manually.
  - Embeddings need Ollama running → too slow for unit tests.
  - ChromaDB storage needs a running instance → integration test.

  We focus on the pure functions (chunking) that we can
  test instantly without external dependencies.

How to run:
  cd E:\\ai-document-agent
  uv run pytest tests/ -v
"""

import pytest

from ai_document_agent.pdf_processor import (
    chunk_pages,
    chunk_text,
)


# ---------------------------------------------------------
# chunk_text() tests
# ---------------------------------------------------------

class TestChunkText:
    """Test the text chunking function."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size → one chunk."""

        result = chunk_text("Hello world", chunk_size=500)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_exact_chunk_size_no_overlap(self):
        """Text exactly equal to chunk_size with no
        overlap → one chunk."""

        text = "a" * 500
        result = chunk_text(
            text,
            chunk_size=500,
            chunk_overlap=0,
        )
        assert len(result) == 1

    def test_splits_long_text(self):
        """Text longer than chunk_size → multiple chunks."""

        text = "a" * 1000
        result = chunk_text(
            text,
            chunk_size=500,
            chunk_overlap=100,
        )
        assert len(result) > 1

    def test_overlap_creates_shared_content(self):
        """Consecutive chunks should share overlapping
        characters."""

        # Create text where we can verify overlap
        text = "ABCDEFGHIJ" * 20  # 200 chars

        result = chunk_text(
            text,
            chunk_size=100,
            chunk_overlap=30,
        )

        # The end of chunk 0 should appear at the
        # start of chunk 1
        assert len(result) >= 2

        # Last 30 chars of chunk 0 should equal
        # first 30 chars of chunk 1
        overlap_from_first = result[0][-30:]
        overlap_from_second = result[1][:30]
        assert overlap_from_first == overlap_from_second

    def test_empty_text_returns_empty(self):
        """Empty string → empty list."""

        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_none_text_returns_empty(self):
        """None → empty list."""

        assert chunk_text(None) == []

    def test_chunks_are_stripped(self):
        """Chunks should have leading/trailing whitespace
        removed."""

        result = chunk_text(
            "  hello world  ",
            chunk_size=500,
        )

        assert result[0] == "hello world"

    def test_custom_chunk_size(self):
        """Different chunk sizes produce different
        numbers of chunks."""

        text = "word " * 200  # 1000 chars

        small_chunks = chunk_text(
            text,
            chunk_size=100,
            chunk_overlap=0,
        )

        large_chunks = chunk_text(
            text,
            chunk_size=500,
            chunk_overlap=0,
        )

        assert len(small_chunks) > len(large_chunks)

    def test_zero_overlap(self):
        """With zero overlap, chunks should not share
        content (except at boundaries)."""

        text = "0123456789" * 10  # 100 chars

        result = chunk_text(
            text,
            chunk_size=50,
            chunk_overlap=0,
        )

        assert len(result) == 2
        assert result[0] == "0123456789" * 5
        assert result[1] == "0123456789" * 5


# ---------------------------------------------------------
# chunk_pages() tests
# ---------------------------------------------------------

class TestChunkPages:
    """Test the page chunking function."""

    def test_basic_chunking(self):
        """Should create chunks with metadata."""

        pages = [
            {"page": 1, "text": "Hello " * 200},
        ]

        result = chunk_pages(
            pages,
            filename="test.pdf",
        )

        assert len(result) > 0
        assert result[0]["text"]
        assert result[0]["metadata"]["source"] == "test.pdf"
        assert result[0]["metadata"]["page"] == 1

    def test_multiple_pages(self):
        """Chunks from different pages should have
        correct page numbers."""

        pages = [
            {"page": 1, "text": "Page one " * 200},
            {"page": 2, "text": "Page two " * 200},
        ]

        result = chunk_pages(
            pages,
            filename="multi.pdf",
        )

        # Find chunks from each page
        page_1_chunks = [
            c for c in result
            if c["metadata"]["page"] == 1
        ]

        page_2_chunks = [
            c for c in result
            if c["metadata"]["page"] == 2
        ]

        assert len(page_1_chunks) > 0
        assert len(page_2_chunks) > 0

    def test_chunk_index_is_sequential(self):
        """chunk_index should count up from 0 across
        all pages."""

        pages = [
            {"page": 1, "text": "Short text."},
            {"page": 2, "text": "Another short text."},
        ]

        result = chunk_pages(
            pages,
            filename="test.pdf",
        )

        indices = [
            c["metadata"]["chunk_index"]
            for c in result
        ]

        assert indices == list(range(len(result)))

    def test_empty_pages_returns_empty(self):
        """No pages → no chunks."""

        result = chunk_pages([], filename="empty.pdf")
        assert result == []

    def test_metadata_includes_source(self):
        """Every chunk should have the source filename."""

        pages = [
            {"page": 1, "text": "Some content here."},
        ]

        result = chunk_pages(
            pages,
            filename="report.pdf",
        )

        for chunk in result:
            assert chunk["metadata"]["source"] == "report.pdf"


# ---------------------------------------------------------
# extract_tables_to_csv(): choose the best table
# ---------------------------------------------------------
#
# Regression test: a page of prose plus a small grid table.
# A loose strategy used to win on raw row count (every text
# line counted as a row) and the real table was lost.

import csv as _csv  # noqa: E402

import fitz  # noqa: E402

from ai_document_agent.pdf_processor import extract_tables_to_csv  # noqa: E402


def _make_pdf_with_table(path):
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for i in range(20):
        page.insert_text((50, y), f"Paragraph line {i} describing the company results in words.")
        y += 14
    rows = [["Region", "Revenue", "Projects"], ["North", "412", "1120"],
            ["West", "356", "980"], ["South", "298", "870"], ["East", "126", "410"]]
    x0, top, cw, rh = 50, y + 20, 110, 20
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, top + r * rh, x0 + (c + 1) * cw, top + (r + 1) * rh)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), cell)
    doc.save(str(path))
    doc.close()


def test_extract_tables_prefers_real_table_over_prose(tmp_path):
    pdf = tmp_path / "report.pdf"
    _make_pdf_with_table(pdf)
    out = extract_tables_to_csv(str(pdf), "doc1", tmp_path)
    assert out is not None
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(_csv.reader(f))
    assert rows[0] == ["Region", "Revenue", "Projects"]
    assert [r[0] for r in rows[1:]] == ["North", "West", "South", "East"]
