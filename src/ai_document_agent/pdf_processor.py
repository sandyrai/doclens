"""PDF processing — extraction, chunking, and embeddings.

What this module does:
  1. EXTRACT — Pull raw text from a PDF, page by page.
  2. CHUNK — Split that text into small overlapping pieces.
  3. EMBED — Convert each chunk into a vector (list of
     numbers) using Ollama's embedding model.
  4. STORE — Save chunks + vectors in ChromaDB so we can
     search them later by meaning (semantic search).

Why each step matters:
  - Extraction: LLMs can't read binary PDF files directly.
    We need the raw text first.
  - Chunking: LLMs have context limits. A 50-page PDF is
    too big to send all at once. We split it into small
    pieces so we can send only the relevant ones.
  - Overlap: If we split at exactly 500 characters, a
    sentence might get cut in half. Overlap (e.g., 100
    chars) means the end of chunk N overlaps with the
    start of chunk N+1, so no sentence is lost.
  - Embeddings: Searching by keywords ("find 'revenue'")
    misses synonyms and context. Embeddings capture
    *meaning* — "revenue" and "income" are close in
    vector space even though they share no letters.
  - Vector DB: ChromaDB stores the vectors so we can
    ask "find the 5 chunks most similar to this question"
    in milliseconds.

How to run:
  cd E:\\ai-document-agent
  uv run uvicorn ai_document_agent.main:app --reload
"""

import csv
import hashlib
import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF — the import name is "fitz"
from ollama import embed
from rank_bm25 import BM25Okapi

import chromadb


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# OCR support (Phase 2) — Tesseract + Pillow
# ---------------------------------------------------------
#
# WHY OCR?
#
#   Many PDFs are "scanned" — they contain page-sized images
#   of paper documents, not real text. When you open a scanned
#   PDF and try to select text, nothing highlights. PyMuPDF's
#   get_text() returns empty strings for these pages.
#
#   OCR (Optical Character Recognition) reads the image pixels
#   and converts them back into text. Tesseract is the most
#   widely used open-source OCR engine, originally developed
#   by HP and now maintained by Google.
#
# HOW IT WORKS:
#
#   1. We detect scanned pages: if get_text() returns very
#      little text BUT the page has images, it's scanned.
#   2. We render the page to a high-resolution image (300 DPI).
#   3. Tesseract reads the image and returns the text.
#   4. The OCR text replaces the empty extraction, so the
#      rest of the pipeline (chunking → embedding → search)
#      works exactly the same.
#
# GRACEFUL DEGRADATION:
#
#   If Tesseract is NOT installed on the system, OCR is
#   silently disabled. The app still works for normal PDFs —
#   only scanned PDFs will show "No text found" instead of
#   crashing.
#
# INSTALLATION:
#
#   Windows:  Download from https://github.com/UB-Mannheim/tesseract/wiki
#             Add to PATH or set TESSERACT_CMD env var.
#   Linux:    sudo apt install tesseract-ocr
#   macOS:    brew install tesseract
#
#   Python packages (already in pyproject.toml):
#     pip install pytesseract Pillow

# Try to import OCR libraries. If Tesseract is not installed,
# we set a flag and skip OCR gracefully instead of crashing.
try:
    import pytesseract
    from PIL import Image

    # Quick check: is the Tesseract binary actually available?
    # pytesseract.get_tesseract_version() throws FileNotFoundError
    # if the binary isn't found on PATH.
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
    logger.info(
        "OCR enabled (Tesseract %s)",
        pytesseract.get_tesseract_version(),
    )
except (ImportError, FileNotFoundError, Exception) as _ocr_err:
    OCR_AVAILABLE = False
    logger.warning(
        "OCR disabled: %s. Scanned PDFs and images "
        "won't be searchable. Install Tesseract to enable OCR.",
        _ocr_err,
    )


# ---------------------------------------------------------
# OCR Configuration
# ---------------------------------------------------------
#
# OCR_DPI = 300:
#   The resolution at which we render PDF pages to images
#   before running OCR. 300 DPI is the standard for document
#   OCR — it balances quality vs speed. Higher DPI (e.g. 600)
#   gives slightly better accuracy but takes 4x more memory
#   and time.
#
# OCR_MIN_TEXT_LENGTH = 50:
#   If regular text extraction finds fewer than 50 characters
#   on a page, we consider it "empty" and try OCR. This
#   threshold handles pages that have tiny amounts of real
#   text (like a page number) but are mostly scanned images.
#
# OCR_LANG = "eng":
#   The language Tesseract should expect. "eng" works for
#   English documents. For Hindi, use "hin". For both,
#   use "eng+hin". You need the corresponding Tesseract
#   language data files installed.

OCR_DPI = 300
OCR_MIN_TEXT_LENGTH = 50
OCR_LANG = "eng"

# ---------------------------------------------------------
# OCR Cache Directory
# ---------------------------------------------------------
#
# WHY CACHE OCR RESULTS?
#
#   OCR is SLOW — 2-10 seconds per page. A 20-page scanned
#   PDF takes 40-200 seconds to OCR. If the user re-uploads
#   the same file, or the server restarts and re-indexes,
#   we don't want to run Tesseract all over again.
#
#   The cache stores OCR text in plain .txt files, named by
#   a hash of the PDF content + page number. Same file =
#   same hash = instant cache hit.
#
#   Cache location: {project}/ocr_cache/
#   Files are small (~2KB each) and accumulate slowly.

# ---------------------------------------------------------
# OCR CACHE DIRECTORY (Fix 4 — per-document subdirectories)
#
#   Previously: all OCR cache files went into a flat
#   ocr_cache/ folder. With many documents, this became
#   a mess of hundreds of files with hash-based names.
#
#   Now: each document gets its own subdirectory, named
#   after the document file (sanitized for filesystem
#   safety). This makes it easy to:
#     - See which documents have been OCR'd
#     - Delete cache for a specific document
#     - Debug OCR issues per document
#
#   Structure:
#     ocr_cache/
#       my_report.pdf/
#         page_0.txt
#         page_1.txt
#       scanned_form.pdf/
#         page_0.txt
# ---------------------------------------------------------
OCR_CACHE_DIR = Path(__file__).resolve().parents[2] / "ocr_cache"
OCR_CACHE_DIR.mkdir(exist_ok=True)

# Upload directory path — needed by the migration function
# to find all uploaded documents and compute their hashes.
_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"


# ---------------------------------------------------------
# Evidence — structured search result
# ---------------------------------------------------------
#
# Why a dataclass instead of raw dicts?
#
#   With dicts, search results look like:
#     result["text"], result.get("rrf_score", 0)
#
#   Problems:
#     1. No autocomplete — your editor can't help you.
#     2. No typo protection — result["sorce"] fails at
#        runtime, not at write time.
#     3. Unclear contract — what keys does a result have?
#        You have to read the code that creates it.
#
#   With a dataclass:
#     result.text, result.score
#
#   Benefits:
#     1. Autocomplete works — your editor knows the fields.
#     2. Typos caught early — result.sorce is an error.
#     3. Self-documenting — the class definition IS the
#        contract. Any developer can read it and know
#        exactly what a search result contains.
#
# This is the "Evidence" pattern — each search result is
# a piece of evidence the LLM uses to answer the question.

@dataclass
class Evidence:
    """A piece of evidence retrieved from document search.

    Returned by search_documents(). Carries the chunk text,
    its origin (source file + page), and the relevance
    score from hybrid retrieval (RRF).
    """

    text: str       # The chunk content
    source: str     # Filename (e.g. "report.pdf")
    page: int       # Page number in the original PDF
    score: float    # RRF score (higher = more relevant)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
#
# Why these specific numbers?
#
# CHUNK_SIZE = 500 characters:
#   - Small enough to fit many chunks in an LLM context.
#   - Large enough to contain a full paragraph/idea.
#   - ~100-125 words, which is a good unit of meaning.
#
# CHUNK_OVERLAP = 100 characters:
#   - Prevents sentences from being cut in half.
#   - 20% overlap is a common starting point.
#   - Too much overlap = wasted storage and slower search.
#   - Too little = lost context at boundaries.
#
# EMBEDDING_MODEL:
#   - nomic-embed-text is small, fast, and free via Ollama.
#   - Produces 768-dimension vectors.
#   - Good quality for document search tasks.
#   - You already have Ollama installed — just pull this
#     model with: ollama pull nomic-embed-text

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# Override with EMBEDDING_MODEL in .env (must be pulled in Ollama).
# Changing it requires re-uploading documents: vectors from
# different embedding models are not comparable.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


# ---------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------
#
# What is ChromaDB?
#   A vector database — it stores text + vectors and lets
#   you search by similarity. Think of it like a database
#   where instead of "SELECT WHERE name = 'Alice'", you
#   say "find the 5 things most similar to this question."
#
# PersistentClient:
#   Saves data to disk so it survives server restarts.
#   Without this, uploaded PDFs would disappear every
#   time you restart the server.
#
# Collection:
#   Like a table in SQL. We use one collection called
#   "documents" for all uploaded PDFs. Each chunk is a
#   "document" in ChromaDB terms (confusing naming, but
#   that's what they call individual records).

CHROMA_PATH = Path(__file__).resolve().parents[2] / "chroma_db"

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine",
    },
)


# ---------------------------------------------------------
# BM25 Keyword Index
# ---------------------------------------------------------
#
# Why BM25 alongside semantic search?
#
#   Semantic search (embeddings) finds text with similar
#   MEANING — great for "what were the key findings?"
#   But it can miss exact keyword matches. If you search
#   for "Rahul Kumar" or "Invoice #4521", semantic search
#   might rank a vague paraphrase higher than the chunk
#   with the exact term.
#
#   BM25 is a classic keyword algorithm that scores chunks
#   by how often query words appear (term frequency) and
#   how rare those words are across all chunks (inverse
#   document frequency). It excels at exact matches.
#
#   By running BOTH and merging with Reciprocal Rank Fusion,
#   we get the best of both: meaning + keywords.
#
# Why in-memory?
#   For a learning project with a few PDFs, the entire
#   corpus fits easily in RAM. We rebuild from ChromaDB
#   on startup — ChromaDB is the durable store.

class BM25Index:
    """In-memory BM25 keyword index built from ChromaDB."""

    def __init__(self):
        self._corpus_texts: list[str] = []
        self._corpus_metadata: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._source_indices: dict[str, list[int]] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenizer.

        Lowercase + split on non-word characters + drop
        short tokens. No NLTK needed — this covers 95%
        of cases for document search.
        """
        return [
            t for t in re.split(r"\W+", text.lower())
            if len(t) >= 2
        ]

    def rebuild(self, coll) -> None:
        """Rebuild the full index from ChromaDB.

        Called on startup and after any add/delete.
        For a small corpus this takes milliseconds.
        """

        if coll.count() == 0:
            self._corpus_texts = []
            self._corpus_metadata = []
            self._bm25 = None
            self._source_indices = {}
            logger.info("BM25 index: empty (no documents)")
            return

        # Fetch everything from ChromaDB
        all_data = coll.get(
            include=["documents", "metadatas"],
        )

        self._corpus_texts = all_data["documents"]
        self._corpus_metadata = all_data["metadatas"]

        # Tokenize each chunk for BM25
        tokenized = [
            self._tokenize(text)
            for text in self._corpus_texts
        ]

        self._bm25 = BM25Okapi(tokenized)

        # Build source → indices lookup for filtering
        self._source_indices = {}
        for i, meta in enumerate(self._corpus_metadata):
            source = meta.get("source", "unknown")
            if source not in self._source_indices:
                self._source_indices[source] = []
            self._source_indices[source].append(i)

        logger.info(
            "BM25 index rebuilt: %d chunks, %d sources",
            len(self._corpus_texts),
            len(self._source_indices),
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Search the BM25 index.

        Returns results in the same dict format as
        semantic search for easy merging.
        """

        if self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        # If source_filter is set, build a temporary
        # BM25 index over just that source's chunks.
        # This keeps keyword results scoped correctly.
        if source_filter and source_filter in self._source_indices:
            indices = self._source_indices[source_filter]
            filtered_texts = [
                self._corpus_texts[i] for i in indices
            ]
            filtered_meta = [
                self._corpus_metadata[i] for i in indices
            ]
            filtered_tokenized = [
                self._tokenize(t) for t in filtered_texts
            ]

            if not filtered_tokenized:
                return []

            temp_bm25 = BM25Okapi(filtered_tokenized)
            scores = temp_bm25.get_scores(tokenized_query)

            # Pair scores with their data
            scored = list(zip(
                scores, filtered_texts, filtered_meta
            ))

        elif source_filter:
            # Source not found in index
            return []

        else:
            # Search full corpus
            scores = self._bm25.get_scores(tokenized_query)
            scored = list(zip(
                scores,
                self._corpus_texts,
                self._corpus_metadata,
            ))

        # Sort by score descending and take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n_results]

        results = []
        for score, text, meta in top:
            if score > 0:  # skip zero-score chunks
                results.append({
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "bm25_score": round(score, 4),
                })

        return results


# Create the global BM25 index and build it from
# whatever is already in ChromaDB (cold start).
bm25_index = BM25Index()
bm25_index.rebuild(collection)


# ---------------------------------------------------------
# Step 1: Extract text from documents
# ---------------------------------------------------------
#
# Supported formats: PDF, DOCX, TXT, CSV.
#
# For PDFs, we do three things beyond plain text extraction:
#
#   1. HEADING DETECTION — uses font size analysis to find
#      headings and mark them with [SECTION: ...] markers.
#      This helps the LLM understand document structure
#      ("this chunk is from the Income section").
#
#   2. TABLE EXTRACTION — uses PyMuPDF's find_tables() to
#      extract structured table data. Regular get_text()
#      often garbles tables (columns collapse into lines).
#      find_tables() gives us rows and columns properly.
#
#   3. Both markers ([SECTION:] and [TABLE]) become part
#      of the chunk text, so the LLM naturally sees them
#      in the context — no metadata changes needed.

# Supported file extensions for upload.
# Phase 2 adds image formats — these are processed via OCR
# (Tesseract) to extract text from photos of documents,
# screenshots, scanned pages saved as images, etc.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".csv", ".docx",  # Original formats
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",  # Phase 2: Images (OCR)
}

# Image-only extensions (used to route to OCR extraction)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def _extract_tables_from_page(page) -> str:
    """Extract tables from a PDF page as formatted text.

    Uses PyMuPDF's find_tables() which detects table
    structures by analyzing cell boundaries and grid lines.
    Much better than get_text() which collapses columns
    into a messy single-line format.

    Returns:
        Formatted table text with [TABLE] markers, or
        empty string if no tables found.
    """

    try:
        tables = page.find_tables()
    except Exception:
        return ""

    if not tables.tables:
        return ""

    table_texts = []

    for table in tables:

        rows = table.extract()

        if not rows:
            continue

        lines = []

        for row in rows:
            # Replace None cells with empty string
            cells = [
                str(c).strip() if c else ""
                for c in row
            ]
            lines.append(" | ".join(cells))

        if lines:
            table_texts.append("\n".join(lines))

    if not table_texts:
        return ""

    return (
        "[TABLE]\n"
        + "\n\n".join(table_texts)
        + "\n[/TABLE]"
    )


# ---------------------------------------------------------
# PDF table → internal CSV
# ---------------------------------------------------------
#
# When a PDF contains tabular data (like student results),
# the LLM can't accurately count or compute from chunked
# text. Phase 6 solved this for CSV files by loading the
# full data and computing in Python.
#
# This function extends that to PDFs: extract all tables,
# save as a CSV file alongside the uploaded PDF. Then the
# analysis tools (filter_rows, aggregate_data) can work
# on PDF table data too.
#
# The CSV is saved as: {document_id}_{filename}.table.csv
# in the same uploads directory.

def _parse_table_text_to_rows(
    table_text: str,
) -> tuple[list[str] | None, list[list[str]]]:
    """Parse pipe-delimited table text into headers + rows.

    The [TABLE]...[/TABLE] text from _extract_tables_from_page
    uses " | " as the column separator. This function parses
    that text into structured rows — a reliable fallback when
    find_tables().extract() misses rows.

    Returns:
        Tuple of (headers, data_rows).
        headers is None if no parseable lines found.
    """

    # Strip the [TABLE] markers
    text = table_text.replace("[TABLE]", "")
    text = text.replace("[/TABLE]", "")

    lines = [
        line.strip() for line in text.strip().split("\n")
        if line.strip()
    ]

    if len(lines) < 2:
        return None, []

    # First line = headers
    headers = [
        col.strip() for col in lines[0].split(" | ")
    ]

    # Remaining lines = data rows
    data_rows = []
    for line in lines[1:]:
        cells = [
            col.strip() for col in line.split(" | ")
        ]
        # Skip rows that look like headers (exact match)
        if cells == headers:
            continue
        data_rows.append(cells)

    return headers, data_rows


def _extract_with_find_tables(
    doc,
    strategy: str = "lines",
) -> tuple[list[str] | None, list[list[str]]]:
    """Extract table data using find_tables() with a given
    strategy.

    Args:
        doc: An open fitz.Document.
        strategy: "lines" (default) or "text".

    Returns:
        Tuple of (headers, data_rows).
    """

    all_rows: list[list[str]] = []
    headers: list[str] | None = None

    for page in doc:

        try:
            tables = page.find_tables(strategy=strategy)
        except Exception:
            continue

        if not tables.tables:
            continue

        for table in tables:
            rows = table.extract()

            if not rows or len(rows) < 2:
                continue

            if headers is None:
                headers = [
                    str(c).strip() if c else f"Col{i}"
                    for i, c in enumerate(rows[0])
                ]
                for row in rows[1:]:
                    cells = [
                        str(c).strip() if c else ""
                        for c in row
                    ]
                    all_rows.append(cells)
            else:
                first_row = [
                    str(c).strip() if c else ""
                    for c in rows[0]
                ]

                if first_row == headers:
                    start = 1
                else:
                    start = 0

                for row in rows[start:]:
                    cells = [
                        str(c).strip() if c else ""
                        for c in row
                    ]
                    all_rows.append(cells)

    return headers, all_rows


def _extract_with_text_positions(
    doc,
) -> tuple[list[str] | None, list[list[str]]]:
    """Extract table data using word positions (get_text).

    This is a robust fallback that works even when
    find_tables() fails. It uses the x/y positions of
    every word on the page to reconstruct table rows
    and columns.

    Strategy:
      1. Get all words with positions via get_text("words")
      2. Group words into rows by y-position (within 3pt)
      3. Sort each row's words by x-position
      4. Use the first multi-column row as header
      5. Map subsequent words to columns by closest
         header x-position

    Returns:
        Tuple of (headers, data_rows).
    """

    MIN_COLUMNS = 4  # a real table has at least 4 columns

    all_headers: list[str] | None = None
    all_rows: list[list[str]] = []

    for page in doc:

        # get_text("words") returns list of tuples:
        # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words = page.get_text("words")

        if not words:
            continue

        # --- Group words by y-position into rows ---
        # Words within 3pt vertical distance = same row.
        y_tolerance = 3.0

        # Sort by y, then x
        words.sort(key=lambda w: (w[1], w[0]))

        rows_by_y: list[list] = []
        current_row: list = [words[0]]
        current_y = words[0][1]

        for word in words[1:]:
            if abs(word[1] - current_y) <= y_tolerance:
                current_row.append(word)
            else:
                rows_by_y.append(current_row)
                current_row = [word]
                current_y = word[1]

        if current_row:
            rows_by_y.append(current_row)

        # --- Find rows with enough "columns" ---
        # A column = a group of words separated by a
        # significant x-gap (> 15pt).
        x_gap = 15.0

        table_rows: list[list[str]] = []

        for row_words in rows_by_y:

            # Sort by x-position
            row_words.sort(key=lambda w: w[0])

            # Group into columns by x-gap
            columns: list[str] = []
            current_col_words = [row_words[0][4]]
            current_x1 = row_words[0][2]  # right edge

            for word in row_words[1:]:
                word_x0 = word[0]  # left edge

                if word_x0 - current_x1 > x_gap:
                    # New column
                    columns.append(
                        " ".join(current_col_words)
                    )
                    current_col_words = [word[4]]
                else:
                    # Same column (words close together)
                    current_col_words.append(word[4])

                current_x1 = word[2]

            # Don't forget last column
            columns.append(" ".join(current_col_words))

            if len(columns) >= MIN_COLUMNS:
                table_rows.append(columns)

        if not table_rows:
            continue

        # First qualifying row = header
        if all_headers is None:
            all_headers = table_rows[0]
            n_cols = len(all_headers)

            for row in table_rows[1:]:
                # Pad or trim to match header count
                padded = row[:n_cols]
                while len(padded) < n_cols:
                    padded.append("")

                # Skip if it looks like a repeat header
                if padded == all_headers:
                    continue

                all_rows.append(padded)
        else:
            n_cols = len(all_headers)
            for row in table_rows:
                padded = row[:n_cols]
                while len(padded) < n_cols:
                    padded.append("")

                if padded == all_headers:
                    continue

                all_rows.append(padded)

    return all_headers, all_rows


def _clean_extracted_table(
    headers: list[str],
    rows: list[list[str]],
) -> tuple[list[str], list[list[str]]]:
    """Clean raw extracted table data into proper CSV.

    Problem this solves:
      PDF table extractors (find_tables, text-positions)
      often grab EVERYTHING on the page — the school title,
      subtitle, teacher name, summary text — as table rows.
      The first row becomes the "header" even though it's
      actually "Delhi Public School, Whitefield" split
      across cells. The real headers (Roll, Student Name,
      Maths...) are buried several rows down.

    Strategy:
      1. Merge headers + rows into one flat list of rows
         (since the current "headers" are probably wrong).
      2. Remove empty rows (all cells blank/whitespace).
      3. Find the REAL header row using heuristics:
         - ≥50% of cells are non-empty
         - Most cells are short text (not long sentences)
         - At least 3 distinct non-empty cells
         - The next non-empty row also has many cells
           (confirming this is a table header, not a title)
      4. Keep only data rows after the header.
      5. Remove empty columns (phantom columns from PDF
         extraction misalignment).

    Note on metadata (school name, teacher, grading policy):
      This function does NOT try to capture the header/footer
      text from the PDF. That info is already available
      through the TEXT view of the PDF — PyMuPDF's
      _extract_text_with_headings() produces clean, readable
      text chunks that naturally contain this context.
      The agent uses BOTH views (text chunks + table CSV)
      so the LLM sees everything it needs.

    Returns:
        Tuple of (clean_headers, clean_data_rows).
        Returns original data if no better header found.
    """

    # Step 1: flatten everything into one list
    all_rows = [headers] + rows
    n_cols = len(headers)

    # Step 2: remove completely empty rows
    non_empty_rows: list[list[str]] = []

    for row in all_rows:
        cells = [c.strip() if c else "" for c in row]
        if any(c for c in cells):
            non_empty_rows.append(cells)

    if len(non_empty_rows) < 2:
        return headers, rows

    # Step 3: find the real header row
    # A header row:
    #   - Has many non-empty cells (≥50% of columns)
    #   - Cells are short text (< 30 chars average)
    #   - Has ≥3 distinct non-empty cells
    #   - Contains words, not just numbers
    #   - Is followed by at least one data-like row

    def _is_numeric(s: str) -> bool:
        """Check if string is a plain number."""
        try:
            float(s.replace(",", ""))
            return True
        except ValueError:
            return False

    # Pattern for auto-generated column names like
    # "Col0", "Col1", "Col10" — these are placeholders
    # from the extractor, not real column headers.
    _col_pattern = re.compile(r"^Col\d+$")

    def _is_header_candidate(
        row: list[str], min_fill: float = 0.4,
    ) -> bool:
        """Check if a row looks like column headers."""
        non_empty = [c for c in row if c]
        fill_ratio = len(non_empty) / max(len(row), 1)

        if fill_ratio < min_fill:
            return False
        if len(non_empty) < 3:
            return False

        # Reject rows with auto-generated "Col0", "Col1"
        # placeholder names — these mean the extractor
        # treated a non-header row as the header.
        auto_cols = sum(
            1 for c in non_empty
            if _col_pattern.match(c)
        )
        if auto_cols >= 2:
            return False

        # Headers are typically short text
        avg_len = (
            sum(len(c) for c in non_empty)
            / len(non_empty)
        )
        if avg_len > 30:
            return False

        # Headers should have some text (not all numbers)
        text_cells = [
            c for c in non_empty if not _is_numeric(c)
        ]
        if len(text_cells) < 2:
            return False

        return True

    def _is_data_row(
        row: list[str], min_fill: float = 0.3,
    ) -> bool:
        """Check if a row looks like data (not metadata)."""
        non_empty = [c for c in row if c]
        return (
            len(non_empty) / max(len(row), 1) >= min_fill
        )

    header_idx = None
    best_fill = 0.0

    # Score ALL candidate rows and pick the one with the
    # highest fill ratio. This avoids picking a subtitle
    # row ("Annual Examination...") at 40% fill over the
    # real header ("Roll | Student Name | Maths...") at
    # 93% fill.

    for i, row in enumerate(non_empty_rows):
        if not _is_header_candidate(row):
            continue

        # Check that the next non-empty row looks like
        # data (confirms this is really a table header,
        # not a one-off title line)
        has_data_after = False
        for j in range(i + 1, len(non_empty_rows)):
            if _is_data_row(non_empty_rows[j]):
                has_data_after = True
                break

        if not has_data_after:
            continue

        # Pick the candidate with highest fill ratio
        non_empty = [c for c in row if c]
        fill = len(non_empty) / max(len(row), 1)

        if fill > best_fill:
            best_fill = fill
            header_idx = i

    if header_idx is None:
        # Couldn't find a better header — return original
        return headers, rows

    # Step 4: extract clean headers and data rows
    #
    # Use "consecutive streak" to find where data ends.
    # Data rows in a table have a consistent fill ratio
    # (e.g., 93% for all 15 student rows). When the fill
    # drops significantly (to 53% for "Class Average..."),
    # that's the summary section — stop collecting.
    #
    # We use 0.7 (70%) as the threshold: high enough to
    # reject fragmented summary text, low enough to keep
    # data rows even if a few cells are empty.

    clean_headers = non_empty_rows[header_idx]
    n_cols_clean = len(clean_headers)

    clean_rows: list[list[str]] = []
    data_streak_broken = False

    for row in non_empty_rows[header_idx + 1:]:
        # Skip rows that match the header (duplicate)
        if row == clean_headers:
            continue

        # Check if this looks like a data row (≥70% fill)
        non_empty_cells = [c for c in row if c]
        fill = (
            len(non_empty_cells) / max(len(row), 1)
        )

        if fill < 0.7:
            # Once we've seen data rows and the fill drops,
            # we've hit the summary section. Stop here.
            if clean_rows:
                data_streak_broken = True
            continue

        # If streak already broken, skip any remaining
        # high-fill rows (like "Subject Toppers..." at 80%)
        if data_streak_broken:
            continue

        # Pad or trim to match header column count
        padded = row[:n_cols_clean]
        while len(padded) < n_cols_clean:
            padded.append("")
        clean_rows.append(padded)

    if not clean_rows:
        return headers, rows

    # Step 5: remove empty columns (header is blank AND
    # all data cells in that column are blank). These are
    # phantom columns from PDF extraction misalignment.
    cols_to_keep = []

    for col_idx in range(len(clean_headers)):
        header_val = clean_headers[col_idx]
        all_empty = all(
            not row[col_idx]
            for row in clean_rows
            if col_idx < len(row)
        )
        if header_val or not all_empty:
            cols_to_keep.append(col_idx)

    if len(cols_to_keep) < len(clean_headers):
        removed = len(clean_headers) - len(cols_to_keep)
        clean_headers = [
            clean_headers[i] for i in cols_to_keep
        ]
        clean_rows = [
            [row[i] for i in cols_to_keep if i < len(row)]
            for row in clean_rows
        ]
        logger.info(
            "Removed %d empty columns", removed,
        )

    logger.info(
        "Table cleaning: found header at row %d, "
        "%d data rows, %d columns (was %d raw rows)",
        header_idx,
        len(clean_rows),
        len(clean_headers),
        len(rows),
    )

    return clean_headers, clean_rows


# A grid-line table with at least this many data rows is
# trusted over looser text-based detections.
MIN_GRID_TABLE_ROWS = 2


def extract_tables_to_csv(
    file_path: str,
    document_id: str,
    upload_dir: Path,
) -> Path | None:
    """Extract tables from a PDF and save as internal CSV.

    Three-strategy approach (uses whichever gets the most
    rows):
      1. find_tables(strategy="lines") — grid-line detection
      2. find_tables(strategy="text") — text-position tables
      3. Word-position parsing — robust fallback using
         get_text("words") to reconstruct rows/columns
         from the x/y positions of every word

    Why three strategies?
      Some PDFs use grid lines (strategy 1 wins). Some use
      only text alignment (strategy 2 wins). Some have
      formatting that confuses both (strategy 3 wins).
      We try all three and pick the best result.

    Args:
        file_path: Path to the PDF file.
        document_id: Document ID (for naming the CSV).
        upload_dir: Directory to save the CSV in.

    Returns:
        Path to the saved CSV, or None if no tables found.
    """

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        logger.warning(
            "Can't open PDF for table extraction: %s",
            exc,
        )
        return None

    # Run the strategies, CLEAN each result, then pick:
    #   - the grid-line table, if it has enough rows, else
    #   - whichever cleaned result has the most data rows.
    #
    # Why clean before comparing? A loose strategy (usually
    # "text") can treat every line of the page as a "row",
    # e.g. 47 rows of paragraph fragments, and would beat a
    # correct 5-row table on raw count, then collapse to 1
    # row after cleaning. Comparing cleaned results avoids
    # that. On a tie the earlier, stricter strategy wins
    # (lines > text > positions).
    #
    # Note: header/footer text (titles, names, policies) is
    # NOT captured here. That info comes through the TEXT
    # view of the PDF (_extract_text_with_headings), which
    # the agent always retrieves alongside the table CSV.

    candidates = [
        ("lines", *_extract_with_find_tables(doc, "lines")),
        ("text", *_extract_with_find_tables(doc, "text")),
        ("positions", *_extract_with_text_positions(doc)),
    ]
    doc.close()

    best_name: str | None = None
    best_headers: list[str] | None = None
    best_rows: list[list[str]] = []

    for name, headers, rows in candidates:
        if not headers or not rows:
            logger.info("Strategy '%s': no table", name)
            continue
        try:
            clean_headers, clean_rows = _clean_extracted_table(
                headers, rows,
            )
        except Exception as exc:
            logger.warning(
                "Strategy '%s': cleaning failed: %s", name, exc,
            )
            continue
        logger.info(
            "Strategy '%s': %d raw rows -> %d clean rows",
            name, len(rows), len(clean_rows),
        )
        if not clean_headers:
            continue
        # A table with drawn grid lines is strong evidence of
        # a real table. Loose strategies can turn prose into
        # many fragmented "rows", so they only win when the
        # grid-line strategy found little or nothing.
        if name == "lines" and len(clean_rows) >= MIN_GRID_TABLE_ROWS:
            best_name, best_headers, best_rows = (
                name, clean_headers, clean_rows,
            )
            break
        if len(clean_rows) > len(best_rows):
            best_name = name
            best_headers = clean_headers
            best_rows = clean_rows

    if not best_headers or not best_rows:
        logger.info(
            "No tables found in PDF for CSV extraction",
        )
        return None

    logger.info("Using table from strategy '%s'", best_name)

    # Save as CSV
    csv_filename = f"{document_id}_extracted_table.csv"
    csv_path = upload_dir / csv_filename

    with open(csv_path, "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(best_headers)
        writer.writerows(best_rows)

    logger.info(
        "Extracted PDF table to CSV: %s "
        "(%d rows, %d columns)",
        csv_path.name,
        len(best_rows),
        len(best_headers),
    )

    return csv_path


# ---------------------------------------------------------
# OCR functions (Phase 2)
# ---------------------------------------------------------


def _is_scanned_page(page) -> bool:
    """Detect if a PDF page is scanned (image-based, no text).

    A scanned page has these characteristics:
      1. Very little or no extractable text (< OCR_MIN_TEXT_LENGTH chars)
      2. Contains at least one image (the scanned page image)

    Why check both conditions?
      - A blank page has no text AND no images → not scanned,
        just empty. No point running OCR.
      - A page with lots of text → normal PDF, no OCR needed
        even if it has decorative images.
      - A page with little/no text BUT has images → scanned!
        The "text" is trapped inside the image pixels.

    Args:
        page: A PyMuPDF page object.

    Returns:
        True if the page appears to be scanned and needs OCR.
    """

    # Get text the normal way
    text = page.get_text().strip()

    # If there's enough real text, no OCR needed
    if len(text) >= OCR_MIN_TEXT_LENGTH:
        return False

    # Check if the page has images. PyMuPDF's get_images()
    # returns a list of image references on the page.
    # "full=True" gives extended info (not needed here,
    # but we just need the count).
    images = page.get_images(full=True)

    # Has images but no text → scanned page
    if images:
        logger.debug(
            "Page %d: scanned (text=%d chars, images=%d)",
            page.number + 1,
            len(text),
            len(images),
        )
        return True

    return False


def _preprocess_for_ocr(image: "Image.Image") -> "Image.Image":
    """Preprocess an image to improve OCR accuracy.

    WHY PREPROCESSING MATTERS:
      Raw scanned images often have problems that confuse
      Tesseract:
        - Low contrast (gray text on light gray background)
        - Color noise (colored backgrounds, watermarks)
        - Uneven lighting (dark corners, shadows)

      Preprocessing fixes these issues BEFORE Tesseract sees
      the image, dramatically improving text recognition.

    PIPELINE (each step and why):

      1. GRAYSCALE — Convert to single-channel gray.
         Why: Tesseract works best on grayscale. Color adds
         no information for text recognition but triples
         the data Tesseract has to process. Colored
         backgrounds and highlights can confuse character
         recognition.

      2. CONTRAST ENHANCEMENT — Boost the difference between
         text (dark) and background (light).
         Why: Scanned documents often have washed-out text,
         especially copies of copies. Boosting contrast by
         1.5x makes the text pixels clearly darker than the
         background pixels.

      3. BINARIZATION (Otsu thresholding) — Convert to pure
         black-and-white (no gray).
         Why: After grayscale + contrast, pixels are either
         "mostly dark" (text) or "mostly light" (background).
         Otsu's method automatically finds the optimal
         threshold to split them into pure black or white.
         This removes ALL background noise — watermarks,
         scanner artifacts, uneven lighting — leaving only
         clean text shapes for Tesseract.

    Args:
        image: A PIL Image (can be color or grayscale).

    Returns:
        A preprocessed PIL Image optimized for OCR.
    """

    from PIL import ImageEnhance

    # Step 1: Convert to grayscale
    # "L" mode = 8-bit grayscale (0=black, 255=white)
    gray = image.convert("L")

    # Step 2: Boost contrast by 1.5x
    # This makes dark pixels darker and light pixels lighter,
    # so text stands out more clearly from the background.
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.5)

    # Step 3: Otsu binarization (adaptive thresholding)
    #
    # HOW OTSU WORKS:
    #   It looks at the histogram of pixel values (how many
    #   pixels are at each brightness level 0-255) and finds
    #   the threshold that best separates the two peaks
    #   (dark=text, light=background). Everything below the
    #   threshold → black, everything above → white.
    #
    #   We use PIL's .point() to apply the threshold.
    #   First, we compute the optimal threshold by finding
    #   the value that minimizes within-class variance.
    #
    # FALLBACK:
    #   If Otsu can't find a good split (e.g., all-white
    #   page), we use 128 as a safe default.

    histogram = enhanced.histogram()

    # Otsu's threshold calculation:
    # Find the threshold that minimizes weighted variance
    # between foreground and background pixel groups.
    total_pixels = sum(histogram)

    if total_pixels == 0:
        # Empty image — return as-is
        return enhanced

    # Cumulative sums and means for Otsu
    sum_total = sum(i * h for i, h in enumerate(histogram))
    sum_bg = 0.0
    weight_bg = 0
    best_threshold = 128  # safe default
    best_variance = 0.0

    for t in range(256):
        weight_bg += histogram[t]

        if weight_bg == 0:
            continue

        weight_fg = total_pixels - weight_bg

        if weight_fg == 0:
            break

        sum_bg += t * histogram[t]

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        # Between-class variance
        variance = (
            weight_bg * weight_fg
            * (mean_bg - mean_fg) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    # Apply threshold: pixels below → black, above → white
    binary = enhanced.point(
        lambda px: 255 if px > best_threshold else 0,
        "L",
    )

    logger.debug(
        "OCR preprocessing: Otsu threshold=%d",
        best_threshold,
    )

    return binary


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename for use as a directory name.

    WHY THIS IS NEEDED:
      Document filenames can contain characters that are
      invalid in directory names on some operating systems
      (e.g. colons, slashes, question marks on Windows).

      We replace any non-alphanumeric character (except dots,
      hyphens, and underscores) with an underscore. This
      keeps names readable while being filesystem-safe.

    Examples:
      "my report.pdf"     → "my_report.pdf"
      "2024/Q1 data.pdf"  → "2024_Q1_data.pdf"
    """
    import re
    # Keep letters, digits, dots, hyphens, underscores
    # Replace everything else with underscore
    return re.sub(r'[^\w.\-]', '_', name)


def _get_ocr_cache_path(
    file_path: str,
    page_num: int,
) -> Path:
    """Get the cache file path for an OCR'd page.

    PER-DOCUMENT CACHE STRUCTURE (Fix 4):

      Previously, all OCR cache files lived in a flat
      directory with hash-based names like:
        ocr_cache/a3f8b2c1_page0.txt
        ocr_cache/a3f8b2c1_page1.txt
        ocr_cache/7e9d4f0a_page0.txt

      This made it impossible to tell which document a
      cache file belonged to.

      Now we use per-document subdirectories:
        ocr_cache/my_report.pdf/page_0.txt
        ocr_cache/my_report.pdf/page_1.txt
        ocr_cache/other_doc.pdf/page_0.txt

      The subdirectory is named after the original document
      filename (sanitized for filesystem safety). Inside
      each subdirectory, files are named page_N.txt.

      We ALSO check for old-style flat cache files and read
      them if they exist (backward compatibility). This way,
      documents already OCR'd don't need to be re-processed
      after this upgrade.

    Args:
        file_path: Path to the PDF or image file.
        page_num: Page number (0-based for PDFs, 0 for images).

    Returns:
        Path to the cache file.
    """

    # Get the document filename for the subdirectory
    doc_name = Path(file_path).name
    safe_name = _sanitize_filename(doc_name)

    # Create per-document subdirectory
    doc_cache_dir = OCR_CACHE_DIR / safe_name
    doc_cache_dir.mkdir(exist_ok=True)

    return doc_cache_dir / f"page_{page_num}.txt"


def _get_legacy_cache_path(
    file_path: str,
    page_num: int,
) -> Path:
    """Get the OLD-STYLE flat cache path (for backward compat).

    This lets us read cache files created before Fix 4
    so existing OCR results aren't lost after the upgrade.
    """
    file_hash = hashlib.sha256(
        Path(file_path).read_bytes()
    ).hexdigest()[:16]

    return OCR_CACHE_DIR / f"{file_hash}_page{page_num}.txt"


def _get_ocr_cache_key(
    file_path: str,
    page_num: int,
) -> str:
    """Generate a cache key for an OCR'd page.

    NOTE: This function now returns a PATH STRING instead
    of just a hash key. The path includes the per-document
    subdirectory. Kept for backward compatibility with
    callers that pass the key to _read/_write_ocr_cache.

    Args:
        file_path: Path to the PDF or image file.
        page_num: Page number (0-based for PDFs, 0 for images).

    Returns:
        A string representing the cache path (relative to
        OCR_CACHE_DIR).
    """
    # Store the full file_path so _read/_write can resolve it
    # We use the new path-based approach internally
    doc_name = Path(file_path).name
    safe_name = _sanitize_filename(doc_name)
    return f"{safe_name}/page_{page_num}"


def _read_ocr_cache(
    cache_key: str,
    file_path: str | None = None,
    page_num: int = 0,
) -> str | None:
    """Read cached OCR text if it exists.

    Checks BOTH the new per-document cache structure AND
    the old flat cache structure for backward compatibility.
    This means documents OCR'd before the upgrade still
    get cache hits without re-processing.

    Args:
        cache_key: The cache key from _get_ocr_cache_key().
        file_path: Original file path (for legacy fallback).
        page_num: Page number (for legacy fallback).

    Returns:
        The cached text string, or None if not cached.
    """

    # Try new per-document cache path first
    cache_file = OCR_CACHE_DIR / f"{cache_key}.txt"

    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8")
        logger.info(
            "OCR cache HIT (per-doc): %s (%d chars)",
            cache_key,
            len(text),
        )
        return text

    # Fallback: try old flat hash-based cache
    # This provides backward compatibility so existing
    # OCR results aren't lost after the upgrade.
    if file_path:
        legacy_path = _get_legacy_cache_path(
            file_path, page_num,
        )
        if legacy_path.exists():
            text = legacy_path.read_text(encoding="utf-8")
            logger.info(
                "OCR cache HIT (legacy): %s (%d chars)",
                legacy_path.name,
                len(text),
            )
            return text

    return None


def _write_ocr_cache(cache_key: str, text: str) -> None:
    """Save OCR text to per-document cache.

    Writes to the new per-document subdirectory structure.
    The subdirectory is created automatically by
    _get_ocr_cache_key() / _get_ocr_cache_path().

    Args:
        cache_key: The cache key from _get_ocr_cache_key().
                   Format: "document_name/page_N"
        text: The OCR-extracted text to cache.
    """

    cache_file = OCR_CACHE_DIR / f"{cache_key}.txt"

    # Ensure parent directory exists (per-document subdir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cache_file.write_text(text, encoding="utf-8")

    logger.debug("OCR cache WRITE: %s", cache_key)


def migrate_legacy_ocr_cache() -> dict:
    """Migrate old flat hash-named OCR cache files into
    per-document subdirectories.

    WHY THIS IS NEEDED:

      Before the per-document subdirectory fix, OCR cache
      files were saved with hash-based names directly in
      the ocr_cache/ folder:

        ocr_cache/27bd17d91a6894d6_page0.txt
        ocr_cache/a3f8b2c1d9e4f0b7_page1.txt

      These are impossible to identify — you can't tell
      which document a file belongs to. As the number of
      documents grows, this becomes an unmanageable mess.

    HOW MIGRATION WORKS:

      1. Scan the uploads/ folder to find all uploaded
         document files (PDFs, images, DOCX, etc.)
      2. For each document, compute the SHA-256 hash of
         its content (first 16 hex chars — same as the
         legacy cache key generation)
      3. Look for legacy flat files matching that hash
         pattern: {hash}_page{N}.txt
      4. Move matched files into the new per-document
         subdirectory: ocr_cache/{document_name}/page_{N}.txt
      5. Delete any remaining unmatched legacy flat files
         (they belong to documents that have been deleted)

    WHEN IT RUNS:

      Called once at module load time (server startup).
      If there are no legacy flat files, it returns
      immediately (zero overhead).

    Returns:
        Dict with migration stats:
        {
            "migrated": 3,     — files moved to subfolders
            "orphans_removed": 2,  — unmatched files deleted
            "already_clean": True/False
        }
    """

    # Step 1: Find all legacy flat files in the root of
    # ocr_cache/. These are files (not directories) whose
    # names match the pattern: {hash}_page{N}.txt
    #
    # New-style files live inside subdirectories, so they
    # won't be caught by this scan.
    import re

    legacy_pattern = re.compile(
        r'^[0-9a-f]+_page(\d+)\.txt$'
    )

    legacy_files = []

    for item in OCR_CACHE_DIR.iterdir():
        if item.is_file() and legacy_pattern.match(item.name):
            legacy_files.append(item)

    if not legacy_files:
        # No legacy files — cache is already clean
        return {
            "migrated": 0,
            "orphans_removed": 0,
            "already_clean": True,
        }

    logger.info(
        "OCR cache migration: found %d legacy flat files",
        len(legacy_files),
    )

    # Step 2: Build a hash → document_name mapping by
    # scanning all uploaded files and computing their hashes.
    #
    # The legacy cache used:
    #   hashlib.sha256(file_bytes).hexdigest()[:16]
    # as the hash prefix in filenames.
    hash_to_docname: dict[str, str] = {}

    if _UPLOAD_DIR.exists():
        for upload_file in _UPLOAD_DIR.iterdir():

            # Skip directories (like the "images" folder)
            if not upload_file.is_file():
                continue

            # Skip non-document files (like extracted CSVs)
            if upload_file.suffix.lower() not in (
                SUPPORTED_EXTENSIONS
            ):
                continue

            try:
                file_bytes = upload_file.read_bytes()
                file_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()[:16]

                # The document name is the original filename
                # (after the document_id prefix in the upload
                # filename: "abc123_report.pdf" → "report.pdf")
                # But for the OCR cache subfolder, we use
                # the full upload filename since that's what
                # the OCR functions receive as file_path.
                doc_name = upload_file.name
                hash_to_docname[file_hash] = doc_name

            except Exception as exc:
                logger.debug(
                    "Skipping %s during migration: %s",
                    upload_file.name,
                    exc,
                )

    logger.info(
        "OCR cache migration: mapped %d document hashes",
        len(hash_to_docname),
    )

    # Step 3: Match legacy files to documents and migrate
    migrated = 0
    orphans_removed = 0

    for legacy_file in legacy_files:

        # Parse the hash and page number from the filename.
        # Format: {hash}_page{N}.txt
        name = legacy_file.stem  # e.g. "27bd17d91a6894d6_page0"
        parts = name.rsplit("_page", 1)

        if len(parts) != 2:
            # Unexpected format — remove as orphan
            legacy_file.unlink()
            orphans_removed += 1
            continue

        file_hash = parts[0]
        page_num_str = parts[1]

        try:
            page_num = int(page_num_str)
        except ValueError:
            legacy_file.unlink()
            orphans_removed += 1
            continue

        # Look up the document name for this hash
        doc_name = hash_to_docname.get(file_hash)

        if doc_name is None:
            # No matching document — this is an orphan from
            # a document that was deleted. Clean it up.
            legacy_file.unlink()
            orphans_removed += 1

            logger.debug(
                "Removed orphan cache: %s (no matching doc)",
                legacy_file.name,
            )
            continue

        # Migrate: move to per-document subdirectory
        safe_name = _sanitize_filename(doc_name)
        new_dir = OCR_CACHE_DIR / safe_name
        new_dir.mkdir(exist_ok=True)

        new_path = new_dir / f"page_{page_num}.txt"

        # Don't overwrite if new-style file already exists
        # (document was re-OCR'd after the fix)
        if new_path.exists():
            legacy_file.unlink()
            orphans_removed += 1
            continue

        # Read content and write to new location, then
        # delete the old file
        content = legacy_file.read_text(encoding="utf-8")
        new_path.write_text(content, encoding="utf-8")
        legacy_file.unlink()
        migrated += 1

        logger.debug(
            "Migrated: %s → %s/page_%d.txt",
            legacy_file.name,
            safe_name,
            page_num,
        )

    logger.info(
        "OCR cache migration complete: "
        "%d migrated, %d orphans removed",
        migrated,
        orphans_removed,
    )

    return {
        "migrated": migrated,
        "orphans_removed": orphans_removed,
        "already_clean": False,
    }


# ---------------------------------------------------------
# Run migration on startup
# ---------------------------------------------------------
#
# This runs once when the module is first imported (server
# startup). If the cache is already clean (no legacy flat
# files), it returns immediately with zero overhead.
#
# After migration, the ocr_cache/ folder will only contain
# per-document subdirectories — no more mystery hash files.

migrate_legacy_ocr_cache()


def _ocr_page(
    page,
    file_path: str | None = None,
) -> str:
    """Run OCR on a single PDF page using Tesseract.

    Enhanced with:
      1. IMAGE PREPROCESSING — grayscale, contrast boost,
         and Otsu binarization to dramatically improve
         Tesseract accuracy on scanned documents.
      2. OCR CACHING — results are saved to disk so the
         same page is never OCR'd twice.
      3. TESSERACT PSM 6 — Page Segmentation Mode 6 assumes
         a uniform block of text, which works better for
         typical document pages than the default auto mode.

    How the full pipeline works:
      1. Check cache → if hit, return instantly (0ms)
      2. Render page to 300 DPI image via PyMuPDF
      3. Preprocess: grayscale → contrast → binarize
      4. Run Tesseract with --psm 6 (block of text mode)
      5. Clean up the OCR output text
      6. Save to cache for next time

    Args:
        page: A PyMuPDF page object.
        file_path: Path to the PDF file (for cache key).
                   If None, caching is skipped.

    Returns:
        The OCR-extracted text, or empty string if OCR fails
        or is not available.
    """

    if not OCR_AVAILABLE:
        return ""

    # Step 0: Check cache first (instant if cached)
    cache_key = None
    if file_path:
        cache_key = _get_ocr_cache_key(
            file_path, page.number,
        )
        # Pass file_path and page_num for legacy cache
        # fallback (backward compatibility with old flat
        # cache structure from before Fix 4)
        cached = _read_ocr_cache(
            cache_key,
            file_path=file_path,
            page_num=page.number,
        )
        if cached is not None:
            return cached

    try:
        # Step 1: Render the page to a high-res image
        #
        # get_pixmap() converts the PDF page into raw pixels.
        # 300 DPI gives print-quality resolution — enough
        # detail for Tesseract to recognize even small text.
        pixmap = page.get_pixmap(dpi=OCR_DPI)

        # Step 2: Convert pixmap → PIL Image
        #
        # pixmap.tobytes("png") encodes as lossless PNG.
        # PIL reads it from a BytesIO buffer (in-memory file).
        img_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        # Step 3: Preprocess the image for better OCR
        #
        # This is the KEY improvement over raw OCR:
        # grayscale → contrast boost → Otsu binarization
        # removes background noise and makes text crisp.
        image = _preprocess_for_ocr(image)

        # Step 4: Run Tesseract OCR with optimized settings
        #
        # --psm 6 = "Assume a single uniform block of text"
        #   This tells Tesseract the page is a document with
        #   paragraphs, not a photo with scattered text.
        #   Much better for book pages, forms, and reports.
        #
        # --oem 3 = "Default OCR Engine Mode"
        #   Uses the LSTM neural network engine (most accurate).
        #   Tesseract 4+ uses this by default, but we set it
        #   explicitly for clarity.
        ocr_text = pytesseract.image_to_string(
            image,
            lang=OCR_LANG,
            config="--psm 6 --oem 3",
        )

        # Step 5: Clean up OCR output
        #
        # Tesseract often produces:
        #   - Excessive blank lines between paragraphs
        #   - Trailing spaces on every line
        #   - Random single characters on their own lines
        #     (from page numbers, watermarks, etc.)
        #
        # We clean these up to get readable text.

        # Remove excessive blank lines (3+ → 2)
        ocr_text = re.sub(r'\n{3,}', '\n\n', ocr_text)

        # Remove lines that are just 1-2 characters
        # (usually OCR noise from dots, dashes, page numbers)
        lines = ocr_text.split('\n')
        cleaned_lines = [
            line for line in lines
            if len(line.strip()) > 2 or not line.strip()
        ]
        ocr_text = '\n'.join(cleaned_lines)

        # Final trim
        ocr_text = ocr_text.strip()

        if ocr_text:
            logger.info(
                "OCR page %d: extracted %d chars",
                page.number + 1,
                len(ocr_text),
            )

            # Step 6: Save to cache
            if cache_key:
                _write_ocr_cache(cache_key, ocr_text)

        return ocr_text

    except Exception as exc:
        logger.warning(
            "OCR failed on page %d: %s",
            page.number + 1,
            exc,
        )
        return ""


def extract_text_from_image(
    image_path: str,
) -> list[dict]:
    """Extract text from a standalone image file using OCR.

    This handles direct image uploads (.png, .jpg, .jpeg,
    .tiff, .bmp) — not PDFs. The user might upload a photo
    of a document, a screenshot of a table, or a scanned
    page saved as an image.

    Enhanced pipeline:
      1. Check OCR cache (instant if previously processed)
      2. Open the image with PIL (Pillow)
      3. Preprocess: grayscale → contrast → binarize
      4. Run Tesseract OCR with --psm 6 (block text mode)
      5. Clean up and cache the result
      6. Return in same format as extract_text_from_pdf()

    Args:
        image_path: Path to the image file on disk.

    Returns:
        A list with one dict: [{"page": 1, "text": "..."}]
        Empty list if OCR is not available or no text found.
    """

    if not OCR_AVAILABLE:
        logger.warning(
            "Cannot process image '%s': OCR is not available. "
            "Install Tesseract to enable image text extraction.",
            image_path,
        )
        return []

    logger.info(
        "Extracting text from image via OCR: %s",
        image_path,
    )

    # Check cache first
    cache_key = _get_ocr_cache_key(image_path, 0)
    # Pass file_path and page_num for legacy cache fallback
    cached = _read_ocr_cache(
        cache_key,
        file_path=image_path,
        page_num=0,
    )

    if cached is not None:
        return [{"page": 1, "text": cached}]

    try:
        # Open the image with PIL
        image = Image.open(image_path)

        # Preprocess for better OCR accuracy
        # (grayscale → contrast → Otsu binarization)
        image = _preprocess_for_ocr(image)

        # Run Tesseract with optimized settings
        ocr_text = pytesseract.image_to_string(
            image,
            lang=OCR_LANG,
            config="--psm 6 --oem 3",
        )

        # Clean up whitespace and noise
        ocr_text = re.sub(r'\n{3,}', '\n\n', ocr_text)

        # Remove short noise lines (1-2 chars)
        lines = ocr_text.split('\n')
        cleaned_lines = [
            line for line in lines
            if len(line.strip()) > 2 or not line.strip()
        ]
        ocr_text = '\n'.join(cleaned_lines).strip()

        if not ocr_text:
            logger.info(
                "No text found in image: %s",
                image_path,
            )
            return []

        logger.info(
            "OCR extracted %d chars from image: %s",
            len(ocr_text),
            image_path,
        )

        # Cache the result
        _write_ocr_cache(cache_key, ocr_text)

        # Return in same format as PDF extraction
        # (page=1 since images are single-page)
        return [{"page": 1, "text": ocr_text}]

    except Exception as exc:
        logger.error(
            "Failed to OCR image '%s': %s",
            image_path,
            exc,
        )
        return []


def _extract_text_with_headings(page) -> str:
    """Extract text from a PDF page with heading markers.

    How heading detection works:
      1. Use get_text("dict") to get every text span with
         its font size, font name, and position.
      2. Find the body font size — the most common size
         on the page (e.g., 11pt or 12pt).
      3. Any text with a significantly larger font size
         (>15% bigger) is marked as a heading.
      4. Headings get a [SECTION: ...] prefix so the LLM
         knows "this is a section title, not body text."

    Why not just use get_text()?
      get_text() returns plain text with no font info.
      A heading like "FINANCIAL SUMMARY" looks exactly
      like body text — the LLM has no way to know it's
      a section title. With markers, the LLM can say
      "this information is from the Financial Summary
      section on page 3."

    Falls back to plain get_text() if the page has no
    measurable text spans (e.g., scanned images).
    """

    data = page.get_text("dict")

    # Collect font sizes from all text spans
    sizes = []

    for block in data["blocks"]:
        if block["type"] != 0:   # 0 = text, 1 = image
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                # Only count substantial text for size stats
                if text and len(text) > 3:
                    sizes.append(round(span["size"], 1))

    if not sizes:
        # No measurable text — fall back to plain extraction
        return page.get_text().strip()

    # Body text = most common font size on the page
    body_size = max(
        set(sizes), key=sizes.count
    )

    # Anything 15%+ larger than body is a heading
    heading_threshold = body_size * 1.15

    # Rebuild text from blocks, marking headings
    parts = []

    for block in data["blocks"]:
        if block["type"] != 0:
            continue

        block_lines = []

        for line in block["lines"]:

            # Join all spans in the line
            line_text = "".join(
                span["text"] for span in line["spans"]
            ).strip()

            if not line_text:
                continue

            # Find the largest font size in this line
            max_size = max(
                (span["size"] for span in line["spans"]),
                default=0,
            )

            # Mark as heading if large font + short text
            if (
                max_size >= heading_threshold
                and len(line_text) < 200
            ):
                block_lines.append(
                    f"\n[SECTION: {line_text}]"
                )
            else:
                block_lines.append(line_text)

        if block_lines:
            parts.append("\n".join(block_lines))

    return "\n\n".join(parts)


def extract_text_from_pdf(
    pdf_path: str,
) -> list[dict]:
    """Extract text from each page of a PDF.

    Enhanced with heading detection, table extraction,
    and OCR for scanned pages (Phase 2).

    The extraction pipeline for each page:
      1. Try normal text extraction (headings + tables)
      2. If the page appears scanned (little/no text but
         has images), fall back to OCR via Tesseract
      3. OCR text gets a [OCR] marker so the LLM knows
         the text quality may be lower than native text

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A list of dicts, one per page:
        [{"page": 1, "text": "..."}, ...]
    """

    logger.info("Extracting text from: %s", pdf_path)

    doc = fitz.open(pdf_path)

    pages = []
    ocr_page_count = 0  # Track how many pages needed OCR

    for page_num in range(len(doc)):

        page = doc[page_num]

        # --------------------------------------------------
        # Step 1: Try normal text extraction (fast, accurate)
        # --------------------------------------------------
        text = _extract_text_with_headings(page)

        # Extract tables separately (get_text often
        # garbles table data — find_tables is better)
        table_text = _extract_tables_from_page(page)

        if table_text:
            text = text + "\n\n" + table_text

        # --------------------------------------------------
        # Step 2: If page looks scanned, try OCR (Phase 2)
        # --------------------------------------------------
        #
        # WHY CHECK AFTER normal extraction?
        #   Some PDFs have a mix: pages 1-3 are normal text,
        #   page 4 is a scanned form. We only run OCR on
        #   pages that actually need it — this saves time
        #   (OCR is ~10x slower than normal extraction).
        #
        # The [OCR] marker tells the LLM that this text
        # came from image recognition, so it might have
        # minor errors (e.g., "rn" misread as "m", or
        # "1" misread as "l"). The LLM can compensate.

        if (
            len(text.strip()) < OCR_MIN_TEXT_LENGTH
            and _is_scanned_page(page)
        ):
            ocr_text = _ocr_page(page, file_path=pdf_path)

            if ocr_text:
                text = f"[OCR]\n{ocr_text}\n[/OCR]"
                ocr_page_count += 1

                logger.info(
                    "Page %d: used OCR (%d chars)",
                    page_num + 1,
                    len(ocr_text),
                )

        if text.strip():
            pages.append(
                {
                    "page": page_num + 1,
                    "text": text.strip(),
                }
            )

    doc.close()

    # Log a summary of OCR usage for this document
    if ocr_page_count > 0:
        logger.info(
            "Extracted %d pages (%d via OCR) from: %s",
            len(pages),
            ocr_page_count,
            pdf_path,
        )
    else:
        logger.info(
            "Extracted %d pages with text", len(pages),
        )

    return pages


# ---------------------------------------------------------
# Multi-format extractors
# ---------------------------------------------------------
#
# Same return format as extract_text_from_pdf():
#   [{"page": N, "text": "..."}]
#
# TXT and CSV don't have pages, so we use page=1.
# DOCX has paragraph styles, so we can detect headings
# without font size analysis — the style IS the intent.

def extract_text_from_txt(file_path: str) -> list[dict]:
    """Extract text from a plain text file.

    Simple — just read the file. No structure to detect.
    """

    logger.info("Extracting text from TXT: %s", file_path)

    text = Path(file_path).read_text(
        encoding="utf-8", errors="replace"
    )

    if not text.strip():
        return []

    return [{"page": 1, "text": text.strip()}]


def extract_text_from_csv(file_path: str) -> list[dict]:
    """Extract text from a CSV file as a table.

    CSVs are inherently tabular, so the entire file is
    wrapped in [TABLE]...[/TABLE] markers. The first row
    is typically the header.
    """

    logger.info("Extracting text from CSV: %s", file_path)

    with open(
        file_path, "r",
        encoding="utf-8", errors="replace",
    ) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    lines = [" | ".join(row) for row in rows]
    text = "[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]"

    return [{"page": 1, "text": text}]


def extract_text_from_docx(file_path: str) -> list[dict]:
    """Extract text from a Word document with headings.

    DOCX files have paragraph styles (Heading 1, Heading 2,
    etc.) that tell us EXACTLY which lines are headings —
    no font size guessing needed. We mark them with
    [SECTION: ...] just like in PDFs.

    Also extracts tables from the document.
    """

    # Import here — python-docx is an optional dependency.
    # If not installed, this gives a clear error message
    # instead of crashing on module import.
    from docx import Document

    logger.info(
        "Extracting text from DOCX: %s", file_path,
    )

    doc = Document(file_path)

    parts = []

    # Extract paragraphs with heading detection
    for para in doc.paragraphs:

        text = para.text.strip()

        if not text:
            continue

        # python-docx paragraph styles tell us if it's
        # a heading — no guessing needed
        style_name = (
            para.style.name if para.style else ""
        )

        if style_name.startswith("Heading"):
            parts.append(f"\n[SECTION: {text}]")
        else:
            parts.append(text)

    # Extract tables
    for table in doc.tables:

        table_lines = []

        for row in table.rows:
            cells = [
                cell.text.strip() for cell in row.cells
            ]
            table_lines.append(" | ".join(cells))

        if table_lines:
            parts.append(
                "\n[TABLE]\n"
                + "\n".join(table_lines)
                + "\n[/TABLE]"
            )

    full_text = "\n".join(parts)

    if not full_text.strip():
        return []

    # DOCX doesn't have reliable page numbers
    return [{"page": 1, "text": full_text.strip()}]


# ---------------------------------------------------------
# Step 2: Chunk text with overlap
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks.

    Why overlap?
      Imagine this sentence sits at the boundary between
      two chunks:

        "The company's revenue was $5M, which represents
         a 20% increase over last year."

      Without overlap, "revenue was $5M" might end up in
      chunk 3 and "20% increase over last year" in chunk 4.
      Neither chunk has the full picture.

      With 100-char overlap, chunk 4 starts 100 characters
      before where chunk 3 ends, so the full sentence
      appears in at least one chunk.

    Args:
        text: The raw text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: How many characters to repeat
                       between consecutive chunks.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap).
        # This creates the overlap between chunks.
        start += chunk_size - chunk_overlap

    return chunks


def chunk_pages(
    pages: list[dict],
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk all pages and attach metadata.

    Args:
        pages: Output from extract_text_from_pdf().
        filename: Original filename for metadata.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunk dicts:
        [
            {
                "text": "chunk content...",
                "metadata": {
                    "source": "report.pdf",
                    "page": 3,
                    "chunk_index": 0,
                }
            },
            ...
        ]

    Why keep metadata?
      When the LLM finds a relevant chunk, we want to
      tell the user WHERE it came from: "Found in
      report.pdf, page 3." Without metadata, we'd just
      have floating text with no source attribution.
    """

    all_chunks = []
    chunk_index = 0

    for page_data in pages:

        page_chunks = chunk_text(
            page_data["text"],
            chunk_size,
            chunk_overlap,
        )

        for chunk in page_chunks:

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page_data["page"],
                        "chunk_index": chunk_index,
                    },
                }
            )

            chunk_index += 1

    logger.info(
        "Created %d chunks from '%s'",
        len(all_chunks),
        filename,
    )

    return all_chunks


# ---------------------------------------------------------
# Step 3: Generate embeddings
# ---------------------------------------------------------

def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Convert text chunks into vector embeddings.

    What is an embedding?
      A list of numbers (a "vector") that captures the
      *meaning* of text. Similar meanings → similar
      vectors. For example:

        "What is the company's revenue?"
        → [0.12, -0.45, 0.78, 0.33, ...]  (768 numbers)

        "How much money did the business earn?"
        → [0.11, -0.43, 0.76, 0.35, ...]  (very similar!)

        "What color is the sky?"
        → [0.89, 0.22, -0.15, 0.67, ...]  (very different)

    Why Ollama for embeddings?
      - You already have Ollama running for chat.
      - nomic-embed-text is small (~270MB) and fast.
      - Runs locally — no API keys, no internet needed.
      - Same `ollama` Python package you're already using.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each is a list of
        floats, typically 768 dimensions).
    """

    logger.info(
        "Generating embeddings for %d chunks",
        len(texts),
    )

    response = embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    logger.info(
        "Generated %d embeddings (dim=%d)",
        len(response.embeddings),
        len(response.embeddings[0]) if response.embeddings else 0,
    )

    return response.embeddings


# ---------------------------------------------------------
# Step 4: Store in ChromaDB
# ---------------------------------------------------------

def store_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: str,
) -> int:
    """Store chunks + embeddings in ChromaDB.

    How ChromaDB stores data:
      Each record has:
        - id: Unique string identifier
        - document: The text content
        - embedding: The vector (list of floats)
        - metadata: Any extra info (source, page, etc.)

    We generate deterministic IDs using a hash of the
    document_id + chunk_index. This means:
      - Re-uploading the same PDF won't create duplicates.
      - Each chunk has a predictable, unique ID.

    Args:
        chunks: Output from chunk_pages().
        embeddings: Output from generate_embeddings().
        document_id: Unique identifier for this document.

    Returns:
        Number of chunks stored.
    """

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        # Deterministic ID: hash of document + chunk index
        chunk_id = hashlib.md5(
            f"{document_id}_{i}".encode()
        ).hexdigest()

        ids.append(chunk_id)
        documents.append(chunk["text"])

        # ChromaDB metadata values must be str, int,
        # float, or bool — no nested objects.
        metadata = {
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "document_id": document_id,
        }

        metadatas.append(metadata)

    # upsert = insert or update. If the chunk already
    # exists (same ID), it gets replaced instead of
    # creating a duplicate.

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "Stored %d chunks for document '%s'",
        len(ids),
        document_id,
    )

    # Rebuild BM25 index to include the new chunks
    bm25_index.rebuild(collection)

    return len(ids)


# ---------------------------------------------------------
# Step 5: Search by similarity
# ---------------------------------------------------------

def _semantic_search(
    query: str,
    n_results: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """Semantic search using ChromaDB embeddings.

    How it works:
      1. Convert the query into an embedding vector.
      2. Ask ChromaDB to find the N stored vectors that
         are closest (most similar) to the query vector.
      3. Return those chunks with their metadata.

    "Cosine similarity" measures the angle between two
    vectors. Vectors pointing in the same direction
    (similar meaning) have high similarity (~1.0).
    Vectors pointing in different directions (unrelated
    meaning) have low similarity (~0.0).
    """

    if collection.count() == 0:
        return []

    query_embedding = generate_embeddings([query])[0]

    where_filter = None
    if source_filter:
        where_filter = {"source": source_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        where=where_filter,
    )

    search_results = []

    for i in range(len(results["ids"][0])):
        search_results.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get(
                "source", "unknown"
            ),
            "page": results["metadatas"][0][i].get(
                "page", 0
            ),
            "distance": results["distances"][0][i],
        })

    return search_results


# ---------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------
#
# When combining results from different search systems,
# you can't compare their raw scores — cosine distance
# (0 to 1) and BM25 scores (0 to infinity) are on
# completely different scales.
#
# RRF solves this by using only the RANK (position) of
# each result, not the score:
#
#   rrf_score = sum( 1 / (k + rank) )
#               for each retriever that returned it
#
# A chunk ranked #1 by both retrievers gets:
#   1/(60+1) + 1/(60+1) = 0.0328
#
# A chunk ranked #1 by only one gets:
#   1/(60+1) = 0.0164
#
# k=60 is the standard value from the original RRF paper
# (Cormack et al., 2009). It dampens the advantage of
# being #1 vs #2, making fusion more stable.

RRF_K = 60


def _reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """Merge two ranked result lists using RRF.

    Deduplicates by chunk text. Returns results sorted
    by RRF score (highest = most relevant).
    """

    # Track RRF scores and keep the best metadata
    # for each unique chunk.
    scores: dict[str, float] = {}
    result_data: dict[str, dict] = {}

    # Score semantic results (rank is 1-based)
    for rank, result in enumerate(semantic_results, 1):
        key = result["text"]
        scores[key] = scores.get(key, 0) + (
            1.0 / (RRF_K + rank)
        )
        if key not in result_data:
            result_data[key] = result

    # Score BM25 results
    for rank, result in enumerate(bm25_results, 1):
        key = result["text"]
        scores[key] = scores.get(key, 0) + (
            1.0 / (RRF_K + rank)
        )
        if key not in result_data:
            result_data[key] = result

    # Sort by RRF score descending
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Build final result list with RRF score
    fused = []
    for text, rrf_score in ranked:
        entry = result_data[text].copy()
        entry["rrf_score"] = round(rrf_score, 6)
        fused.append(entry)

    return fused


def get_page_chunks(
    page_num: int,
    source_filter: str | None = None,
) -> list[Evidence]:
    """Fetch ALL chunks from a specific page number.

    WHY THIS EXISTS:

      When the user asks "show me page 4" or "what's on page 3",
      semantic search fails because page numbers have no semantic
      meaning — page 4 content could be about anything.

      Instead of searching by meaning, we go directly to ChromaDB's
      metadata and fetch ALL chunks tagged with page=4. This
      guarantees we get the right page's content every time.

    HOW IT WORKS:

      ChromaDB stores metadata for each chunk:
        {"source": "report.pdf", "page": 4, "chunk_index": 7, ...}

      We use ChromaDB's `where` filter to fetch chunks where
      page == page_num. If source_filter is also provided, we
      combine both filters with $and so we only get chunks
      from that specific document + page.

    Args:
        page_num: The page number to fetch (1-based, matching
                  how users think about pages: "page 1", "page 2").
        source_filter: Optional filename to scope to one document.

    Returns:
        List of Evidence objects containing all chunks from
        that page, sorted by chunk_index (reading order).
    """

    if collection.count() == 0:
        return []

    # Build a metadata filter for ChromaDB.
    #
    # ChromaDB's `where` clause supports:
    #   {"page": 4}                    — single condition
    #   {"$and": [{"page": 4}, ...]}   — multiple conditions
    #
    # We always filter by page number. If the user selected
    # a specific document (source_filter), we add that too
    # so we don't accidentally return page 4 from a DIFFERENT
    # uploaded document.
    if source_filter:
        where_filter = {
            "$and": [
                {"page": page_num},
                {"source": source_filter},
            ]
        }
    else:
        where_filter = {"page": page_num}

    try:
        results = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.warning(
            "Page-specific query failed: %s", exc,
        )
        return []

    if not results["documents"]:
        logger.info(
            "No chunks found for page %d (filter=%s)",
            page_num,
            source_filter or "all",
        )
        return []

    # Build Evidence objects from the results.
    # Sort by chunk_index so the text appears in
    # reading order (top of page → bottom of page).
    evidence = []

    for i in range(len(results["documents"])):
        evidence.append(
            Evidence(
                text=results["documents"][i],
                source=results["metadatas"][i].get(
                    "source", "unknown",
                ),
                page=results["metadatas"][i].get(
                    "page", 0,
                ),
                score=1.0,  # direct fetch = perfect match
            )
        )

    # Sort by chunk_index for reading order
    evidence.sort(
        key=lambda e: e.page,
    )

    logger.info(
        "Page %d: fetched %d chunks directly "
        "(filter=%s)",
        page_num,
        len(evidence),
        source_filter or "all",
    )

    return evidence


def search_documents(
    query: str,
    n_results: int = 5,
    source_filter: str | None = None,
) -> list[Evidence]:
    """Hybrid search: semantic + BM25, merged with RRF.

    Runs BOTH retrievers and merges with Reciprocal Rank
    Fusion. This gives the best of both worlds:

      - Semantic search finds meaning-similar chunks
        ("revenue" matches "income")
      - BM25 finds exact keyword matches
        ("Rahul Kumar" matches "Rahul Kumar")

    Args:
        query: The user's question.
        n_results: How many results to return (default 5).
        source_filter: Optional filename to search within
                       only one document.

    Returns:
        List of Evidence objects sorted by RRF score.
    """

    logger.info(
        "Hybrid search for: '%s' (n=%d, filter=%s)",
        query[:80],
        n_results,
        source_filter or "all",
    )

    if collection.count() == 0:
        logger.info("No documents in collection")
        return []

    # Fetch more candidates than needed from each
    # retriever. After deduplication and fusion, we
    # want at least n_results unique chunks.
    fetch_count = n_results * 2

    # 1. Semantic search (embeddings / cosine similarity)
    semantic_results = _semantic_search(
        query, fetch_count, source_filter
    )

    # 2. BM25 keyword search
    bm25_results = bm25_index.search(
        query, fetch_count, source_filter
    )

    logger.info(
        "Semantic: %d results, BM25: %d results",
        len(semantic_results),
        len(bm25_results),
    )

    # 3. Merge with Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(
        semantic_results, bm25_results
    )

    # 4. Return top n_results as Evidence objects
    top = fused[:n_results]

    evidence = [
        Evidence(
            text=r["text"],
            source=r["source"],
            page=r["page"],
            score=r.get("rrf_score", 0),
        )
        for r in top
    ]

    if evidence:
        logger.info(
            "Hybrid results: %d (best RRF=%.6f)",
            len(evidence),
            evidence[0].score,
        )

    return evidence


# ---------------------------------------------------------
# High-level: Process a PDF end-to-end
# ---------------------------------------------------------

# ---------------------------------------------------------
# Phase 8: Image/Diagram Extraction from PDFs
# ---------------------------------------------------------
#
# WHY EXTRACT IMAGES?
#
#   PDFs often contain embedded images — charts, diagrams,
#   signatures, logos, photos, infographics. Text extraction
#   captures the WORDS around a chart, but not the chart
#   itself. By extracting images separately, users can:
#     - Browse visual content from their documents
#     - See charts/diagrams that text search misses
#     - Download specific images for reports
#
# HOW IT WORKS:
#
#   PyMuPDF stores each image as a numbered "xref" (cross-
#   reference) object inside the PDF. A page can reference
#   multiple xrefs. We:
#     1. Walk every page, call page.get_images() to list
#        image xrefs on that page
#     2. For each unique xref, call doc.extract_image() to
#        get the raw bytes + format (png, jpeg, etc.)
#     3. Skip tiny images (<5KB) — these are usually icons,
#        bullets, decorative dots, or 1-pixel spacers
#     4. Cap at 2MB per image — larger images eat disk and
#        slow down the thumbnail gallery
#     5. Save each image to uploads/images/{document_id}/
#        with a descriptive filename: page{N}_img{M}.{ext}
#
# OPTIMIZATION — DEDUPLICATION:
#
#   The same image xref can appear on multiple pages (e.g.,
#   a company logo on every page header). We track seen
#   xrefs in a set to avoid saving the same image twice.
#   This often cuts image count by 50-80% for documents
#   with repeated headers/footers.
#
# DIRECTORY STRUCTURE:
#
#   uploads/
#     images/
#       abc123def456/          ← document_id
#         page1_img1.jpeg
#         page1_img2.png
#         page3_img1.jpeg
#       def789abc012/
#         page1_img1.png

# Size thresholds for image extraction
IMAGE_MIN_BYTES = 5 * 1024       # 5 KB — skip tiny icons
IMAGE_MAX_BYTES = 2 * 1024 * 1024  # 2 MB — cap large images

# Where extracted images are stored
IMAGES_DIR = Path(__file__).resolve().parents[2] / "uploads" / "images"


def extract_images_from_pdf(
    pdf_path: str,
    document_id: str,
) -> list[dict]:
    """Extract embedded images from a PDF file.

    Walks every page, finds image objects via their xref
    (cross-reference ID), deduplicates by xref, filters
    by size, and saves to disk.

    Args:
        pdf_path: Path to the PDF file on disk.
        document_id: Unique ID for this document (used
                     to create the output subdirectory).

    Returns:
        List of dicts describing extracted images:
        [
            {
                "filename": "page1_img1.jpeg",
                "page": 1,
                "size_bytes": 45678,
                "width": 800,
                "height": 600,
                "format": "jpeg",
            },
            ...
        ]
        Empty list if no images found or PDF can't be opened.
    """

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning(
            "Can't open PDF for image extraction: %s",
            exc,
        )
        return []

    # Create output directory for this document's images.
    # Each document gets its own subdirectory to keep
    # things organized and make cleanup easy.
    img_dir = IMAGES_DIR / document_id
    img_dir.mkdir(parents=True, exist_ok=True)

    # Track which image xrefs we've already saved.
    # A single image object (identified by xref number)
    # can appear on multiple pages — e.g., a logo in
    # every page header. We only save it once.
    seen_xrefs: set[int] = set()

    extracted: list[dict] = []

    # Counter for naming images within each page.
    # Reset per page so filenames are: page1_img1, page1_img2, ...
    for page_num in range(len(doc)):

        page = doc[page_num]

        # get_images() returns a list of tuples:
        #   (xref, smask, width, height, bpc, colorspace, ...)
        #
        # xref = the image's cross-reference ID inside the PDF.
        #        This is how PyMuPDF identifies each image object.
        #
        # smask = soft mask xref (for transparency). We don't
        #         need it for extraction.
        #
        # full=True gives us all fields including color info.
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        img_counter = 0

        for img_info in image_list:

            xref = img_info[0]  # Image cross-reference ID

            # ----- DEDUPLICATION -----
            # Skip if we already extracted this image from
            # a previous page. Common for logos, watermarks,
            # and header/footer graphics.
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                # extract_image() returns a dict with:
                #   "image": raw bytes of the image
                #   "ext": file extension ("png", "jpeg", etc.)
                #   "width": pixel width
                #   "height": pixel height
                #   "cs-name": color space name
                #
                # It decodes the image from the PDF's internal
                # format (which might be DCTDecode for JPEG,
                # FlateDecode for PNG, etc.) into usable bytes.
                img_data = doc.extract_image(xref)
            except Exception as exc:
                logger.debug(
                    "Skipping xref %d: extract failed (%s)",
                    xref,
                    exc,
                )
                continue

            if not img_data:
                continue

            raw_bytes = img_data["image"]
            img_ext = img_data["ext"]
            img_width = img_data.get("width", 0)
            img_height = img_data.get("height", 0)

            # ----- SIZE FILTERING -----
            # Skip tiny images: these are almost always
            # decorative — bullet points, line separators,
            # 1×1 tracking pixels, small icons.
            size_bytes = len(raw_bytes)

            if size_bytes < IMAGE_MIN_BYTES:
                logger.debug(
                    "Skipping xref %d: too small "
                    "(%d bytes < %d minimum)",
                    xref,
                    size_bytes,
                    IMAGE_MIN_BYTES,
                )
                continue

            # Skip oversized images to prevent disk bloat.
            # 2MB is generous for document images — even a
            # high-res chart rarely exceeds 500KB.
            if size_bytes > IMAGE_MAX_BYTES:
                logger.info(
                    "Skipping xref %d: too large "
                    "(%d bytes > %d maximum)",
                    xref,
                    size_bytes,
                    IMAGE_MAX_BYTES,
                )
                continue

            # ----- SAVE TO DISK -----
            # Naming convention: page{N}_img{M}.{ext}
            # N = page number (1-based, user-friendly)
            # M = image index within that page (1-based)
            img_counter += 1
            img_filename = (
                f"page{page_num + 1}_img{img_counter}"
                f".{img_ext}"
            )
            img_path = img_dir / img_filename

            img_path.write_bytes(raw_bytes)

            extracted.append({
                "filename": img_filename,
                "page": page_num + 1,
                "size_bytes": size_bytes,
                "width": img_width,
                "height": img_height,
                "format": img_ext,
            })

            logger.debug(
                "Extracted image: %s (%dx%d, %d bytes)",
                img_filename,
                img_width,
                img_height,
                size_bytes,
            )

    doc.close()

    # If no images were extracted, clean up the empty
    # directory to avoid clutter.
    if not extracted:
        try:
            img_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other issue
        logger.info(
            "No extractable images found in PDF: %s",
            pdf_path,
        )
    else:
        logger.info(
            "Extracted %d images from PDF: %s "
            "(skipped %d duplicates)",
            len(extracted),
            pdf_path,
            len(seen_xrefs) - len(extracted),
        )

    return extracted


def list_document_images(
    document_id: str,
) -> list[dict]:
    """List all extracted images for a document.

    Reads the image directory for the given document_id
    and returns metadata about each image file.

    Args:
        document_id: The document's unique ID.

    Returns:
        List of dicts with filename, size, and format info.
        Empty list if no images directory exists.
    """

    img_dir = IMAGES_DIR / document_id

    if not img_dir.exists():
        return []

    images = []

    for img_path in sorted(img_dir.iterdir()):

        if not img_path.is_file():
            continue

        # Parse page number from filename (page3_img1.jpeg → 3)
        name = img_path.stem  # "page3_img1"
        page_num = 0

        try:
            # Extract the number after "page" and before "_"
            page_part = name.split("_")[0]  # "page3"
            page_num = int(
                page_part.replace("page", "")
            )
        except (ValueError, IndexError):
            pass

        images.append({
            "filename": img_path.name,
            "page": page_num,
            "size_bytes": img_path.stat().st_size,
            "format": img_path.suffix.lstrip("."),
        })

    return images


def delete_document_images(
    document_id: str,
) -> bool:
    """Delete all extracted images for a document.

    Called when a document is deleted — cleans up the
    images directory so we don't leave orphaned files.

    Args:
        document_id: The document's unique ID.

    Returns:
        True if images were deleted, False if no images
        directory existed.
    """

    img_dir = IMAGES_DIR / document_id

    if not img_dir.exists():
        return False

    # Delete all image files in the directory
    for img_path in img_dir.iterdir():
        if img_path.is_file():
            img_path.unlink()

    # Remove the now-empty directory
    try:
        img_dir.rmdir()
    except OSError:
        pass

    logger.info(
        "Deleted images for document: %s",
        document_id,
    )

    return True


def process_document(
    file_path: str,
    filename: str,
    document_id: str,
) -> dict:
    """Process any supported document: extract → chunk →
    embed → store.

    This is the main entry point called by the API when
    a user uploads a file. It dispatches to the right
    extractor based on file extension, then runs the
    chunking → embedding → storage pipeline.

    Supported formats:
      - .pdf  — text + headings + tables via PyMuPDF
      - .docx — text + headings + tables via python-docx
      - .txt  — plain text
      - .csv  — tabular data

    Args:
        file_path: Path to the file on disk.
        filename: Original filename (for metadata).
        document_id: Unique ID for this upload.

    Returns:
        Summary dict with status, page/chunk counts.
    """

    ext = Path(filename).suffix.lower()

    logger.info(
        "Processing document: %s (type=%s, id=%s)",
        filename,
        ext,
        document_id,
    )

    # Step 1: Extract — pick the right extractor
    #
    # Each file type has its own extractor that understands
    # its format. All extractors return the same structure:
    #   [{"page": N, "text": "..."}]
    #
    # This uniform output means Steps 2-4 (chunk → embed →
    # store) work identically regardless of input format.

    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)

        # Extract tables to internal CSV so analysis tools
        # (filter_rows, aggregate_data) can compute on PDF
        # table data with 100% accuracy instead of guessing
        # from text chunks.
        upload_dir = Path(file_path).parent
        csv_path = extract_tables_to_csv(
            file_path, document_id, upload_dir,
        )
        if csv_path:
            logger.info(
                "PDF table extracted to CSV: %s",
                csv_path.name,
            )

        # Phase 8: Extract embedded images from the PDF.
        #
        # WHY DO THIS DURING PROCESSING?
        #
        #   Extracting images at upload time means they're
        #   ready to browse instantly — no second pass needed.
        #   The overhead is minimal (~100ms for a typical PDF)
        #   compared to the embedding step (~5-30 seconds).
        #
        #   Images are saved to disk (not embedded in the
        #   vector store) because:
        #     1. They're binary data, not searchable text
        #     2. ChromaDB is for text embeddings, not files
        #     3. Serving from disk is fast and simple
        extracted_images = extract_images_from_pdf(
            file_path, document_id,
        )
        if extracted_images:
            logger.info(
                "Extracted %d images from PDF: %s",
                len(extracted_images),
                filename,
            )

    elif ext in IMAGE_EXTENSIONS:
        # Phase 2: Image files → OCR extraction
        #
        # The user uploaded a photo/screenshot of a document.
        # We run Tesseract OCR to extract the text, then
        # process it through the same chunking → embedding
        # pipeline as any other document.
        #
        # Common use cases:
        #   - Photo of a paper invoice or receipt
        #   - Screenshot of a report or spreadsheet
        #   - Scanned page saved as .jpg instead of .pdf
        pages = extract_text_from_image(file_path)

        if not pages and not OCR_AVAILABLE:
            return {
                "document_id": document_id,
                "filename": filename,
                "pages": 0,
                "chunks": 0,
                "status": "error",
                "message": (
                    "Cannot process images: Tesseract OCR "
                    "is not installed. Please install "
                    "Tesseract to enable image uploads."
                ),
            }

    elif ext == ".txt":
        pages = extract_text_from_txt(file_path)
    elif ext == ".csv":
        pages = extract_text_from_csv(file_path)
    elif ext == ".docx":
        pages = extract_text_from_docx(file_path)
    else:
        return {
            "document_id": document_id,
            "filename": filename,
            "pages": 0,
            "chunks": 0,
            "status": "error",
            "message": f"Unsupported file type: {ext}",
        }

    if not pages:
        return {
            "document_id": document_id,
            "filename": filename,
            "pages": 0,
            "chunks": 0,
            "status": "error",
            "message": "No text found in document.",
        }

    # Step 2: Chunk
    chunks = chunk_pages(pages, filename)

    if not chunks:
        return {
            "document_id": document_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": 0,
            "status": "error",
            "message": "No chunks created from text.",
        }

    # Step 3: Embed
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)

    # Step 4: Store
    stored = store_chunks(chunks, embeddings, document_id)

    # Count extracted images (only for PDFs).
    # The `extracted_images` variable only exists if we
    # went through the PDF branch above. For other file
    # types, there are no embedded images to extract.
    image_count = 0
    if ext == ".pdf":
        try:
            image_count = len(extracted_images)
        except NameError:
            image_count = 0

    result = {
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
        "chunks": stored,
        "images": image_count,
        "status": "success",
    }

    logger.info(
        "Document processed: %s — %d pages, %d chunks, "
        "%d images",
        filename,
        len(pages),
        stored,
        image_count,
    )

    return result


# ---------------------------------------------------------
# get_document_chunk_count — count chunks for ONE document
# ---------------------------------------------------------
#
# WHY A SEPARATE FUNCTION (instead of using list_documents)?
#
#   list_documents() loads ALL metadata for EVERY document
#   in the vector store, just to count chunks. That's
#   wasteful when we only need the count for ONE document.
#
#   This function uses ChromaDB's `where` filter to query
#   only chunks belonging to a specific file. ChromaDB
#   returns just the matching IDs (no embeddings, no text),
#   so it's very lightweight — even for documents with
#   hundreds of chunks.
#
# USED BY:
#   agent.py's build_context_prompt() to dynamically decide
#   how many chunks to retrieve based on document size.
#   Small document (20 chunks) → retrieve ~8
#   Large document (200 chunks) → retrieve ~20
#
#   This replaces the old hardcoded n_results=5, which
#   only covered ~10% of a large document's content.

def get_document_chunk_count(
    source: str | None = None,
) -> int:
    """Count how many chunks exist for a document.

    Args:
        source: Filename to count chunks for (e.g. "report.pdf").
                If None, returns total chunks across ALL documents.

    Returns:
        Number of chunks in ChromaDB for the given document.
        Returns 0 if the document doesn't exist or collection
        is empty.
    """

    # If no source specified, return total count
    # (useful for "how many chunks total?" diagnostics)
    if source is None:
        return collection.count()

    # -------------------------------------------------
    # WHY collection.get() WITH where FILTER?
    #
    #   ChromaDB doesn't have a direct "count where" API.
    #   But collection.get() with a where filter returns
    #   only matching document IDs. We don't request
    #   embeddings or documents (text), so the response
    #   is tiny — just a list of ID strings.
    #
    #   len(result["ids"]) gives us the exact chunk count
    #   for this specific file.
    # -------------------------------------------------
    try:
        result = collection.get(
            where={"source": source},
            include=[],  # No embeddings, no text — just IDs
        )
        return len(result["ids"])
    except Exception as exc:
        logger.warning(
            "Failed to count chunks for '%s': %s",
            source, exc,
        )
        return 0


def list_documents() -> list[dict]:
    """List all uploaded documents.

    Returns a summary of each unique document in the
    vector store.
    """

    if collection.count() == 0:
        return []

    # Get all metadata to find unique documents
    all_data = collection.get(
        include=["metadatas"],
    )

    # Group by document_id
    docs = {}

    for metadata in all_data["metadatas"]:

        doc_id = metadata.get("document_id", "unknown")

        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "filename": metadata.get(
                    "source", "unknown"
                ),
                "chunks": 0,
                "pages": set(),
            }

        docs[doc_id]["chunks"] += 1
        docs[doc_id]["pages"].add(
            metadata.get("page", 0)
        )

    # Convert sets to counts
    result = []

    for doc in docs.values():
        result.append(
            {
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "chunks": doc["chunks"],
                "pages": len(doc["pages"]),
            }
        )

    return result


def delete_document(document_id: str) -> bool:
    """Delete all chunks AND extracted images for a document.

    Phase 8 addition: also deletes the images directory
    for this document, so no orphaned image files remain
    on disk after document deletion.

    Args:
        document_id: The document to delete.

    Returns:
        True if chunks were deleted, False if none found.
    """

    try:
        collection.delete(
            where={"document_id": document_id},
        )

        # Rebuild BM25 index without the deleted chunks
        bm25_index.rebuild(collection)

        # Phase 8: Clean up extracted images from disk.
        # Even if there are no images, this is a no-op
        # (returns False), so it's safe to call always.
        delete_document_images(document_id)

        logger.info(
            "Deleted document: %s",
            document_id,
        )

        return True

    except Exception as exc:

        logger.error(
            "Failed to delete document %s: %s",
            document_id,
            exc,
        )

        return False