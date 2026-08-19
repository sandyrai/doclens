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
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF — the import name is "fitz"
from ollama import embed
from rank_bm25 import BM25Okapi

import chromadb


logger = logging.getLogger(__name__)


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
EMBEDDING_MODEL = "nomic-embed-text"


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

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx"}


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


# Common short English words — used by _smart_join_cells
# to distinguish real word boundaries from split fragments.
# If a cell ends with one of these words and the next cell
# starts lowercase, it's a real boundary (keep the space).
# If the last word ISN'T in this set, it's probably a
# fragment and the cells should be merged without a space.

_REAL_WORDS = {
    "a", "i", "an", "am", "as", "at", "be", "by",
    "do", "go", "he", "if", "in", "is", "it", "me",
    "my", "no", "of", "ok", "on", "or", "so", "to",
    "up", "us", "we", "oh", "the", "and", "for",
    "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his",
    "how", "its", "may", "new", "now", "old", "see",
    "two", "way", "who", "did", "get", "let", "say",
    "she", "too", "use", "per", "min", "max", "avg",
    "each", "from", "have", "been", "more", "when",
    "will", "with", "into", "that", "this", "them",
    "then", "than", "also", "just", "only", "over",
    "such", "very", "some", "year", "most", "much",
}


def _smart_join_cells(cells: list[str]) -> str:
    """Join table cells with smart word-boundary detection.

    Problem:
      PDF table extractors split words across cells, creating
      fragments like ['Class T', 'eacher:'] or ['Examin',
      'ation']. Naively joining with spaces gives broken text:
      "Class T eacher:" instead of "Class Teacher:".

    Solution:
      When a cell ends with letters and the next cell starts
      with a lowercase letter, they're probably one word that
      was split at the cell boundary → join WITHOUT a space.

      Exception: if the last word of the previous cell is a
      common English word (like "in", "the", "for"), it's a
      real word boundary → keep the space.

    Examples:
      ['Class T', 'eacher:']   → 'Class Teacher:'
      ['Examin', 'ation']      → 'Examination'
      ['minimum in', 'each']   → 'minimum in each'
    """

    if not cells:
        return ""

    parts = [cells[0]]

    for cell in cells[1:]:
        if not cell:
            continue

        prev = parts[-1] if parts else ""

        if (
            prev
            and prev[-1].isalpha()
            and cell[0].islower()
        ):
            # Check: is the last word of prev a real word?
            # If so, this is a real word boundary.
            last_word = (
                prev.split()[-1] if prev.strip() else ""
            )
            # Strip trailing punctuation for the check
            last_word_clean = last_word.rstrip(":.,;!?")

            if last_word_clean.lower() in _REAL_WORDS:
                # Real word boundary — keep space
                parts.append(" " + cell)
            else:
                # Split word — merge without space
                parts.append(cell)
        else:
            # Normal join with space
            parts.append(" " + cell)

    return "".join(parts).strip()


def _clean_extracted_table(
    headers: list[str],
    rows: list[list[str]],
) -> tuple[list[str], list[list[str]], str]:
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
      5. Filter out non-data rows (summary text, footer
         notes) — rows where most cells are empty.
      6. Capture metadata (rows above header) and summary
         (rows below data) as human-readable text so the
         LLM can answer questions about school name, class
         teacher, grading policy, etc.

    Returns:
        Tuple of (clean_headers, clean_data_rows, metadata).
        metadata: Human-readable text from rows above the
          header (school name, teacher name, etc.) and below
          the data (class summary, grading policy, etc.).
          These rows are NOT in the CSV but contain important
          context the LLM needs for answering questions.
        Returns original data + empty metadata if no better
        header found.
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
        return headers, rows, ""

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
        return headers, rows, ""

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

    # Track the index (in non_empty_rows) of the last row
    # we added to clean_rows. We need this in Step 6 to
    # know where the data section ends and the summary
    # section begins.
    last_data_row_idx = header_idx

    for idx, row in enumerate(
        non_empty_rows[header_idx + 1:],
        start=header_idx + 1,
    ):
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
        last_data_row_idx = idx

    if not clean_rows:
        return headers, rows, ""

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

    # -------------------------------------------------
    # Step 6: Capture metadata and summary text
    # -------------------------------------------------
    #
    # Why this matters:
    #   The rows ABOVE the header contain document metadata
    #   like "Delhi Public School, Whitefield" and "Class
    #   Teacher: Mrs. Sunita Krishnamurthy". The rows BELOW
    #   the data contain summary info like "Class Average:
    #   72.2%" and "Pass Rate: 13/15".
    #
    #   These rows are stripped from the CSV (they're not
    #   tabular data), but they contain important context
    #   the LLM needs to answer questions like "who is the
    #   class teacher?" or "what is the grading policy?"
    #
    #   We join these cells into readable text lines and
    #   return them as a metadata string.

    metadata_lines: list[str] = []

    # Rows ABOVE the header = document metadata
    # (school name, exam title, teacher name, etc.)
    for row in non_empty_rows[:header_idx]:
        # Skip rows that are just auto-generated ColN
        # placeholder names (e.g. "Col0 Col1 Col2...").
        # These are garbage from the extractor, not real
        # metadata.
        non_empty_cells = [c for c in row if c]
        auto_cols = sum(
            1 for c in non_empty_cells
            if _col_pattern.match(c)
        )
        if auto_cols >= 2:
            continue

        # Use smart join to merge word fragments that
        # were split across cells. This turns:
        #   ['Class T', 'eacher:', 'Mrs. S', 'unita']
        # into:
        #   'Class Teacher: Mrs. Sunita'
        # instead of the broken:
        #   'Class T eacher: Mrs. S unita'
        text = _smart_join_cells(non_empty_cells)
        if text:
            metadata_lines.append(text)

    # Rows BELOW the data = summary/footer
    # (class average, pass rate, grading policy, etc.)
    #
    # last_data_row_idx was tracked during Step 4 — it's
    # the index in non_empty_rows of the last row we
    # added to clean_rows. Everything after that index
    # (that isn't a data row) is summary text.

    # Collect summary rows (after the last data row)
    for row in non_empty_rows[last_data_row_idx + 1:]:
        non_empty_cells = [c for c in row if c]
        text = _smart_join_cells(non_empty_cells)
        if text:
            metadata_lines.append(text)

    metadata_text = "\n".join(metadata_lines)

    if metadata_lines:
        logger.info(
            "Captured %d metadata/summary lines "
            "(%d chars)",
            len(metadata_lines),
            len(metadata_text),
        )

    logger.info(
        "Table cleaning: found header at row %d, "
        "%d data rows, %d columns (was %d raw rows)",
        header_idx,
        len(clean_rows),
        len(clean_headers),
        len(rows),
    )

    return clean_headers, clean_rows, metadata_text


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

    best_headers: list[str] | None = None
    best_rows: list[list[str]] = []

    # ----- Strategy 1: find_tables (lines) -----

    h1, r1 = _extract_with_find_tables(doc, "lines")

    logger.info(
        "Strategy 'lines': %d rows extracted",
        len(r1),
    )

    if h1 and len(r1) > len(best_rows):
        best_headers = h1
        best_rows = r1

    # ----- Strategy 2: find_tables (text) -----

    h2, r2 = _extract_with_find_tables(doc, "text")

    logger.info(
        "Strategy 'text': %d rows extracted",
        len(r2),
    )

    if h2 and len(r2) > len(best_rows):
        best_headers = h2
        best_rows = r2

    # ----- Strategy 3: word-position parsing -----

    h3, r3 = _extract_with_text_positions(doc)

    logger.info(
        "Strategy 'positions': %d rows extracted",
        len(r3),
    )

    if h3 and len(r3) > len(best_rows):
        best_headers = h3
        best_rows = r3

    doc.close()

    if not best_headers or not best_rows:
        logger.info(
            "No tables found in PDF for CSV extraction",
        )
        return None

    # Clean up: find real header row, remove empty/title
    # rows, fix column names. This is critical because
    # find_tables often grabs page titles, subtitles, and
    # metadata as table rows — the first row might be the
    # school name, not the actual column headers.
    #
    # Also captures metadata text (school name, teacher,
    # summary stats, grading policy) that was stripped
    # from the table. This gets saved to a companion file
    # so the LLM can still answer questions about it.
    best_headers, best_rows, metadata = (
        _clean_extracted_table(best_headers, best_rows)
    )

    if not best_rows:
        logger.info(
            "No data rows after table cleaning",
        )
        return None

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

    # Save metadata to companion file if any was captured.
    #
    # Why a separate file?
    #   The CSV should be pure tabular data (headers + rows)
    #   so csv.DictReader can parse it cleanly. Metadata
    #   like "Class Teacher: Mrs. Sunita Krishnamurthy" is
    #   free-form text, not a table row. Storing it in a
    #   companion .txt file keeps the CSV clean while still
    #   preserving this info for the LLM.
    #
    # The companion file is named:
    #   {document_id}_table_metadata.txt
    # and lives alongside the CSV in the uploads directory.

    if metadata:
        meta_filename = (
            f"{document_id}_table_metadata.txt"
        )
        meta_path = upload_dir / meta_filename

        with open(meta_path, "w",
                  encoding="utf-8") as f:
            f.write(metadata)

        logger.info(
            "Saved table metadata: %s (%d chars)",
            meta_filename,
            len(metadata),
        )

    return csv_path


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

    Enhanced with heading detection and table extraction.
    Headings are marked with [SECTION: ...] and tables
    with [TABLE]...[/TABLE] so the LLM can understand
    the document structure.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A list of dicts, one per page:
        [{"page": 1, "text": "..."}, ...]
    """

    logger.info("Extracting text from: %s", pdf_path)

    doc = fitz.open(pdf_path)

    pages = []

    for page_num in range(len(doc)):

        page = doc[page_num]

        # Extract text with heading markers
        text = _extract_text_with_headings(page)

        # Extract tables separately (get_text often
        # garbles table data — find_tables is better)
        table_text = _extract_tables_from_page(page)

        if table_text:
            text = text + "\n\n" + table_text

        if text.strip():
            pages.append(
                {
                    "page": page_num + 1,
                    "text": text.strip(),
                }
            )

    doc.close()

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

    result = {
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
        "chunks": stored,
        "status": "success",
    }

    logger.info(
        "Document processed: %s — %d pages, %d chunks",
        filename,
        len(pages),
        stored,
    )

    return result



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
    """Delete all chunks for a document.

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