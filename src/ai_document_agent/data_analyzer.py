"""Data analysis engine for tabular documents.

Why this module exists:

  When a user asks "how many Hard questions?" or "average
  score by category?", the normal RAG flow sends 5 random
  text chunks to the LLM. The LLM tries to count from
  fragments — and gets it wrong.

  This module loads the ACTUAL structured data (CSV rows)
  and computes answers in Python. Python counts, sums,
  and groups with 100% accuracy in <1ms. The computed
  results are injected into the LLM context so it can
  format a nice answer from real numbers.

  Think of it this way:
    - Chunk retrieval = "read a few pages and guess"
    - Data analysis   = "open the spreadsheet and compute"

How it fits in the pipeline:

  1. User asks a question about a CSV document
  2. agent.py detects the source is a CSV file
  3. This module loads the CSV and computes stats
  4. Results are injected into the LLM prompt alongside
     the chunk excerpts
  5. The LLM uses the COMPUTED data to write its answer

What this module does NOT do:
  - No LLM calls (zero latency cost)
  - No complex NLP to parse the question
  - No pandas dependency (just stdlib csv module)
"""

import csv
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Load CSV data
# ---------------------------------------------------------

def load_csv(
    file_path: str,
) -> tuple[list[str], list[dict]]:
    """Load a CSV file into headers + list of row dicts.

    Args:
        file_path: Path to the CSV file on disk.

    Returns:
        Tuple of (headers, rows).
        headers: List of column names from the first row.
        rows: List of dicts, one per data row.

    Example:
        headers = ["ID", "Name", "Score"]
        rows = [
            {"ID": "1", "Name": "Alice", "Score": "95"},
            {"ID": "2", "Name": "Bob",   "Score": "87"},
        ]
    """

    with open(
        file_path, "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    logger.info(
        "Loaded CSV: %d rows, %d columns",
        len(rows),
        len(headers),
    )

    return headers, rows


# ---------------------------------------------------------
# Column type detection
# ---------------------------------------------------------
#
# CSV values are all strings. To compute sum/average/min/max,
# we need to know which columns contain numbers.
#
# Strategy: sample the first 10 non-empty values. If ALL of
# them parse as float, treat the column as numeric.
# This is a heuristic — not perfect, but good enough for
# real-world CSVs (IDs like "C-R2-001" will correctly be
# detected as non-numeric because of the letters).

def _is_numeric_value(value: str) -> bool:
    """Check if a single string value is a number."""

    try:
        float(value.replace(",", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def _to_number(value: str) -> float:
    """Convert string to float, handling commas."""

    return float(value.replace(",", "").strip())


def _detect_column_type(
    rows: list[dict],
    column: str,
) -> str:
    """Detect if a column is 'numeric' or 'categorical'.

    Samples up to 10 non-empty values. If all parse as
    float, it's numeric. Otherwise, categorical.
    """

    sample = []

    for row in rows:
        val = row.get(column, "").strip()
        if val:
            sample.append(val)
        if len(sample) >= 10:
            break

    if not sample:
        return "empty"

    if all(_is_numeric_value(v) for v in sample):
        return "numeric"

    return "categorical"


# ---------------------------------------------------------
# Build analysis context
# ---------------------------------------------------------
#
# This is the main function. It loads a CSV and builds
# a rich text summary that gets injected into the LLM's
# system prompt.
#
# The output includes:
#   1. Dataset overview (row count, column names)
#   2. Column analysis:
#      - Numeric columns: min, max, average, sum
#      - Categorical columns: unique values + counts
#   3. Raw data (all rows if ≤20, first 8 if larger)
#
# Why include raw data?
#   The LLM needs to see actual rows to answer questions
#   like "what is question C-R2-001 about?" or "list all
#   Hard questions". Stats alone aren't enough.
#
# Why limit to 20/8 rows?
#   Context size directly affects generation speed on CPU.
#   Every extra token in the prompt adds prefill time.
#   20 rows ≈ 1500-2000 chars — enough for the LLM to
#   see the data structure and answer most questions.
#   The LLM also has filter_rows and aggregate_data tools
#   to query the FULL dataset when it needs specific rows
#   beyond this sample. Reduced from 40/15 for ~40% less
#   prompt tokens.

MAX_FULL_DATA_ROWS = 20
MAX_SAMPLE_ROWS = 8


def build_analysis_context(
    file_path: str,
) -> str:
    """Load CSV and build analysis context for the LLM.

    Args:
        file_path: Path to the CSV file on disk.

    Returns:
        Formatted analysis text to inject into the LLM
        context. Empty string if the file can't be loaded.
    """

    try:
        headers, rows = load_csv(file_path)
    except Exception as exc:
        logger.warning(
            "Failed to load CSV for analysis: %s", exc,
        )
        return ""

    if not headers or not rows:
        return ""

    parts = [
        "\n\nDATA ANALYSIS (computed by Python, "
        "100% accurate):",
        f"Dataset: {Path(file_path).name}",
        f"Total rows: {len(rows)}",
        f"Columns ({len(headers)}): "
        + ", ".join(headers),
    ]

    # -------------------------------------------------------
    # Column-by-column analysis
    # -------------------------------------------------------

    parts.append("\nCOLUMN STATISTICS:")

    for col in headers:

        col_type = _detect_column_type(rows, col)
        values = [
            row.get(col, "").strip() for row in rows
        ]
        non_empty = [v for v in values if v]

        if not non_empty:
            parts.append(f"  {col}: all empty")
            continue

        if col_type == "numeric":
            nums = [
                _to_number(v) for v in non_empty
                if _is_numeric_value(v)
            ]

            if nums:
                avg = sum(nums) / len(nums)

                parts.append(
                    f"  {col} (numeric): "
                    f"count={len(nums)}, "
                    f"min={min(nums):g}, "
                    f"max={max(nums):g}, "
                    f"avg={avg:.2f}, "
                    f"sum={sum(nums):g}"
                )

        else:
            # Categorical — count unique values
            counts: dict[str, int] = {}

            for v in non_empty:
                counts[v] = counts.get(v, 0) + 1

            n_unique = len(counts)

            if n_unique <= 10:
                # Show all value counts (≤10 is compact)
                sorted_counts = sorted(
                    counts.items(),
                    key=lambda x: -x[1],
                )
                breakdown = ", ".join(
                    f"{k} ({v})"
                    for k, v in sorted_counts
                )
                parts.append(
                    f"  {col}: {n_unique} unique "
                    f"values — {breakdown}"
                )
            else:
                # Too many to list — show top 5
                # Reduced from top 10 to save tokens.
                # The LLM can use filter_rows to find
                # specific values when needed.
                sorted_counts = sorted(
                    counts.items(),
                    key=lambda x: -x[1],
                )[:5]
                breakdown = ", ".join(
                    f"{k} ({v})"
                    for k, v in sorted_counts
                )
                parts.append(
                    f"  {col}: {n_unique} unique values "
                    f"(top 10: {breakdown})"
                )

    # -------------------------------------------------------
    # Raw data inclusion
    # -------------------------------------------------------
    #
    # For small datasets: include everything so the LLM
    # can answer any question.
    # For large datasets: include a sample so the LLM
    # has examples but we don't blow up the context.

    header_line = " | ".join(headers)

    if len(rows) <= MAX_FULL_DATA_ROWS:

        parts.append(
            f"\nCOMPLETE DATA ({len(rows)} rows):"
        )
        parts.append(header_line)

        for row in rows:
            parts.append(
                " | ".join(
                    row.get(h, "") for h in headers
                )
            )
    else:

        parts.append(
            f"\nSAMPLE DATA (first {MAX_SAMPLE_ROWS} "
            f"of {len(rows)} rows):"
        )
        parts.append(header_line)

        for row in rows[:MAX_SAMPLE_ROWS]:
            parts.append(
                " | ".join(
                    row.get(h, "") for h in headers
                )
            )

        parts.append(
            f"... ({len(rows) - MAX_SAMPLE_ROWS} "
            f"more rows not shown)"
        )

    analysis_text = "\n".join(parts)

    logger.info(
        "Analysis context: %d chars for %d rows",
        len(analysis_text),
        len(rows),
    )

    return analysis_text


# ---------------------------------------------------------
# File path lookup
# ---------------------------------------------------------
#
# Upload files are saved as {document_id}_{filename} in
# the uploads directory. To find a file by its original
# filename, we scan the directory for a match.

def find_upload_path(
    filename: str,
    upload_dir: Path,
) -> Path | None:
    """Find an uploaded file by its original filename.

    The upload endpoint saves files as:
        {sha256_hash}_{original_filename}

    This function scans the upload directory for a file
    whose name ends with _{filename}.

    Args:
        filename: The original filename (e.g. "data.csv").
        upload_dir: The uploads directory path.

    Returns:
        Path to the file, or None if not found.
    """

    if not upload_dir.exists():
        return None

    for path in upload_dir.iterdir():
        if (
            path.is_file()
            and path.name.endswith(f"_{filename}")
        ):
            return path

    return None


def is_tabular_file(filename: str) -> bool:
    """Check if a file is tabular data (CSV).

    Currently only CSV is supported for analysis.
    Future: could add TSV, Excel support.
    """

    return filename.lower().endswith(".csv")


def find_extracted_table_csv(
    filename: str,
    upload_dir: Path,
) -> Path | None:
    """Find the extracted table CSV for a PDF document.

    When a PDF with tables is uploaded, process_document()
    extracts the tables into a CSV file named:
        {document_id}_extracted_table.csv

    This function scans the upload directory for that file
    by looking for files that match the pattern:
        *_extracted_table.csv
    where the document_id prefix matches the PDF's upload.

    Args:
        filename: The original PDF filename (e.g. "report.pdf").
        upload_dir: The uploads directory path.

    Returns:
        Path to the extracted CSV, or None if not found.
    """

    if not upload_dir.exists():
        return None

    # The PDF was saved as {document_id}_{filename}.
    # The extracted CSV is {document_id}_extracted_table.csv.
    # We find the PDF file first, then derive the CSV name.

    # Find the uploaded PDF to get its document_id prefix
    pdf_suffix = f"_{filename}"

    for path in upload_dir.iterdir():
        if (
            path.is_file()
            and path.name.endswith(pdf_suffix)
        ):
            # Found the PDF. The document_id is everything
            # before _{filename}
            doc_id = path.name[: -len(pdf_suffix)]
            csv_name = f"{doc_id}_extracted_table.csv"
            csv_path = upload_dir / csv_name

            if csv_path.exists():
                return csv_path

    return None