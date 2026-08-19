"""Agent module — real agent with autonomous tool selection.

Phase 10: Optimized for CPU performance.

The LLM is a REAL agent that decides which tools to call.
Tools stripped to just 2 (filter_rows + aggregate_data)
for tabular data. Non-tabular documents get NO tools —
the LLM answers directly from forced-retrieval context.

Removed tools (Phase 10):
  - calculator: LLM can do basic arithmetic itself
  - search_more: forced retrieval + query enrichment cover it
  - get_page_content: forced retrieval grabs relevant chunks

Architecture:
  1. FORCED RETRIEVAL — search always happens first, giving
     the LLM document context.
  2. TOOL SCHEMAS — only offered for tabular data (CSV/PDF
     with extracted tables). Non-tabular = no tools = faster.
  3. MULTI-STEP LOOP — the LLM can call a tool, see the
     result, then call another tool or answer. Capped at
     2 rounds to limit CPU time.
  4. TOOL EXECUTION — Python functions execute on real data
     (CSV files) with 100% accuracy.

How to run:
  cd E:\\ai-document-agent
  uv run uvicorn ai_document_agent.main:app --reload
"""

import logging
import re
from collections.abc import Callable, Generator
from pathlib import Path
from time import perf_counter

from ollama import chat
from ollama._types import Message

from ai_document_agent.data_analyzer import (
    build_analysis_context,
    find_extracted_table_csv,
    find_upload_path,
    is_tabular_file,
    load_csv,
)
from ai_document_agent.pdf_processor import (
    search_documents,
)


logger = logging.getLogger(__name__)


# Phase 9: Keeping qwen3:8b — the 4b model was too weak
# to parse messy extracted tables. Performance gains come
# from context reduction and /no_think instead.
MODEL = "qwen3:8b"

# Ryzen 5 5500U has 6 cores / 12 threads. Tell Ollama
# to use all physical cores for maximum throughput.
#
# Performance tuning:
#   num_ctx=2048 — limits the context window size.
#     Default is 4096+, which allocates more memory and
#     slows prefill. Our prompts fit in ~2048 tokens for
#     most queries. This alone can cut first-token time
#     by 30-40% on CPU.
#   keep_alive="30m" — keeps the model loaded in RAM for
#     30 minutes after the last request (default is 5m).
#     Avoids cold-start delay (~10-20s) when the user
#     sends questions with gaps between them.
LLM_OPTIONS = {
    "num_thread": 6,
    "num_ctx": 2048,
    "keep_alive": "30m",
}


# ---------------------------------------------------------
# Upload directory — needed to find CSV files for analysis
# ---------------------------------------------------------

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"


# ---------------------------------------------------------
# How many recent messages to send to the LLM
# ---------------------------------------------------------
#
# The full conversation stays in the session store (up to
# 50 messages), but sending all of them to the LLM wastes
# context and slows down generation on CPU.
#
# 4 messages = 2 user/assistant pairs. Enough for
# immediate follow-ups like:
#   "What was the total?"  →  "₹54,999"
#   "And for January?"     →  (needs context from above)
#
# Reduced from 6 → 4 to cut prompt size. Each pair
# we remove saves ~200-500 tokens of prefill time.
# The query enrichment (_enrich_query) handles most
# follow-up context by prepending the previous question
# to the search query, so 2 pairs is sufficient.
#
# The system prompt (with document context) is always
# included on top of these 4 messages.

LLM_CONTEXT_MESSAGES = 4


# ---------------------------------------------------------
# System prompt
# ---------------------------------------------------------
#
# The LLM is now a REAL AGENT. It gets:
#   1. Document context (from forced retrieval)
#   2. Tool schemas (always included)
#   3. The decision: answer directly OR call a tool
#
# Forced retrieval is KEPT — search always happens first.
# But now the LLM also has tools it can choose to call
# for deeper analysis on the retrieved data.

# ---------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------
#
# Key instructions and WHY:
#
# /no_think — disables Qwen3's "thinking" mode, which
#   adds internal reasoning tokens before each answer.
#   On CPU this wastes 30-60s of generation time.
#
# "For counting/filtering, use filter_rows" — without
#   this, the LLM tries to count rows visually from the
#   raw data table in its context and gets the number
#   wrong (e.g., saying "12 students" when it's really 7).
#   filter_rows runs Python, which counts 100% accurately.
#
# "DOCUMENT CONTEXT contains metadata" — tells the LLM
#   to look at the metadata section for info like school
#   name, class teacher, grading policy, etc. Without this
#   hint, the LLM only looks at the CSV data rows and
#   says "not found" for metadata questions.

# ---------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------
#
# Key design decisions:
#
# 1. /no_think — disables Qwen3's internal reasoning
#    (saves ~20s on CPU by skipping the <think> block)
#
# 2. "Answer from DOCUMENT TEXT first" — the text chunks
#    contain header/footer info (school name, teacher,
#    grading policy). We want the model to check these
#    BEFORE jumping to tool calls. Without this, the 8B
#    model sees the tools and immediately calls filter_rows
#    for every question — even ones that are already
#    answered in the context (like "who is the teacher?").
#
# 3. "Use filter_rows ONLY for filtering/counting" — we
#    still need the tool for questions like "how many
#    students scored above 80%" where the model would
#    miscount from raw data. But we don't want it used
#    for simple lookups.
#
# 4. "1-3 sentences" — keeps answers concise and reduces
#    generation time on CPU.

SYSTEM_PROMPT = (
    "/no_think\n"
    "Document analysis agent. Answer in 1-3 sentences.\n"
    "Use ONLY document context and tool results.\n"
    "Cite source and page. Use exact numbers/names.\n"
    "IMPORTANT: First check DOCUMENT TEXT section for the "
    "answer (it has school name, teacher name, class info, "
    "grading policy, total students). Answer directly from "
    "it if possible — do NOT call tools unnecessarily.\n"
    "Use filter_rows tool ONLY for filtering or counting "
    "questions (e.g. 'how many students scored above 80%'). "
    "Do NOT use it for simple lookups like totals or names.\n"
    "Prefer DATA ANALYSIS values (they are accurate).\n"
    "Do NOT explain your reasoning. Just give the answer.\n"
)


# ---------------------------------------------------------
# Maximum tool-calling rounds
# ---------------------------------------------------------
#
# Each round = one LLM call (~50-60s on CPU).
# With only 2 tools (filter_rows + aggregate_data),
# the LLM rarely needs more than 1-2 rounds. 3 rounds
# means: if the model calls a tool when it shouldn't
# have, it still has rounds left to give a final answer.
#
# Previously 2 rounds — but the model would sometimes
# waste both rounds on unnecessary tool calls (e.g.
# filter_rows + aggregate_data) and never actually
# answer the question. 3 rounds = safety margin.

MAX_TOOL_ROUNDS = 3


# ---------------------------------------------------------
# Tool schemas — tabular data tools only
# ---------------------------------------------------------
#
# Phase 10: Only 2 tools remain. These are offered ONLY
# when the document has tabular data (CSV or PDF with
# extracted table). The schemas are ~200 tokens total.
#
# For non-tabular documents, no tools are offered — the
# LLM answers directly from forced-retrieval context.
# This is faster because Ollama skips tool-calling mode.

FILTER_ROWS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "filter_rows",
        "description": (
            "Filter CSV/table rows by column value. "
            "Use '>90','<50' for numeric comparison."
        ),
        "parameters": {
            "type": "object",
            "required": ["column", "value"],
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Column name (exact).",
                },
                "value": {
                    "type": "string",
                    "description": (
                        "Value or comparison: '>90', "
                        "'Hard', '<=30'."
                    ),
                },
            },
        },
    },
}


AGGREGATE_DATA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "aggregate_data",
        "description": (
            "Stats on a CSV/table column, optionally "
            "grouped by another column."
        ),
        "parameters": {
            "type": "object",
            "required": ["column", "operation"],
            "properties": {
                "column": {"type": "string"},
                "operation": {
                    "type": "string",
                    "enum": [
                        "count",
                        "sum",
                        "average",
                        "min",
                        "max",
                    ],
                },
                "group_by": {
                    "type": "string",
                    "description": "Column to group by.",
                },
            },
        },
    },
}


# ---------------------------------------------------------
# Tool sets by document type
# ---------------------------------------------------------
#
# Phase 10: Stripped down to just 2 tools.
#
# Removed tools and WHY:
#   calculator — Qwen3 8B can do basic arithmetic itself.
#     A dedicated tool for 2+2 wastes ~100 schema tokens.
#   search_more — forced retrieval already searches before
#     every question. Calling search_more triggers another
#     full LLM round (~80-160s on CPU). The query enrichment
#     (_enrich_query) handles follow-ups already.
#   get_page_content — forced retrieval grabs relevant
#     chunks. For tabular data, analysis context has the
#     data. Not worth the schema tokens.
#
# Remaining tools (CSV/tabular only):
#   filter_rows — Python filtering is 100% accurate.
#     The LLM can't reliably scan 46 rows of raw text.
#   aggregate_data — Python math (count, sum, avg) is
#     100% accurate. No hallucination risk.
#
# For non-tabular documents (plain PDFs, text), the LLM
# answers directly from the forced-retrieval context
# with NO tools. This is faster because Ollama skips
# tool-calling mode entirely when tools=[] is empty.

# Tools for tabular data (CSV files / PDFs with tables)
CSV_TOOLS = [
    FILTER_ROWS_SCHEMA,
    AGGREGATE_DATA_SCHEMA,
]

# Non-tabular documents get NO tools — the LLM answers
# directly from forced retrieval context. Faster because
# Ollama doesn't enter tool-calling mode.
NO_TOOLS: list[dict] = []


# ---------------------------------------------------------
# Conversation-aware query enrichment (Phase 4.2)
# ---------------------------------------------------------
#
# Problem: follow-up questions like "what about Hindi?"
# or "tell me more about that" have almost no searchable
# keywords. BM25 and semantic search return irrelevant
# chunks or nothing.
#
# Fix: detect follow-up questions using lightweight
# heuristics and enrich the search query by prepending
# the previous user question. Zero latency cost — no
# extra LLM call.
#
# Example:
#   Q1: "list the subject wise topper name"
#   Q2: "what about Hindi?"
#   Enriched: "list the subject wise topper name — Hindi"
#   → BM25 matches "topper" + "Hindi"

# Words that signal a follow-up question
_REFERENTIAL_WORDS = {
    "that", "it", "this", "those", "these", "them",
    "the same", "above", "previous", "more",
}

_CONNECTING_STARTS = (
    "and ", "but ", "also ", "what about ",
    "how about ", "or ",
)

_STANDALONE_STARTS = (
    "what is", "what are", "who is", "who are",
    "list", "show", "how many", "how much",
    "tell me about", "explain", "describe",
    "define", "calculate", "compute",
)


def _enrich_query(
    latest_question: str,
    messages: list[dict],
) -> str:
    """Enrich a follow-up question with prior context.

    If the question looks like a follow-up (contains
    referential words, starts with connecting words, or
    is very short), prepend the previous user question
    to give BM25 and semantic search more keywords.

    If the question looks standalone, return it unchanged.

    Args:
        latest_question: The user's current question.
        messages: Full conversation history.

    Returns:
        The original or enriched search query.
    """

    if not latest_question or not messages:
        return latest_question

    q_lower = latest_question.lower().strip()
    word_count = len(q_lower.split())
    is_follow_up = False

    # Check 1: contains referential words
    for ref in _REFERENTIAL_WORDS:
        if ref in q_lower:
            is_follow_up = True
            break

    # Check 2: starts with connecting words
    if not is_follow_up:
        for prefix in _CONNECTING_STARTS:
            if q_lower.startswith(prefix):
                is_follow_up = True
                break

    # Check 3: very short and not a standalone question
    if not is_follow_up and word_count < 6:
        looks_standalone = any(
            q_lower.startswith(s) for s in _STANDALONE_STARTS
        )
        if not looks_standalone:
            is_follow_up = True

    if not is_follow_up:
        return latest_question

    # Find the previous user question (not the current one)
    prev_question = ""
    found_current = False

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        if not found_current:
            found_current = True
            continue
        prev_question = msg.get("content", "")
        break

    if not prev_question:
        return latest_question

    enriched = f"{prev_question} — {latest_question}"

    logger.info(
        "Enriched query: '%s' → '%s'",
        latest_question,
        enriched,
    )

    return enriched


# ---------------------------------------------------------
# Source filter resolver
# ---------------------------------------------------------
#
# WHY THIS EXISTS:
#
#   When the user asks a question WITHOUT selecting a
#   specific document, the frontend sends source_filter
#   as "all" (meaning "search all documents").
#
#   But all our downstream functions need a SPECIFIC
#   filename to work:
#     - _find_csv_for_source("all") → None (no CSV found)
#     - _get_tools_for_source("all") → [] (no tools)
#     - build_context_prompt() → no analysis injected
#
#   This means filtering questions like "list students
#   who failed" get NO tools and NO analysis context.
#   The model tries to answer from text chunks alone
#   and gets wrong results.
#
# THE FIX:
#
#   If only ONE document is uploaded, we auto-resolve
#   "all" to that document's filename. This way all
#   downstream functions see "report.pdf" instead of
#   "all" and work correctly — tools are provided,
#   analysis context is injected, everything works.
#
#   If multiple documents exist, we keep "all" as-is
#   (we can't guess which document the user means).
#
# HOW IT FITS IN THE PIPELINE:
#
#   1. User asks "list students who failed"
#   2. source_filter = "all" (no document selected)
#   3. _resolve_source_filter("all") → "report.pdf"
#      (auto-detected: only one document uploaded)
#   4. _get_tools_for_source("report.pdf", question)
#      → CSV_TOOLS (because "failed" is a tool keyword)
#   5. build_context_prompt() finds the extracted CSV
#      → injects analysis context with all 15 rows
#   6. LLM calls filter_rows → Python filters accurately
#   7. Correct answer!

def _resolve_source_filter(
    source_filter: str | None,
) -> str | None:
    """Resolve 'all'/None to a specific filename.

    When the user hasn't selected a specific document
    (source_filter is "all" or None), but there's only
    one document uploaded, we auto-detect its filename.

    This fixes the bug where questions like "list students
    who failed" get no tools and no analysis context when
    no specific document is selected.

    Args:
        source_filter: The raw filter from the frontend.
            Could be "all", None, or a specific filename.

    Returns:
        A specific filename if auto-detected, or the
        original value if we can't resolve it.
    """

    # Already a specific filename — use as-is.
    # Examples: "report.pdf", "data.csv"
    if (
        source_filter
        and source_filter.lower() != "all"
    ):
        return source_filter

    # No uploads directory — nothing to resolve
    if not UPLOAD_DIR.exists():
        return source_filter

    # -------------------------------------------------
    # Scan the uploads directory for original filenames.
    #
    # Upload files are saved as:
    #   {document_id}_{original_filename}
    #
    # For example:
    #   a1b2c3_report.pdf        ← uploaded PDF
    #   a1b2c3_extracted_table.csv  ← derived file
    #
    # We want to find the ORIGINAL uploads only, not
    # the derived files (extracted CSVs). We skip files
    # ending with "_extracted_table.csv" since those are
    # created by our PDF processor, not uploaded by user.
    # -------------------------------------------------

    original_names: list[str] = []

    for path in UPLOAD_DIR.iterdir():
        if not path.is_file():
            continue

        name = path.name

        # Skip derived files (extracted table CSVs)
        if name.endswith("_extracted_table.csv"):
            continue

        # Extract original filename after the first "_"
        # {document_id}_{original_filename}
        underscore_idx = name.find("_")
        if underscore_idx > 0:
            original = name[underscore_idx + 1:]
            if original not in original_names:
                original_names.append(original)

    # Auto-resolve only when exactly ONE document exists.
    # With multiple documents, we can't guess which one
    # the user is asking about — keep "all" as-is.
    if len(original_names) == 1:
        resolved = original_names[0]
        logger.info(
            "Auto-resolved source_filter: '%s' → '%s' "
            "(only one document uploaded)",
            source_filter,
            resolved,
        )
        return resolved

    if len(original_names) > 1:
        logger.info(
            "Cannot auto-resolve source_filter '%s': "
            "%d documents found (%s)",
            source_filter,
            len(original_names),
            ", ".join(original_names),
        )

    return source_filter


def _question_needs_tools(question: str) -> bool:
    """Detect if a question needs filtering/counting tools.

    Why this exists:
      The 8B model can't resist calling tools when they're
      available. If we give it filter_rows for a question
      like "who is the class teacher?", it will call
      filter_rows({'column': 'Class Teacher', ...}) instead
      of just reading the answer from the context.

      By only providing tools when the question actually
      needs them (counting, filtering, comparing), we
      prevent the model from wasting 2-3 tool rounds on
      simple lookup questions.

    Keyword detection is intentionally broad — it's better
    to give tools when not needed (worst case: model calls
    a tool but still answers) than to withhold tools when
    needed (model can't count and gives wrong numbers).
    """

    # Normalize to lowercase for matching
    q = question.lower()

    # Keywords that signal a filtering/counting question
    tool_keywords = [
        # Counting
        "how many", "count", "total number",
        "number of students",
        # Filtering
        "above", "below", "more than", "less than",
        "greater than", "scored", "scoring",
        "higher than", "lower than",
        "at least", "at most",
        # Comparison / aggregation
        "average", "mean", "sum", "highest", "lowest",
        "top ", "bottom ", "best", "worst",
        "rank", "compare", "group by",
        "percentage", "percent", "pass rate",
        "failed", "distinction",
        # Specific patterns
        ">", "<", ">=", "<=",
    ]

    return any(kw in q for kw in tool_keywords)


def _get_tools_for_source(
    source_filter: str | None,
    question: str = "",
) -> list[dict]:
    """Pick the right tool set for the document + question.

    Tool selection now depends on BOTH the document type
    AND the question being asked:

    - CSV files always get tools (they have no text view,
      so the model always needs tools to work with data)
    - PDFs with tables get tools ONLY if the question
      needs filtering/counting (detected by keywords).
      Simple lookup questions (school name, teacher name)
      get NO tools so the model answers from context.
    - Everything else gets NO tools.

    Why question-based filtering?
      The 8B model treats tools as compulsory — if tools
      are available, it calls them for EVERY question,
      even when the answer is right there in the text.
      This wastes 2-3 LLM rounds (~50s each on CPU) on
      unnecessary tool calls and often produces wrong
      answers because the model answers from tool results
      instead of the context.
    """

    # Pure CSV files always get tools — they have no
    # text view, so the model needs tools to work with
    # the data for ANY question.
    if source_filter and is_tabular_file(source_filter):
        return CSV_TOOLS

    # PDFs with extracted tables: only give tools when
    # the question actually needs filtering/counting.
    #
    # For simple lookups ("who is the teacher?", "school
    # name?", "total students?"), NO tools are given.
    # The model answers directly from the context — much
    # faster (1 LLM call instead of 3-4) and more
    # accurate (no tool-call distractions).
    if source_filter and source_filter.lower().endswith(
        ".pdf"
    ):
        csv_path = find_extracted_table_csv(
            source_filter, UPLOAD_DIR,
        )
        if csv_path:
            try:
                _, rows = load_csv(str(csv_path))
                if len(rows) >= 10:
                    # Check if question needs tools
                    if _question_needs_tools(question):
                        logger.info(
                            "PDF has extracted table CSV: "
                            "%s (%d rows — question needs "
                            "tools, providing CSV tools)",
                            csv_path.name,
                            len(rows),
                        )
                        return CSV_TOOLS
                    else:
                        logger.info(
                            "PDF has extracted table CSV: "
                            "%s (%d rows — simple lookup, "
                            "no tools needed)",
                            csv_path.name,
                            len(rows),
                        )
                        return []
                else:
                    logger.info(
                        "PDF table CSV has only %d rows "
                        "— too few, no tools needed",
                        len(rows),
                    )
            except Exception:
                pass

    return NO_TOOLS


# ---------------------------------------------------------
# Forced retrieval — search before every question
# ---------------------------------------------------------

def build_context_prompt(
    question: str,
    source_filter: str | None = None,
) -> str:
    """Search documents and build context for the LLM.

    Args:
        question: The user's question.
        source_filter: Optional filename to search within
                       only one document (e.g. "report.pdf").
                       If None, searches all documents.

    Returns:
        A context string to append to the system prompt.
        Empty string if no documents are found.
    """

    # For tabular data, chunk retrieval returns random text
    # fragments of a table — useless noise that bloats the
    # prompt. Skip retrieval when:
    #   1. Source is a CSV file, OR
    #   2. Source is a PDF with an extracted CSV (≥10 rows)
    # In both cases, the analysis context below has the
    # real data the LLM needs.
    is_csv = (
        source_filter
        and source_filter.lower().endswith(".csv")
    )

    # Check if PDF has a usable extracted table
    has_table_csv = False
    if (
        not is_csv
        and source_filter
        and source_filter.lower().endswith(".pdf")
    ):
        table_csv = _find_csv_for_source(source_filter)
        if table_csv:
            try:
                _, check_rows = load_csv(str(table_csv))
                if len(check_rows) >= 10:
                    has_table_csv = True
            except Exception:
                pass

    # -------------------------------------------------
    # When to skip chunk retrieval:
    #
    #   ONLY for pure CSV files. CSV files have no "text
    #   view" — they're just rows and columns. The analysis
    #   context (stats + raw data) is all we need.
    #
    #   PDFs with tables get BOTH views:
    #     1. Text chunks — from _extract_text_with_headings()
    #        These contain the HEADER (school name, teacher,
    #        exam title) and FOOTER (grading policy, summary)
    #        as clean, readable text.
    #     2. Analysis context — from build_analysis_context()
    #        This has the TABLE data (student rows, stats).
    #
    #   Previously we skipped chunks for PDFs too, which
    #   meant the LLM never saw header/footer text and
    #   couldn't answer "who is the class teacher?" etc.
    # -------------------------------------------------
    skip_chunks = is_csv

    if skip_chunks:
        logger.info(
            "Pure CSV source — skipping chunk retrieval, "
            "using analysis context only",
        )
        parts = ["\n\nDOCUMENT CONTEXT:"]
        results = []
    else:
        try:
            results = search_documents(
                query=question,
                n_results=3,
                source_filter=source_filter,
            )
        except Exception as exc:
            logger.warning(
                "Document search failed: %s", exc
            )
            return ""

        if not results:
            return (
                "\n\nDOCUMENT CONTEXT:\n"
                "No relevant documents found. The user "
                "may not have uploaded any PDFs yet, or "
                "no uploaded documents match this question."
            )

        # -------------------------------------------------
        # Build text chunk excerpts.
        #
        # For PDFs with tables, we DON'T add chunks here
        # yet — we defer them to AFTER the analysis block.
        #
        # Why? Small LLMs (8B) have strong "recency bias":
        # they pay most attention to what's closest to the
        # end of the context. If we put the text chunks
        # (containing school name, teacher name) first and
        # then dump 2777 chars of table data, the model
        # "forgets" the header/footer by the time it
        # answers.
        #
        # By putting text chunks AFTER the analysis, the
        # document header/footer info (school, teacher,
        # grading policy) is the LAST thing the model
        # reads before generating its answer.
        # -------------------------------------------------

        # Log chunk content for debugging
        for i, result in enumerate(results, 1):
            logger.info(
                "Chunk %d (score=%.4f, page=%s): %s",
                i,
                result.score,
                result.page,
                result.text[:120].replace("\n", " "),
            )

        if not has_table_csv:
            # Non-tabular PDF: put chunks in the normal
            # position (no analysis block will follow)
            parts = ["\n\nDOCUMENT CONTEXT:"]

            for i, result in enumerate(results, 1):
                parts.append(
                    f"\n--- Excerpt {i} "
                    f"(Score: {result.score:.4f}) ---\n"
                    f"Source: {result.source}, "
                    f"Page {result.page}\n"
                    f"Content:\n{result.text}\n"
                )

            parts.append(
                "\nUse the above excerpts to answer the "
                "user's question. Cite the source and page."
            )
        else:
            # PDF with table: start with just the header.
            # Text chunks will be appended AFTER analysis.
            parts = ["\n\nDOCUMENT CONTEXT:"]

    # -------------------------------------------------------
    # Data analysis for CSV files
    # -------------------------------------------------------
    #
    # For CSV files, chunk retrieval gives random text
    # fragments of a table. That's like reading 5 random
    # cells from a spreadsheet and guessing the total.
    #
    # Instead, we load the FULL CSV and compute real stats
    # (count, sum, average, group-by) in Python. These
    # computed results are injected alongside the chunks
    # so the LLM has accurate numbers to work with.
    #
    # The analysis module handles:
    #   - Column type detection (numeric vs categorical)
    #   - Stats for numeric columns (min/max/avg/sum)
    #   - Value counts for categorical columns
    #   - Raw data (all rows if small, sample if large)

    # Inject data analysis for tabular sources.
    # Works for:
    #   1. Direct CSV files (uploaded CSVs)
    #   2. PDFs with extracted tables (internal CSVs)
    #
    # For PDFs: only inject if the extracted CSV has
    # enough rows (≥10). Partial data (e.g. 8 of 30
    # rows) confuses the LLM and adds latency for
    # no benefit.
    analysis_csv_path = _find_csv_for_source(source_filter)

    if analysis_csv_path:

        # For PDF sources, check row count first
        skip_analysis = False

        if (
            source_filter
            and source_filter.lower().endswith(".pdf")
        ):
            try:
                _, check_rows = load_csv(
                    str(analysis_csv_path),
                )
                if len(check_rows) < 10:
                    logger.info(
                        "Skipping analysis injection: "
                        "only %d rows in extracted CSV",
                        len(check_rows),
                    )
                    skip_analysis = True
            except Exception:
                skip_analysis = True

        if skip_analysis:
            return "\n".join(parts)

        analysis = build_analysis_context(
            str(analysis_csv_path),
        )

        if analysis:

            logger.info(
                "Data analysis: %d chars of context "
                "(from %s)",
                len(analysis),
                analysis_csv_path.name,
            )

            parts.append(analysis)

    # -------------------------------------------------
    # Deferred text chunks for PDFs with tables
    # -------------------------------------------------
    #
    # For tabular PDFs, we intentionally put the text
    # chunks LAST — after the analysis block. This way
    # the document header (school name, teacher) and
    # footer (grading policy, class summary) are the
    # LAST thing the LLM reads before answering.
    #
    # Small LLMs (8B) have strong recency bias — they
    # pay most attention to content near the end. If we
    # put the analysis data (2777 chars of stats + rows)
    # after the text chunks, the LLM "forgets" the
    # header/footer info by the time it generates.
    #
    # This ordering means:
    #   1. Analysis block (table stats + sample rows)
    #   2. Text excerpts (header + footer text)
    #   → LLM answers with BOTH data AND context
    if has_table_csv and results:
        parts.append(
            "\n\nDOCUMENT TEXT (from the original PDF — "
            "contains document title, class teacher, "
            "school name, grading policy, and other "
            "information NOT in the table data above):"
        )

        for i, result in enumerate(results, 1):
            parts.append(
                f"\n--- Excerpt {i} "
                f"(Page {result.page}) ---\n"
                f"{result.text}\n"
            )

    return "\n".join(parts)


# ---------------------------------------------------------
# Context limiting helper
# ---------------------------------------------------------
#
# Takes the full message list (system + all history) and
# trims it to system prompt + last N messages. This keeps
# the LLM's input small and generation fast on CPU.

def _limit_context(
    full_messages: list[dict],
    max_messages: int = LLM_CONTEXT_MESSAGES,
) -> list[dict]:
    """Keep system prompt + only the most recent messages.

    Args:
        full_messages: [system_msg, msg1, msg2, ...].
        max_messages: How many recent messages to keep.

    Returns:
        Trimmed message list.
    """

    # full_messages[0] is always the system prompt.
    # full_messages[1:] is the conversation history.

    if len(full_messages) <= max_messages + 1:
        # Already small enough, no trimming needed.
        return full_messages

    return (
        [full_messages[0]]          # system prompt
        + full_messages[-max_messages:]  # recent msgs
    )


# ---------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------
#
# Each tool is a Python function that operates on REAL
# document data — CSV files on disk or chunks in ChromaDB.
# The LLM calls these tools; Python executes them with
# 100% accuracy; results go back to the LLM.
#
# Phase 8.2: source_filter is now passed explicitly to
# each tool function instead of using a module-level
# global. This fixes a concurrency bug where simultaneous
# requests would overwrite each other's source filter.
#
# The TOOL_REGISTRY is now a function that returns a
# registry bound to the current request's source_filter.


def _find_csv_for_source(
    source: str | None,
) -> Path | None:
    """Find the CSV file for the active document.

    For CSV files: finds the uploaded CSV directly.
    For PDF files: finds the extracted table CSV.
    Returns None if no CSV is available.
    """

    if not source:
        return None

    # Direct CSV file
    if is_tabular_file(source):
        return find_upload_path(source, UPLOAD_DIR)

    # PDF with extracted table
    if source.lower().endswith(".pdf"):
        return find_extracted_table_csv(
            source, UPLOAD_DIR,
        )

    return None


def filter_rows(
    column: str,
    value: str,
    source_filter: str | None = None,
) -> str:
    """Filter CSV rows by column value.

    Supports exact match and numeric comparisons:
      - "Hard"   → exact match
      - ">90"    → greater than 90
      - "<=50"   → less than or equal to 50

    Returns filtered rows as formatted text.
    Works on CSV files AND PDFs with extracted tables.
    """

    source = source_filter
    csv_path = _find_csv_for_source(source)

    if not csv_path:
        return ("Error: This tool only works on tabular "
                "data (CSV files or PDFs with tables). "
                "No tabular data found for this document. "
                "Answer the question using the document "
                "context provided above instead. "
                "Do NOT call this tool again.")

    try:
        headers, rows = load_csv(str(csv_path))
    except Exception as exc:
        return f"Error loading CSV: {exc}"

    if column not in headers:
        return (
            f"Error: Column '{column}' not found. "
            f"Available columns: {', '.join(headers)}"
        )

    # Parse comparison operator
    op_match = re.match(
        r'^(>=|<=|>|<|!=|=)\s*(.+)$', value.strip(),
    )

    filtered = []

    if op_match:
        # Numeric comparison
        operator = op_match.group(1)
        compare_val = op_match.group(2).strip()

        try:
            compare_num = float(
                compare_val.replace(",", ""),
            )
        except ValueError:
            return (
                f"Error: '{compare_val}' is not a number."
            )

        for row in rows:
            cell = row.get(column, "").strip()
            if not cell:
                continue
            try:
                cell_num = float(
                    cell.replace(",", ""),
                )
            except ValueError:
                continue

            if operator == ">" and cell_num > compare_num:
                filtered.append(row)
            elif operator == ">=" and cell_num >= compare_num:
                filtered.append(row)
            elif operator == "<" and cell_num < compare_num:
                filtered.append(row)
            elif operator == "<=" and cell_num <= compare_num:
                filtered.append(row)
            elif operator == "!=" and cell_num != compare_num:
                filtered.append(row)
            elif operator == "=" and cell_num == compare_num:
                filtered.append(row)
    else:
        # Exact string match (case-insensitive)
        target = value.strip().lower()

        for row in rows:
            cell = row.get(column, "").strip().lower()
            if cell == target:
                filtered.append(row)

    if not filtered:
        return (
            f"No rows found where {column} matches "
            f"'{value}'. Total rows in dataset: {len(rows)}."
        )

    # Format as table text
    header_line = " | ".join(headers)
    lines = [
        f"FILTERED RESULTS ({len(filtered)} rows "
        f"where {column} = '{value}'):",
        header_line,
    ]

    for row in filtered[:50]:  # cap at 50 rows
        lines.append(
            " | ".join(
                row.get(h, "") for h in headers
            )
        )

    if len(filtered) > 50:
        lines.append(
            f"... ({len(filtered) - 50} more rows)"
        )

    return "\n".join(lines)


def aggregate_data(
    column: str,
    operation: str,
    group_by: str | None = None,
    source_filter: str | None = None,
) -> str:
    """Compute aggregate statistics on a CSV column.

    Supports count, sum, average, min, max — optionally
    grouped by another column.
    """

    source = source_filter
    csv_path = _find_csv_for_source(source)

    if not csv_path:
        return ("Error: This tool only works on tabular "
                "data (CSV files or PDFs with tables). "
                "No tabular data found for this document. "
                "Answer the question using the document "
                "context provided above instead. "
                "Do NOT call this tool again.")

    try:
        headers, rows = load_csv(str(csv_path))
    except Exception as exc:
        return f"Error loading CSV: {exc}"

    if column not in headers:
        return (
            f"Error: Column '{column}' not found. "
            f"Available columns: {', '.join(headers)}"
        )

    if group_by and group_by not in headers:
        return (
            f"Error: Group-by column '{group_by}' not "
            f"found. Available: {', '.join(headers)}"
        )

    def _compute(subset: list[dict]) -> str:
        """Compute one aggregation on a subset."""

        values = [
            row.get(column, "").strip()
            for row in subset
        ]
        non_empty = [v for v in values if v]

        if operation == "count":
            return str(len(non_empty))

        # Numeric operations
        nums = []
        for v in non_empty:
            try:
                nums.append(
                    float(v.replace(",", "")),
                )
            except ValueError:
                pass

        if not nums:
            return "N/A (no numeric values)"

        if operation == "sum":
            return f"{sum(nums):g}"
        if operation == "average":
            return f"{sum(nums) / len(nums):.2f}"
        if operation == "min":
            return f"{min(nums):g}"
        if operation == "max":
            return f"{max(nums):g}"

        return "Unknown operation"

    if group_by:
        # Group rows by the group_by column
        groups: dict[str, list[dict]] = {}

        for row in rows:
            key = row.get(group_by, "").strip() or "(empty)"
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        # Compute per group
        lines = [
            f"AGGREGATION: {operation}({column}) "
            f"grouped by {group_by}:",
        ]

        for group_name in sorted(groups.keys()):
            result = _compute(groups[group_name])
            lines.append(f"  {group_name}: {result}")

        return "\n".join(lines)

    else:
        # No grouping — compute on all rows
        result = _compute(rows)

        return (
            f"AGGREGATION: {operation}({column}) = "
            f"{result}  (total rows: {len(rows)})"
        )


# ---------------------------------------------------------
# Tool registry — maps tool names to Python functions
# ---------------------------------------------------------
#
# Adding a new tool:
#   1. Define the JSON schema above
#   2. Write the Python function above
#   3. Add one line here
#   4. Add the schema to CSV_TOOLS list
#
# The LLM decides WHEN to call each tool.
# Python decides HOW to execute it.

def _build_tool_registry(
    source_filter: str | None,
) -> dict[str, Callable]:
    """Build a tool registry bound to this request's
    source_filter.

    Phase 8.2: each request gets its own registry with
    source_filter captured in closures. This replaces
    the old module-level _active_source_filter global,
    fixing a concurrency bug where simultaneous requests
    would overwrite each other's filter.

    Phase 10: stripped to just filter_rows + aggregate_data.
    Calculator, search_more, get_page_content removed —
    the LLM handles those directly (see tool set comments).
    """

    return {
        "filter_rows": lambda args: filter_rows(
            column=args["column"],
            value=args["value"],
            source_filter=source_filter,
        ),
        "aggregate_data": lambda args: aggregate_data(
            column=args["column"],
            operation=args["operation"],
            group_by=args.get("group_by"),
            source_filter=source_filter,
        ),
    }


# ---------------------------------------------------------
# ask_agent — non-streaming (used by /chat endpoint)
# ---------------------------------------------------------

def ask_agent(
    messages: list[dict],
    request_id: str = "",
    source_filter: str | None = None,
) -> str:
    """Run the agent with full conversation history.

    Phase 7 flow:
      1. Forced retrieval (always)
      2. LLM sees context + ALL tool schemas
      3. LLM decides: answer directly or call tool(s)
      4. Multi-step loop (up to MAX_TOOL_ROUNDS)

    Args:
        messages: The complete conversation so far.
        request_id: Unique ID for tracing.
        source_filter: Optional filename to scope search.

    Returns:
        The assistant's final text answer.
    """

    # -------------------------------------------------
    # STEP 0: Resolve source_filter
    # -------------------------------------------------
    #
    # If source_filter is "all" (no document selected)
    # and only one document is uploaded, auto-resolve
    # to that document's filename. This ensures tools
    # and analysis context work correctly.
    source_filter = _resolve_source_filter(source_filter)

    # Build per-request tool registry (Phase 8.2)
    tool_registry = _build_tool_registry(source_filter)

    start_time = perf_counter()

    logger.info(
        "[%s] Agent thinking (%d messages in history)",
        request_id,
        len(messages),
    )

    # -------------------------------------------------------
    # STEP 1: Forced retrieval
    # -------------------------------------------------------

    latest_question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_question = msg.get("content", "")
            break

    context = ""
    if latest_question:
        # Enrich follow-up questions with prior context
        search_query = _enrich_query(
            latest_question, messages,
        )

        context = build_context_prompt(
            search_query,
            source_filter=source_filter,
        )

        logger.info(
            "[%s] Forced retrieval: %d chars of context",
            request_id,
            len(context),
        )

    # -------------------------------------------------------
    # STEP 2: Build system prompt with context + tools
    # -------------------------------------------------------

    system_content = SYSTEM_PROMPT + context

    if source_filter:
        system_content += (
            f"\nYou are answering questions about: "
            f"{source_filter}\n"
        )

    # Pick tools based on document type AND question
    active_tools = _get_tools_for_source(
        source_filter, latest_question,
    )

    full_messages = [
        {
            "role": "system",
            "content": system_content,
        },
        *messages,
    ]

    full_messages = _limit_context(full_messages)

    # -------------------------------------------------------
    # STEP 3: Multi-step tool loop
    # -------------------------------------------------------
    #
    # The LLM can call tools in sequence. After each tool
    # result, the LLM decides: call another tool or answer.
    # Capped at MAX_TOOL_ROUNDS to prevent runaway loops.

    working_messages = list(full_messages)
    tools_used = []

    for round_num in range(MAX_TOOL_ROUNDS):

        llm_start = perf_counter()

        response = chat(
            model=MODEL,
            messages=working_messages,
            tools=active_tools,
            think=False,
            options=LLM_OPTIONS,
        )

        llm_time = perf_counter() - llm_start

        logger.info(
            "[%s] LLM round %d: %.2fs",
            request_id,
            round_num + 1,
            llm_time,
        )

        # No tool calls → LLM gave a direct answer
        if not response.message.tool_calls:
            total_time = perf_counter() - start_time

            if tools_used:
                logger.info(
                    "[%s] Final answer after tools "
                    "[%s]: %.2fs",
                    request_id,
                    ", ".join(tools_used),
                    total_time,
                )
            else:
                logger.info(
                    "[%s] Direct answer (no tools): "
                    "%.2fs",
                    request_id,
                    total_time,
                )

            return response.message.content

        # Tool calls — execute each one
        working_messages.append(response.message)

        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            logger.info(
                "[%s] Tool call: %s(%s)",
                request_id,
                tool_name,
                arguments,
            )

            tool_fn = tool_registry.get(tool_name)

            if tool_fn is None:
                logger.error(
                    "[%s] Unknown tool: %s",
                    request_id,
                    tool_name,
                )
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": (
                            f"Error: Unknown tool "
                            f"'{tool_name}'."
                        ),
                    }
                )
                continue

            tool_start = perf_counter()

            try:
                result = tool_fn(arguments)
            except Exception as exc:
                result = f"Tool error: {exc}"

            tool_time = perf_counter() - tool_start

            logger.info(
                "[%s] Tool '%s' result: %.4fs",
                request_id,
                tool_name,
                tool_time,
            )

            tools_used.append(tool_name)

            working_messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(result),
                }
            )

    # If we exhausted all rounds, do one final LLM call
    # without tools to force a text answer.

    logger.info(
        "[%s] Max tool rounds reached, forcing answer",
        request_id,
    )

    final_response = chat(
        model=MODEL,
        messages=working_messages,
        think=False,
        options=LLM_OPTIONS,
    )

    total_time = perf_counter() - start_time

    logger.info(
        "[%s] Forced final answer: %.2fs",
        request_id,
        total_time,
    )

    return final_response.message.content


# ---------------------------------------------------------
# stream_agent — streaming with multi-step tool loop
# ---------------------------------------------------------
#
# Phase 7: REAL AGENT with streaming.
#
# The LLM sees ALL tool schemas on every call and
# autonomously decides which (if any) to use. It can
# call multiple tools in sequence — each result feeds
# back into the next LLM call.
#
# Streaming works in two modes:
#   1. DIRECT ANSWER: tokens stream to the browser
#      word by word (most common, fastest).
#   2. TOOL CALL(S): tools execute silently, then the
#      final answer streams. Status events keep the
#      user informed during tool execution.
#
# Multi-step example:
#   Round 1: LLM calls filter_rows → sees filtered data
#   Round 2: LLM calls aggregate_data → sees stats
#   Round 3: LLM writes final answer using both results

def stream_agent(
    messages: list[dict],
    request_id: str = "",
    source_filter: str | None = None,
) -> Generator[dict, None, None]:
    """Stream agent events with forced retrieval + tools.

    Phase 7 flow:
      1. Search documents (forced retrieval)
      2. LLM sees context + ALL tool schemas
      3. Multi-step tool loop (stream final answer)

    Args:
        messages: The complete conversation so far.
        request_id: Unique ID for tracing.
        source_filter: Optional filename to scope search.

    Yields:
        Event dicts for the browser to display.
    """

    # -------------------------------------------------
    # STEP 0: Resolve source_filter
    # -------------------------------------------------
    #
    # Same as in ask_agent(): if source_filter is "all"
    # and only one document exists, auto-resolve to the
    # actual filename so tools and analysis work.
    source_filter = _resolve_source_filter(source_filter)

    # Build per-request tool registry (Phase 8.2)
    tool_registry = _build_tool_registry(source_filter)

    start_time = perf_counter()

    logger.info(
        "[%s] Stream agent started "
        "(%d messages in history)",
        request_id,
        len(messages),
    )

    # -------------------------------------------------------
    # STEP 1: Forced retrieval
    # -------------------------------------------------------

    yield {
        "type": "status",
        "status": "thinking",
        "message": "Searching your documents...",
        "elapsed_seconds": 0,
        "request_id": request_id,
    }

    latest_question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_question = msg.get("content", "")
            break

    context = ""
    if latest_question:
        # Enrich follow-up questions with prior context
        search_query = _enrich_query(
            latest_question, messages,
        )

        search_start = perf_counter()
        context = build_context_prompt(
            search_query,
            source_filter=source_filter,
        )
        search_time = perf_counter() - search_start

        logger.info(
            "[%s] Forced retrieval: %d chars (%.2fs)",
            request_id,
            len(context),
            search_time,
        )

        yield {
            "type": "status",
            "status": "search_complete",
            "message": "Found relevant documents. "
                       "Analyzing...",
            "elapsed_seconds": round(
                perf_counter() - start_time, 2,
            ),
            "request_id": request_id,
        }

    # -------------------------------------------------------
    # STEP 2: Build system prompt
    # -------------------------------------------------------

    system_content = SYSTEM_PROMPT + context

    if source_filter:
        system_content += (
            f"\nYou are answering questions about: "
            f"{source_filter}\n"
        )

    # Pick tools based on document type AND question.
    # Simple lookup questions ("who is the teacher?")
    # get NO tools — the model answers from context.
    # Filtering/counting questions get tools.
    active_tools = _get_tools_for_source(
        source_filter, latest_question,
    )

    full_messages = [
        {
            "role": "system",
            "content": system_content,
        },
        *messages,
    ]

    full_messages = _limit_context(full_messages)

    logger.info(
        "[%s] Sending %d messages to LLM "
        "(%d tools available)",
        request_id,
        len(full_messages),
        len(active_tools),
    )

    # -------------------------------------------------------
    # STEP 3: Multi-step tool loop with streaming
    # -------------------------------------------------------
    #
    # Each round:
    #   a) Stream LLM response
    #   b) If content tokens → direct answer, done
    #   c) If tool_calls → execute tools, loop back
    #
    # Only the FINAL answer is streamed to the browser.
    # Tool rounds use non-streaming (faster turnaround).

    working_messages = list(full_messages)
    tools_used: list[str] = []

    for round_num in range(MAX_TOOL_ROUNDS):

        # For tool rounds (not the first), use
        # non-streaming — faster turnaround since
        # we're just getting tool decisions.
        if round_num > 0:

            logger.info(
                "[%s] Tool round %d",
                request_id,
                round_num + 1,
            )

            try:

                response = chat(
                    model=MODEL,
                    messages=working_messages,
                    tools=active_tools,
                    think=False,
                    options=LLM_OPTIONS,
                )

            except Exception as exc:

                logger.error(
                    "[%s] LLM call failed: %s",
                    request_id,
                    exc,
                )

                yield {
                    "type": "error",
                    "message": f"LLM error: {exc}",
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": round(
                        perf_counter() - start_time, 2,
                    ),
                    "request_id": request_id,
                }

                return

            if not response.message.tool_calls:
                # LLM decided to answer — but we got
                # the answer non-streamed. Emit it as
                # one big token block.

                content = response.message.content or ""

                yield {
                    "type": "status",
                    "status": "first_token",
                    "message": "Generating answer...",
                    "first_token_seconds": round(
                        perf_counter() - start_time, 2,
                    ),
                    "request_id": request_id,
                }

                yield {
                    "type": "token",
                    "content": content,
                }

                total_time = perf_counter() - start_time

                logger.info(
                    "[%s] Answer after tools [%s]: "
                    "%.2fs",
                    request_id,
                    ", ".join(tools_used),
                    total_time,
                )

                yield {
                    "type": "completed",
                    "response_time_seconds": round(
                        total_time, 2,
                    ),
                    "assistant_content": content,
                    "tools_used": tools_used,
                    "request_id": request_id,
                }

                return

            # More tool calls — execute and loop
            working_messages.append(response.message)

            for tc in response.message.tool_calls:
                yield from _execute_tool_streaming(
                    tc, working_messages,
                    tools_used, start_time,
                    request_id, tool_registry,
                )

            continue

        # -------------------------------------------------
        # First round: STREAMING
        # -------------------------------------------------

        decision_start = perf_counter()

        try:

            stream = chat(
                model=MODEL,
                messages=working_messages,
                tools=active_tools,
                think=False,
                stream=True,
                options=LLM_OPTIONS,
            )

        except Exception as exc:

            logger.error(
                "[%s] LLM call failed: %s",
                request_id,
                exc,
            )

            yield {
                "type": "error",
                "message": f"LLM error: {exc}",
                "error_type": type(exc).__name__,
                "elapsed_seconds": round(
                    perf_counter() - start_time, 2,
                ),
                "request_id": request_id,
            }

            return

        full_content_parts: list[str] = []
        tool_calls = None
        first_token = True

        for chunk in stream:

            if chunk.message.content:

                if first_token:

                    first_token = False

                    first_token_time = (
                        perf_counter() - start_time
                    )

                    logger.info(
                        "[%s] First token at %.2fs",
                        request_id,
                        first_token_time,
                    )

                    yield {
                        "type": "status",
                        "status": "first_token",
                        "message": "Generating answer...",
                        "first_token_seconds": round(
                            first_token_time, 2,
                        ),
                        "request_id": request_id,
                    }

                full_content_parts.append(
                    chunk.message.content,
                )

                yield {
                    "type": "token",
                    "content": chunk.message.content,
                }

            if chunk.message.tool_calls:
                tool_calls = chunk.message.tool_calls

        decision_time = perf_counter() - decision_start

        logger.info(
            "[%s] LLM stream done: %.2fs",
            request_id,
            decision_time,
        )

        # Direct answer — done
        if not tool_calls or full_content_parts:

            total_time = perf_counter() - start_time

            logger.info(
                "[%s] Direct answer (streamed): %.2fs",
                request_id,
                total_time,
            )

            yield {
                "type": "completed",
                "response_time_seconds": round(
                    total_time, 2,
                ),
                "assistant_content": "".join(
                    full_content_parts,
                ),
                "request_id": request_id,
            }

            return

        # Tool calls — execute and loop
        yield {
            "type": "status",
            "status": "decision_completed",
            "message": "Agent deciding which tool to use...",
            "elapsed_seconds": round(
                perf_counter() - start_time, 2,
            ),
            "decision_time_seconds": round(
                decision_time, 2,
            ),
            "request_id": request_id,
        }

        assistant_msg = Message(
            role="assistant",
            content="",
            tool_calls=tool_calls,
        )

        working_messages.append(assistant_msg)

        for tc in tool_calls:
            yield from _execute_tool_streaming(
                tc, working_messages,
                tools_used, start_time,
                request_id, tool_registry,
            )

    # -------------------------------------------------
    # Exhausted all tool rounds — force a final answer
    # -------------------------------------------------

    logger.info(
        "[%s] Max tool rounds reached, streaming "
        "final answer",
        request_id,
    )

    yield {
        "type": "status",
        "status": "generating",
        "message": "Preparing final answer...",
        "elapsed_seconds": round(
            perf_counter() - start_time, 2,
        ),
        "request_id": request_id,
    }

    try:

        final_stream = chat(
            model=MODEL,
            messages=working_messages,
            think=False,
            stream=True,
            options=LLM_OPTIONS,
        )

    except Exception as exc:

        yield {
            "type": "error",
            "message": f"LLM error: {exc}",
            "error_type": type(exc).__name__,
            "elapsed_seconds": round(
                perf_counter() - start_time, 2,
            ),
            "request_id": request_id,
        }

        return

    first_final = True
    final_parts: list[str] = []

    for chunk in final_stream:

        if not chunk.message.content:
            continue

        if first_final:
            first_final = False

            yield {
                "type": "status",
                "status": "first_token",
                "message": "Generating answer...",
                "first_token_seconds": round(
                    perf_counter() - start_time, 2,
                ),
                "request_id": request_id,
            }

        final_parts.append(chunk.message.content)

        yield {
            "type": "token",
            "content": chunk.message.content,
        }

    total_time = perf_counter() - start_time

    logger.info(
        "[%s] Completed (max rounds, tools=[%s]) "
        "in %.2fs",
        request_id,
        ", ".join(tools_used),
        total_time,
    )

    yield {
        "type": "completed",
        "response_time_seconds": round(
            total_time, 2,
        ),
        "assistant_content": "".join(final_parts),
        "tools_used": tools_used,
        "request_id": request_id,
    }


# ---------------------------------------------------------
# Helper: execute a single tool call during streaming
# ---------------------------------------------------------

def _execute_tool_streaming(
    tool_call,
    working_messages: list,
    tools_used: list[str],
    start_time: float,
    request_id: str,
    tool_registry: dict[str, Callable] | None = None,
) -> Generator[dict, None, None]:
    """Execute one tool call and yield status events.

    Appends the tool result to working_messages so the
    next LLM call sees it.
    """

    if tool_registry is None:
        tool_registry = {}

    tool_name = tool_call.function.name
    arguments = tool_call.function.arguments

    logger.info(
        "[%s] Tool call: %s(%s)",
        request_id,
        tool_name,
        arguments,
    )

    yield {
        "type": "status",
        "status": "tool_call",
        "tool": tool_name,
        "message": f"Using {tool_name}...",
        "elapsed_seconds": round(
            perf_counter() - start_time, 2,
        ),
        "request_id": request_id,
    }

    tool_start = perf_counter()

    tool_fn = tool_registry.get(tool_name)

    if tool_fn is None:

        logger.error(
            "[%s] Unknown tool: %s",
            request_id,
            tool_name,
        )

        working_messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": (
                    f"Error: Unknown tool '{tool_name}'."
                ),
            }
        )

        return

    try:
        result = tool_fn(arguments)
    except Exception as exc:

        logger.error(
            "[%s] Tool '%s' failed: %s",
            request_id,
            tool_name,
            exc,
        )

        result = f"Tool error: {exc}"

    tool_time = perf_counter() - tool_start

    logger.info(
        "[%s] Tool '%s' done: %.4fs",
        request_id,
        tool_name,
        tool_time,
    )

    tools_used.append(tool_name)

    yield {
        "type": "tool_result",
        "tool": tool_name,
        "result": str(result)[:200],  # preview only
        "tool_time_seconds": round(tool_time, 4),
        "elapsed_seconds": round(
            perf_counter() - start_time, 2,
        ),
        "request_id": request_id,
    }

    working_messages.append(
        {
            "role": "tool",
            "tool_name": tool_name,
            "content": str(result),
        }
    )