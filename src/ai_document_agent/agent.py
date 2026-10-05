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

# ---------------------------------------------------------
# LLM Provider — swappable backend (Ollama / OpenRouter)
# ---------------------------------------------------------
#
# Previously we imported ollama directly here:
#   from ollama import chat
#
# Now we import from llm_provider, which routes to the
# right backend based on LLM_PROVIDER in .env.
# This means agent.py works identically whether you're
# using local Ollama or cloud OpenRouter.
from ai_document_agent.llm_provider import (
    get_system_prefix,
    llm_chat,
    llm_stream,
    make_assistant_msg,
    make_tool_msg,
)

from ai_document_agent.data_analyzer import (
    build_analysis_context,
    find_extracted_table_csv,
    find_upload_path,
    is_tabular_file,
    load_csv,
)
from ai_document_agent.pdf_processor import (
    generate_embeddings,
    get_document_chunk_count,
    get_page_chunks,
    list_documents,
    search_documents,
)

# ---------------------------------------------------------
# Semantic Query Cache — fast answers for repeated questions
# ---------------------------------------------------------
#
# WHY IMPORT HERE?
#
#   The cache sits in the agent's main flow (stream_agent):
#
#   1. BEFORE the LLM call: check if a similar question
#      was already answered → if yes, return cached answer
#      instantly (skip the entire LLM pipeline).
#
#   2. AFTER the LLM call: store the new answer so future
#      similar questions can be served from cache.
#
#   generate_embeddings() from pdf_processor is needed to
#   create a query embedding vector — the same model
#   (nomic-embed-text) used for document chunks. This
#   ensures the cache embeddings are in the SAME vector
#   space as the document embeddings, making cosine
#   similarity comparisons meaningful.

from ai_document_agent.tenancy import visitor_upload_dir
from ai_document_agent.query_cache import (
    lookup_cache,
    store_in_cache,
)

# ---------------------------------------------------------
# Content Renderer — HTML templates for generated content
# ---------------------------------------------------------
#
# Phase 3: When the user asks for a quiz, summary, article,
# or Q&A, the LLM generates structured JSON and the
# content_renderer converts it into rich, interactive HTML.
from ai_document_agent.content_renderer import (
    CONTENT_PATTERNS,
    render_content,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Model and LLM options have been MOVED to llm_provider.py
# ---------------------------------------------------------
#
# Previously these lived here:
#   MODEL = "qwen3:8b"
#   LLM_OPTIONS = {"num_thread": 6, "num_ctx": 2048, ...}
#
# Now they're in llm_provider.py, which manages all
# provider-specific settings (model names, API keys,
# performance options). This keeps agent.py focused on
# the LOGIC (what to send to the LLM) rather than the
# MECHANICS (how to call the LLM).


# ---------------------------------------------------------
# Upload directory — needed to find CSV files for analysis
# ---------------------------------------------------------
#
# Each visitor has their own upload folder (tenancy.py), so
# this is a function call rather than a constant: filename
# lookups must only ever see the current visitor's files.


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

# get_system_prefix() returns "/no_think\n" for Ollama
# (disables Qwen3's slow reasoning mode) and "" for
# cloud providers (they don't have this feature).
# ---------------------------------------------------------
# SYSTEM PROMPT — instructions for the LLM
#
#   This is the "personality" of our document agent. It
#   tells the LLM how to behave when answering questions.
#
#   KEY DISTINCTION (Fix 5 — full content display):
#
#     The prompt now handles TWO types of user requests
#     differently:
#
#     1. ANALYTICAL QUESTIONS
#        "What tool is used for machine learning?"
#        "How many students scored above 80%?"
#        → Short, concise 1-3 sentence answers
#        → Extract the specific answer from the document
#
#     2. DISPLAY / SHOW REQUESTS
#        "Show me the content on page 3"
#        "What is on page 5?"
#        "Display page 2"
#        → Reproduce the FULL text from the document
#        → Don't summarize, don't skip anything
#        → The user wants to READ the actual content
#
#   WHY THIS MATTERS:
#     Previously, the prompt said "Answer in 1-3 sentences"
#     for ALL questions. So when a user asked "show me
#     page 3", the LLM compressed 11 questions into 3
#     bullet points — losing 8 questions entirely.
#
#     Now the LLM checks: is the user asking a question
#     (give a short answer) or asking to SEE the content
#     (show everything)?
# ---------------------------------------------------------
SYSTEM_PROMPT = (
    get_system_prefix()
    + "Document analysis agent.\n"
    "\n"
    "RESPONSE LENGTH RULES:\n"
    "- For specific questions (who, what, how many, which), "
    "answer in 1-3 sentences. Be concise.\n"
    "- For 'show me', 'display', 'what is on page X', or "
    "'content of page X' requests, reproduce the FULL text "
    "from the document context. Do NOT summarize or skip "
    "any content. The user wants to read everything on "
    "that page.\n"
    "\n"
    "GENERAL RULES:\n"
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
# Page-number detection in user queries
# ---------------------------------------------------------
#
# WHY THIS EXISTS:
#
#   When the user asks "what's on page 4" or "show me page 3",
#   semantic search can't help — page numbers have no meaning
#   in vector space. "Page 4" could contain anything.
#
#   Instead, we detect the page number from the question and
#   fetch chunks directly from ChromaDB using metadata
#   filtering (where page=4). This bypasses semantic search
#   entirely for page-reference queries.
#
# HOW IT WORKS:
#
#   We use a regex to find patterns like:
#     "page 4", "page 12", "pg 3", "pg. 5"
#     "content on page 4", "what's on page 3"
#     "page number 7", "page no 2", "page no. 6"
#
#   The regex is intentionally broad to catch common
#   variations. If it matches, we return the page number.
#   If not, we return None and normal search proceeds.
#
# EXAMPLES:
#   "what is on page 4?"          → 4
#   "show me pg 3"                → 3
#   "content of page no. 12"      → 12
#   "summarize the revenue data"  → None (not a page query)

import re as _re

_PAGE_PATTERN = _re.compile(
    r'\bpag(?:e|es?)\.?\s*(?:no\.?\s*|number\s*)?'
    r'(\d{1,4})\b',
    _re.IGNORECASE,
)

# Also catch "pg 4", "pg. 4" (abbreviation)
_PG_PATTERN = _re.compile(
    r'\bpg\.?\s*(\d{1,4})\b',
    _re.IGNORECASE,
)


def _detect_page_query(question: str) -> int | None:
    """Detect if the user is asking about a specific page.

    Looks for patterns like "page 4", "pg 3", "page no. 12"
    in the user's question.

    Args:
        question: The user's question text.

    Returns:
        The page number (int) if detected, or None if the
        question is not about a specific page.
    """

    # Try "page X" pattern first (most common)
    match = _PAGE_PATTERN.search(question)
    if match:
        return int(match.group(1))

    # Try "pg X" abbreviation
    match = _PG_PATTERN.search(question)
    if match:
        return int(match.group(1))

    return None


def _merge_overlapping_chunks(chunks: list, overlap: int = 100) -> str:
    """Merge overlapping text chunks into one continuous string.

    WHY THIS IS NEEDED (Fix 6 — chunk boundary merging):

      When we split a page into chunks with overlap=100,
      each chunk repeats the last 100 characters of the
      previous chunk. For example:

        Chunk 1: "...Question 9. Which learning method (c)"
        Chunk 2: "method (c) Reinforcement learning (d)..."

      Without merging, the LLM shows these as separate
      "Excerpt 1" and "Excerpt 2", cutting Question 9
      in half. The user sees "(c)" then a break.

      This function detects the overlapping text between
      consecutive chunks and stitches them together into
      one continuous string — so the LLM sees the full
      page as one block of text, not fragments.

    HOW IT WORKS:

      For each pair of consecutive chunks, we look for
      the longest suffix of chunk A that matches a prefix
      of chunk B. That's the overlap. We skip it when
      appending chunk B.

      Example:
        Chunk A ends with:   "...learning method (c)"
        Chunk B starts with: "method (c) Reinforcement..."
        Overlap detected:    "method (c)"
        Result: "...learning method (c) Reinforcement..."

    Args:
        chunks: List of Evidence objects with .text attribute,
                sorted by page/position order.
        overlap: Expected overlap size in characters.
                 We search ±50% around this for flexibility.

    Returns:
        A single merged text string.
    """

    if not chunks:
        return ""

    if len(chunks) == 1:
        return chunks[0].text

    # Start with the first chunk's full text
    merged = chunks[0].text

    for i in range(1, len(chunks)):
        next_text = chunks[i].text

        # Search for overlap between the end of 'merged'
        # and the start of 'next_text'.
        #
        # We check overlap sizes from largest to smallest
        # (greedy). The expected overlap is ~100 chars, but
        # we search from 150 down to 20 to handle edge cases
        # (e.g. chunks near page boundaries may have shorter
        # or longer overlaps).
        best_overlap = 0
        max_check = min(len(merged), len(next_text), overlap * 2)

        for ov_size in range(max_check, 19, -1):
            # Does the end of 'merged' match the start of
            # 'next_text'?
            if merged[-ov_size:] == next_text[:ov_size]:
                best_overlap = ov_size
                break

        if best_overlap > 0:
            # Skip the overlapping part of next_text
            merged += next_text[best_overlap:]
        else:
            # No overlap found — just concatenate with
            # a newline separator (may be from different
            # parts of the page)
            merged += "\n" + next_text

    return merged


# =============================================================
# PHASE 3: Content Generation Detection & Prompts
# =============================================================
#
# These functions detect when the user wants to GENERATE
# content (quiz, summary, article, Q&A) rather than just
# ASK a question. When detected, we use a special prompt
# that tells the LLM to output structured JSON instead
# of plain text.
#
# FLOW:
#   1. _detect_content_request() checks if the question
#      matches any content generation keywords
#   2. If matched, _build_generation_prompt() creates a
#      structured prompt for that content type
#   3. The LLM returns JSON
#   4. render_content() converts JSON → HTML
#   5. HTML is sent to the frontend for rendering
# =============================================================

import json as _json


# ---------------------------------------------------------
# Smart Suggestions — follow-up question generation
# ---------------------------------------------------------
#
# WHY SUGGEST FOLLOW-UP QUESTIONS?
#
#   Users often don't know what to ask next. After getting
#   an answer, they might think "that's interesting, but
#   what else can I learn?" Follow-up suggestions guide
#   the user deeper into the document.
#
# TWO TYPES OF SUGGESTIONS:
#
#   1. DOCUMENT SUGGESTIONS (after upload):
#      Based on the first ~1000 chars of the document,
#      generate 4 starter questions like:
#        "What are the key findings?"
#        "Who are the main people mentioned?"
#
#   2. FOLLOW-UP SUGGESTIONS (after each answer):
#      Based on the Q&A just completed, generate 3
#      follow-up questions the user might want to ask.
#
# HOW IT WORKS:
#
#   We make a lightweight LLM call with a simple prompt
#   asking for 3-4 short questions as a JSON array.
#   The LLM call is fast because:
#     - Short prompt (no document context needed)
#     - Short output (just 3-4 one-line questions)
#     - No tools, no streaming, no multi-step loop

def generate_document_suggestions(
    document_text: str,
    filename: str,
) -> list[str]:
    """Generate starter questions for a newly uploaded document.

    Called once after document processing completes. The
    suggestions are cached by main.py so we don't regenerate
    them on every page load.

    Args:
        document_text: First ~1000 chars of the document.
        filename: The document's filename (for context).

    Returns:
        List of 3-4 suggested questions, or empty list
        if generation fails.
    """

    prompt = (
        "You are a helpful assistant. Based on this document "
        "excerpt, suggest exactly 4 short questions a user "
        "would want to ask about this document. Each question "
        "should be specific to the content, not generic.\n\n"
        f"Document: {filename}\n"
        f"Excerpt:\n{document_text[:1000]}\n\n"
        "Return ONLY a JSON array of 4 question strings, "
        "nothing else. Example:\n"
        '["What is the total revenue?", '
        '"Who scored the highest?", '
        '"How many items are listed?", '
        '"What date was this issued?"]\n'
    )

    try:
        response = llm_chat(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "Generate 4 questions.",
                },
            ],
            tools=None,
        )

        raw = response.content or ""

        # Parse JSON from the LLM response.
        # The LLM might wrap it in ```json ... ``` blocks,
        # so we strip those first.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove markdown code fences
            cleaned = re.sub(
                r"^```(?:json)?\s*", "", cleaned
            )
            cleaned = re.sub(r"\s*```$", "", cleaned)

        suggestions = _json.loads(cleaned)

        # Validate: must be a list of strings
        if (
            isinstance(suggestions, list)
            and all(isinstance(s, str) for s in suggestions)
        ):
            logger.info(
                "Generated %d document suggestions for %s",
                len(suggestions),
                filename,
            )
            return suggestions[:4]  # Cap at 4

        logger.warning(
            "Document suggestions not a list of strings: %s",
            type(suggestions),
        )
        return []

    except Exception as exc:
        logger.warning(
            "Failed to generate document suggestions: %s",
            exc,
        )
        return []


def generate_followup_suggestions(
    question: str,
    answer: str,
    content_type: str | None = None,
) -> list[str]:
    """Generate follow-up questions after an answer.

    Called after each completed answer. The suggestions
    are sent as a streaming event so the frontend can
    display them as clickable chips.

    Args:
        question: The user's original question.
        answer: The assistant's answer (truncated to
                first 500 chars for efficiency).
        content_type: "flashcards", "quiz", etc. or None
                      for plain text answers.

    Returns:
        List of 3 suggested follow-up questions, or
        empty list if generation fails.
    """

    # For interactive content, tailor the follow-ups
    if content_type:
        context_hint = (
            f"The user asked for {content_type} and received "
            f"them. Suggest follow-up questions they might "
            f"want to ask about the document (not about the "
            f"{content_type} themselves)."
        )
    else:
        context_hint = (
            f"The user asked: \"{question}\"\n"
            f"The answer was: \"{answer[:500]}\"\n"
            f"Suggest 3 natural follow-up questions."
        )

    prompt = (
        "You are a helpful assistant. Based on this Q&A "
        "exchange, suggest exactly 3 short follow-up "
        "questions the user might want to ask next. "
        "Questions should be specific, varied, and "
        "explore different aspects of the document.\n\n"
        f"{context_hint}\n\n"
        "Return ONLY a JSON array of 3 question strings, "
        "nothing else. Keep each question under 60 chars.\n"
    )

    try:
        response = llm_chat(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "Generate 3 follow-ups.",
                },
            ],
            tools=None,
        )

        raw = response.content or ""

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*", "", cleaned
            )
            cleaned = re.sub(r"\s*```$", "", cleaned)

        suggestions = _json.loads(cleaned)

        if (
            isinstance(suggestions, list)
            and all(isinstance(s, str) for s in suggestions)
        ):
            logger.info(
                "Generated %d follow-up suggestions",
                len(suggestions),
            )
            return suggestions[:3]  # Cap at 3

        return []

    except Exception as exc:
        logger.warning(
            "Failed to generate follow-up suggestions: %s",
            exc,
        )
        return []


def _detect_content_request(question: str) -> str | None:
    """Detect if the user wants to generate content.

    Checks the user's question against keyword patterns
    for each content type (quiz, summary, article, qa,
    flashcards).

    HOW IT WORKS:
      We lowercase the question and check if ANY keyword
      from each content type appears in it. The first
      match wins.

      Examples:
        "Create a quiz from page 2"     → "quiz"
        "Summarize this document"       → "summary"
        "Write an article about AI"     → "article"
        "Give me questions and answers" → "qa"
        "Make flashcards for revision"  → "flashcards"
        "What is Big Data?"             → None (normal Q&A)

    Args:
        question: The user's question text.

    Returns:
        Content type string ("quiz", "summary", etc.)
        or None if this is a normal question.
    """

    q_lower = question.lower()

    for content_type, keywords in CONTENT_PATTERNS.items():
        for keyword in keywords:
            if keyword in q_lower:
                logger.info(
                    "Content request detected: '%s' "
                    "(matched keyword: '%s')",
                    content_type,
                    keyword,
                )
                return content_type

    return None


def _build_generation_prompt(
    content_type: str,
    question: str,
    context: str,
    source: str,
) -> str:
    """Build a structured prompt for content generation.

    Each content type has a specific prompt that tells the
    LLM exactly what JSON structure to return.

    WHY STRUCTURED PROMPTS?
      Free/small LLMs need very explicit instructions to
      output valid JSON. We:
      1. Tell it the EXACT JSON structure with field names
      2. Give a concrete example
      3. Say "output ONLY the JSON, nothing else"
      4. Remind it to use the document context

    Args:
        content_type: One of "quiz", "summary", "article",
                      "qa", "flashcards".
        question: The user's original question.
        context: Document context text (from retrieval).
        source: Source document filename.

    Returns:
        Complete prompt string for the LLM.
    """

    # -------------------------------------------------
    # QUIZ PROMPT
    #
    #   Tells the LLM to generate MCQ questions with
    #   4 options each and a correct answer index.
    #   The LLM extracts questions from the document
    #   or creates new ones based on the content.
    # -------------------------------------------------
    if content_type == "quiz":
        return f"""You are a quiz generator. Based on the document content below,
create a multiple-choice quiz.

DOCUMENT CONTEXT:
{context}

USER REQUEST: {question}

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no explanation, no code fences).
The JSON must have this exact structure:
{{
  "type": "quiz",
  "title": "Quiz title here",
  "source": "{source}",
  "questions": [
    {{
      "q": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": 0
    }}
  ]
}}

RULES:
- "answer" is the 0-based index of the correct option (0=first, 1=second, etc.)
- Generate 5-10 questions from the document content
- Each question must have exactly 4 options
- Questions should cover different topics from the document
- If the document already contains MCQ questions, use those directly
- Make sure the correct answer is actually correct based on the document
- Output ONLY the JSON object, nothing else"""

    # -------------------------------------------------
    # SUMMARY PROMPT
    # -------------------------------------------------
    elif content_type == "summary":
        return f"""You are a document summarizer. Based on the document content below,
create a summary with key points.

DOCUMENT CONTEXT:
{context}

USER REQUEST: {question}

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no explanation, no code fences).
The JSON must have this exact structure:
{{
  "type": "summary",
  "title": "Summary title here",
  "source": "{source}",
  "points": [
    "First key point here",
    "Second key point here"
  ]
}}

RULES:
- Extract 5-10 key points from the document
- Each point should be 1-2 sentences
- Cover the most important information
- Use clear, simple language
- Output ONLY the JSON object, nothing else"""

    # -------------------------------------------------
    # ARTICLE PROMPT
    # -------------------------------------------------
    elif content_type == "article":
        return f"""You are an article writer. Based on the document content below,
write a well-structured article.

DOCUMENT CONTEXT:
{context}

USER REQUEST: {question}

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no explanation, no code fences).
The JSON must have this exact structure:
{{
  "type": "article",
  "title": "Article title here",
  "source": "{source}",
  "sections": [
    {{
      "heading": "Section heading",
      "content": "Section content paragraph(s) here."
    }}
  ]
}}

RULES:
- Create 3-5 sections with clear headings
- Each section should have 2-4 paragraphs
- Write in clear, educational language
- Base all content on the document — do not invent facts
- Output ONLY the JSON object, nothing else"""

    # -------------------------------------------------
    # Q&A PROMPT
    # -------------------------------------------------
    elif content_type == "qa":
        return f"""You are a Q&A generator. Based on the document content below,
create questions with detailed answers.

DOCUMENT CONTEXT:
{context}

USER REQUEST: {question}

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no explanation, no code fences).
The JSON must have this exact structure:
{{
  "type": "qa",
  "title": "Questions & Answers title",
  "source": "{source}",
  "pairs": [
    {{
      "question": "Question text here?",
      "answer": "Detailed answer here."
    }}
  ]
}}

RULES:
- Generate 5-10 question-answer pairs
- Answers should be detailed (2-3 sentences each)
- Cover different topics from the document
- Base all answers on the document content
- Output ONLY the JSON object, nothing else"""

    # -------------------------------------------------
    # FLASHCARDS PROMPT
    # -------------------------------------------------
    #
    # COUNT EXTRACTION:
    #   The user often specifies how many flashcards they
    #   want: "create 5 flashcards", "make 10 cards", etc.
    #   We extract this number from the question and inject
    #   it into the prompt. If no number is specified, we
    #   default to 8-12 flashcards.
    #
    #   Previously the prompt always said "Create 8-15",
    #   which caused the LLM to ignore "create 5 flashcards"
    #   and generate 10+ cards instead.

    elif content_type == "flashcards":

        # Extract requested count from the user's question
        # Match patterns like "5 flashcard", "10 cards",
        # "create 3 flash cards", etc.
        import re
        count_match = re.search(
            r"(\d+)\s*(?:flash\s*cards?|cards?)",
            question.lower(),
        )

        if count_match:
            # User specified an exact count
            requested_count = int(count_match.group(1))
            count_instruction = (
                f"- Create EXACTLY {requested_count} "
                f"flashcards (the user asked for "
                f"{requested_count})"
            )
        else:
            # No count specified — use a reasonable default
            count_instruction = "- Create 8-12 flashcards"

        return f"""You are a flashcard creator. Based on the document content below,
create study flashcards with a question on the front and answer on the back.

DOCUMENT CONTEXT:
{context}

USER REQUEST: {question}

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no explanation, no code fences).
The JSON must have this exact structure:
{{
  "type": "flashcards",
  "title": "Flashcards title",
  "source": "{source}",
  "cards": [
    {{
      "front": "Question or term on front",
      "back": "Answer or definition on back"
    }}
  ]
}}

RULES:
{count_instruction}
- Front should be a short question or key term
- Back should be a concise answer or definition
- Cover different topics from the document
- Output ONLY the JSON object, nothing else"""

    else:
        # Fallback — shouldn't reach here
        return question


def _parse_llm_json(text: str) -> dict | None:
    """Parse JSON from LLM output, handling common issues.

    WHY THIS IS NEEDED:
      LLMs don't always return clean JSON. Common issues:
      1. Markdown code fences: ```json ... ```
      2. Extra text before/after the JSON
      3. Trailing commas (invalid JSON)
      4. Single quotes instead of double quotes

      This function tries to extract and parse the JSON
      despite these issues.

    Args:
        text: Raw LLM output text.

    Returns:
        Parsed dictionary, or None if parsing fails.
    """

    if not text:
        return None

    # Step 1: Remove markdown code fences if present
    #   LLMs sometimes wrap JSON in ```json ... ```
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.find("\n")
        if first_newline > 0:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Step 2: Find the JSON object boundaries
    #   Look for the outermost { ... } pair
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        logger.warning(
            "No JSON object found in LLM output: %s...",
            text[:100],
        )
        return None

    json_str = text[start:end + 1]

    # Step 3: Try to parse the JSON
    try:
        data = _json.loads(json_str)
        logger.info(
            "Successfully parsed LLM JSON: type=%s",
            data.get("type", "unknown"),
        )
        return data
    except _json.JSONDecodeError as exc:
        # Step 4: Try fixing common issues
        logger.warning(
            "JSON parse failed: %s. Attempting fixes...",
            exc,
        )

        # Fix: Remove trailing commas before } or ]
        import re as _re_fix
        fixed = _re_fix.sub(r',\s*([}\]])', r'\1', json_str)

        try:
            data = _json.loads(fixed)
            logger.info(
                "JSON parsed after fixing trailing commas: "
                "type=%s",
                data.get("type", "unknown"),
            )
            return data
        except _json.JSONDecodeError:
            logger.error(
                "Failed to parse LLM JSON even after fixes. "
                "Raw output: %s",
                json_str[:200],
            )
            return None


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
#   But some downstream functions need a SPECIFIC filename:
#     - _find_csv_for_source("all") → None (no CSV found)
#     - _get_tools_for_source("all") → [] (no tools)
#     - build_context_prompt() → no analysis injected
#
# THE FIX (updated for multi-document support):
#
#   1 document  → auto-resolve "all" to that filename.
#                  Tools, analysis, everything works.
#
#   N documents → return None (not the string "all").
#                  None tells search_documents() to search
#                  across ALL documents without a where
#                  clause. Tools won't work in multi-doc
#                  mode (can't know which CSV to query),
#                  but text Q&A across all docs works.
#
#   This is the key change for MULTI-DOCUMENT SUPPORT:
#   before, "all" with N docs returned the string "all"
#   which broke everything. Now it returns None, which
#   search_documents() handles correctly by querying
#   ChromaDB without a source filter.
#
# HOW IT FITS IN THE PIPELINE:
#
#   Single doc:
#     1. source_filter = "all"
#     2. _resolve_source_filter("all") → "report.pdf"
#     3. Full tool + analysis support
#
#   Multiple docs:
#     1. source_filter = "all"
#     2. _resolve_source_filter("all") → None
#     3. search_documents(query, source_filter=None)
#        → searches ALL documents in ChromaDB
#     4. Text Q&A works, tools disabled (no specific CSV)

def _resolve_source_filter(
    source_filter: str | None,
) -> str | None:
    """Resolve 'all'/None to a specific filename or None.

    When the user hasn't selected a specific document
    (source_filter is "all" or None), but there's only
    one document uploaded, we auto-detect its filename.

    With multiple documents, returns None so downstream
    search functions query across all documents.

    Args:
        source_filter: The raw filter from the frontend.
            Could be "all", None, or a specific filename.

    Returns:
        A specific filename if auto-detected, or None
        for multi-document / cross-document search.
    """

    # Already a specific filename — use as-is.
    # Examples: "report.pdf", "data.csv"
    if (
        source_filter
        and source_filter.lower() != "all"
    ):
        return source_filter

    # No uploads directory — nothing to resolve
    upload_dir = visitor_upload_dir()

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

    for path in upload_dir.iterdir():
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
    if len(original_names) == 1:
        resolved = original_names[0]
        logger.info(
            "Auto-resolved source_filter: '%s' → '%s' "
            "(only one document uploaded)",
            source_filter,
            resolved,
        )
        return resolved

    # -------------------------------------------------
    # MULTI-DOCUMENT MODE (Phase 4):
    #
    #   Multiple documents uploaded but no specific one
    #   selected. Return None so search_documents() can
    #   query across ALL documents in ChromaDB.
    #
    #   Tools (filter_rows, aggregate_data) are NOT
    #   available in multi-doc mode because they need a
    #   specific CSV file. The LLM will answer from
    #   text context only — which is correct for cross-
    #   document questions like "compare Chapter 2 and
    #   Chapter 5".
    # -------------------------------------------------
    if len(original_names) > 1:
        logger.info(
            "Multi-document mode: %d documents (%s). "
            "Searching across all.",
            len(original_names),
            ", ".join(original_names),
        )
        return None

    return None


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


def _question_needs_arithmetic(question: str) -> bool:
    """True when answering needs a calculation over rows.

    Narrower than _question_needs_tools(): used for SMALL
    tables, where lookups are answered fine from context but
    sums, averages and percentages are not.
    """
    q = question.lower()
    arithmetic_keywords = [
        "average", "mean", "sum", "total",
        "percentage", "percent", "combined", "aggregate",
    ]
    return any(kw in q for kw in arithmetic_keywords)


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
            source_filter, visitor_upload_dir(),
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
                elif _question_needs_arithmetic(question):
                    # Small table, but the question needs a
                    # calculation. Even a 5-row table is
                    # enough for an LLM to get a sum or an
                    # average wrong, so compute it in code.
                    logger.info(
                        "PDF table CSV has %d rows — question "
                        "needs arithmetic, providing CSV tools",
                        len(rows),
                    )
                    return CSV_TOOLS
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

    # ---------------------------------------------------------
    # DOCUMENT METADATA — page counts, chunk counts
    # ---------------------------------------------------------
    #
    # WHY IS THIS NEEDED?
    #
    #   Without this, the LLM has NO way to know how many
    #   pages a document has. When a user asks "how many
    #   pages?", the LLM would guess from the page numbers
    #   it sees in the retrieved chunks — e.g. if chunks
    #   reference pages 1-4 but not page 5, it answers "4"
    #   even though the document has 5 pages.
    #
    #   By injecting the actual metadata (from list_documents,
    #   which counts unique page numbers in ChromaDB), the
    #   LLM has the definitive answer for metadata questions.
    #
    # WHAT'S INCLUDED:
    #   - Filename
    #   - Total pages (from ChromaDB metadata, not guessing)
    #   - Total chunks indexed
    #
    # WHERE IT GOES:
    #   At the very top of the DOCUMENT CONTEXT section,
    #   before any text excerpts or analysis data.

    doc_metadata_lines = []
    try:
        all_docs = list_documents()
        if all_docs:
            doc_metadata_lines.append(
                "\nDOCUMENT METADATA (use these for "
                "questions about document properties "
                "like page count, file names, etc.):"
            )
            for doc in all_docs:
                # If user is filtering to a specific document,
                # only show that document's metadata.
                # If no filter, show all documents.
                if (
                    source_filter is None
                    or doc["filename"] == source_filter
                ):
                    doc_metadata_lines.append(
                        f"  - {doc['filename']}: "
                        f"{doc['pages']} pages, "
                        f"{doc['chunks']} chunks indexed"
                    )
    except Exception as exc:
        logger.warning(
            "Failed to get document metadata: %s", exc
        )

    metadata_block = "\n".join(doc_metadata_lines)

    if skip_chunks:
        logger.info(
            "Pure CSV source — skipping chunk retrieval, "
            "using analysis context only",
        )
        parts = ["\n\nDOCUMENT CONTEXT:" + metadata_block]
        results = []
    else:
        # ---------------------------------------------------------
        # PAGE-SPECIFIC RETRIEVAL (Fix 2)
        #
        #   When the user asks about a specific page (e.g.
        #   "what is on page 3?", "summarise pg 5"), we detect
        #   the page number with _detect_page_query() and pull
        #   ALL chunks for that page directly from ChromaDB
        #   using metadata filtering (WHERE page == N).
        #
        #   This is much more reliable than semantic search for
        #   page-specific questions because:
        #     1. Semantic search matches by meaning, not page
        #        number — "page 3" has no semantic similarity
        #        to the actual content on page 3.
        #     2. ChromaDB metadata filtering is exact — it
        #        returns every chunk tagged with that page.
        #
        #   We ALSO run the normal semantic search and merge
        #   results: page chunks come first (sorted by page),
        #   then any additional semantic results that aren't
        #   duplicates. This way the LLM sees the exact page
        #   content PLUS any related context from other pages.
        # ---------------------------------------------------------
        requested_page = _detect_page_query(question)
        page_chunks: list = []

        if requested_page is not None:
            # User asked about a specific page — fetch all its
            # chunks via ChromaDB metadata filter.
            logger.info(
                "Page-specific query detected: page %d",
                requested_page,
            )
            page_chunks = get_page_chunks(
                requested_page,
                source_filter=source_filter,
            )
            logger.info(
                "Found %d chunks for page %d",
                len(page_chunks), requested_page,
            )

        try:
            # -------------------------------------------------
            # DYNAMIC CHUNK RETRIEVAL (was hardcoded n_results=5)
            # -------------------------------------------------
            #
            # THE PROBLEM WITH HARDCODED n_results=5:
            #
            #   A 14-page budget PDF produces ~50-80 chunks
            #   (500 chars each). Retrieving only 5 chunks means
            #   the LLM sees ~6-10% of the document. When a user
            #   asks "total budget allocated?", the answer might
            #   be spread across 10+ chunks (tables, summaries,
            #   department breakdowns). 5 chunks can't capture
            #   that — the LLM gives a partial, incorrect answer.
            #
            # THE FIX — DYNAMIC CALCULATION:
            #
            #   Instead of a fixed number, Python checks how many
            #   chunks this document actually has in ChromaDB,
            #   then calculates a smart retrieval count:
            #
            #   Formula: n_results = max(5, min(total * 0.3, 25))
            #
            #   - MINIMUM 5: Even tiny documents get enough context
            #   - 30% OF TOTAL: Scales with document size
            #     • 20-chunk doc → 6 chunks (30%)
            #     • 50-chunk doc → 15 chunks (30%)
            #     • 100-chunk doc → 25 chunks (capped)
            #   - MAXIMUM 25: Prevents enormous prompts that would
            #     slow down the LLM and waste tokens. OpenRouter's
            #     context window is 262K tokens, but bigger prompts
            #     = slower responses + higher cost.
            #
            # WHY 30%?
            #
            #   Hybrid search (semantic + BM25) is smart about
            #   WHICH chunks it picks — it finds the most relevant
            #   ones, not random ones. So 30% of well-chosen chunks
            #   usually covers the answer. Going higher (50%+) adds
            #   diminishing returns: more irrelevant padding, slower
            #   responses, same quality.
            #
            # WHY NOT RETRIEVE ALL CHUNKS?
            #
            #   1. LLM prompt size: 100 chunks × 500 chars = 50K
            #      chars ≈ 12K tokens just for context. That leaves
            #      less room for the conversation + answer.
            #   2. Noise: Irrelevant chunks confuse the LLM. It
            #      might quote a header or footer instead of the
            #      actual answer.
            #   3. Speed: Bigger prompts = slower inference.
            #   4. Cost: More input tokens = higher API cost.
            # -------------------------------------------------

            # Step 1: Get chunk count for this document
            # (If source_filter is None, counts ALL chunks)
            total_chunks = get_document_chunk_count(
                source=source_filter,
            )

            # Step 2: Calculate dynamic retrieval count
            #   - At least 5 (floor for small documents)
            #   - Up to 30% of total chunks
            #   - Never more than 25 (ceiling for huge docs)
            if total_chunks > 0:
                dynamic_n = max(5, min(
                    int(total_chunks * 0.3),
                    25,
                ))
            else:
                # No chunks found (maybe wrong filename?) —
                # fall back to safe default
                dynamic_n = 5

            logger.info(
                "Dynamic retrieval: %d total chunks → "
                "retrieving %d (%.0f%%)",
                total_chunks,
                dynamic_n,
                (dynamic_n / total_chunks * 100)
                if total_chunks > 0 else 0,
            )

            results = search_documents(
                query=question,
                n_results=dynamic_n,
                source_filter=source_filter,
            )
        except Exception as exc:
            logger.warning(
                "Document search failed: %s", exc
            )
            # If we have page chunks, we can still answer —
            # only return empty if we have nothing at all.
            if not page_chunks:
                return ""
            results = []

        # ---------------------------------------------------------
        # MERGE page-specific chunks with semantic search results.
        #
        #   Page chunks go FIRST so the LLM sees the exact page
        #   content at the top of the context. Then we add any
        #   semantic results whose text isn't already included
        #   (deduplication by exact text match).
        #
        #   This means: if the user asks "what's on page 3?",
        #   they get ALL of page 3 plus any semantically related
        #   content from other pages.
        # ---------------------------------------------------------
        if page_chunks:
            # Collect text from page chunks for deduplication
            seen_texts = {e.text for e in page_chunks}
            # Start with page chunks, then add unique semantic results
            merged = list(page_chunks)
            for r in results:
                if r.text not in seen_texts:
                    merged.append(r)
                    seen_texts.add(r.text)
            results = merged
            logger.info(
                "Merged results: %d page + %d semantic = %d total",
                len(page_chunks),
                len(results) - len(page_chunks),
                len(results),
            )

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
            parts = ["\n\nDOCUMENT CONTEXT:" + metadata_block]

            # -------------------------------------------------
            # PAGE-SPECIFIC DISPLAY (Fix 6)
            #
            #   When the user asked about a specific page
            #   (requested_page is set), we GROUP chunks by
            #   page and MERGE same-page chunks into one
            #   continuous text block.
            #
            #   WHY? Without merging, each chunk is shown as
            #   a separate "Excerpt" with boundaries. Content
            #   that spans two chunks (like Question 9 split
            #   across Chunk 3 and Chunk 4) gets cut in half.
            #
            #   By merging overlapping chunks from the same
            #   page, the LLM sees one clean block of text
            #   per page — no cuts, no fragments.
            #
            #   For normal (non-page) queries, we keep the
            #   original per-chunk format because the scores
            #   and sources are useful for the LLM.
            # -------------------------------------------------
            if requested_page is not None:
                # Group chunks by page number
                from collections import defaultdict
                pages_map = defaultdict(list)
                for result in results:
                    pages_map[result.page].append(result)

                # Merge chunks within each page, then format
                # as one block per page
                for page_num in sorted(pages_map.keys()):
                    page_results = pages_map[page_num]
                    source = page_results[0].source

                    # Merge overlapping chunks into one string
                    merged_text = _merge_overlapping_chunks(
                        page_results,
                    )

                    parts.append(
                        f"\n--- Page {page_num} "
                        f"(Source: {source}) ---\n"
                        f"Content:\n{merged_text}\n"
                    )

                parts.append(
                    "\nThe above is the FULL content from the "
                    "requested page(s). Reproduce it completely "
                    "— do not summarize or skip any content. "
                    "Cite the source and page."
                )
            else:
                # Normal query: show individual excerpts with
                # scores (helps the LLM weigh relevance)
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
            parts = ["\n\nDOCUMENT CONTEXT:" + metadata_block]

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
        return find_upload_path(source, visitor_upload_dir())

    # PDF with extracted table
    if source.lower().endswith(".pdf"):
        return find_extracted_table_csv(
            source, visitor_upload_dir(),
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
        # Single-document mode — tell the LLM which
        # document it's working with
        system_content += (
            f"\nYou are answering questions about: "
            f"{source_filter}\n"
        )
    else:
        # -------------------------------------------------
        # MULTI-DOCUMENT MODE (Phase 4):
        #
        #   No specific document selected. The context may
        #   contain chunks from MULTIPLE documents. Tell
        #   the LLM to cite which document each piece of
        #   information comes from — this is critical when
        #   the user asks cross-document questions like
        #   "compare Chapter 2 and Chapter 5".
        # -------------------------------------------------
        system_content += (
            "\nYou are searching across ALL uploaded "
            "documents. When answering, always cite which "
            "document (Source filename) each piece of "
            "information comes from.\n"
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

        # -------------------------------------------------
        # CHANGED: was ollama.chat(), now llm_chat()
        # llm_chat() routes to Ollama or OpenRouter based
        # on LLM_PROVIDER in .env. Returns a unified
        # LLMResponse with .content and .tool_calls.
        # -------------------------------------------------
        response = llm_chat(
            messages=working_messages,
            tools=active_tools,
        )

        llm_time = perf_counter() - llm_start

        logger.info(
            "[%s] LLM round %d: %.2fs",
            request_id,
            round_num + 1,
            llm_time,
        )

        # No tool calls → LLM gave a direct answer
        # CHANGED: was response.message.tool_calls,
        #   now response.tool_calls (unified format)
        if not response.tool_calls:
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

            # CHANGED: was response.message.content,
            #   now response.content (unified format)
            return response.content

        # Tool calls — add the assistant's decision to
        # the message history so the next LLM call sees
        # which tools were called.
        #
        # CHANGED: was working_messages.append(response.message)
        #   Now we use make_assistant_msg() which creates
        #   the right format for Ollama or OpenRouter.
        working_messages.append(
            make_assistant_msg(
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )

        for tool_call in response.tool_calls:

            # CHANGED: was tool_call.function.name/arguments,
            #   now tool_call.name/arguments (unified format)
            tool_name = tool_call.name
            arguments = tool_call.arguments

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
                # CHANGED: use make_tool_msg() for the
                #   right format per provider
                working_messages.append(
                    make_tool_msg(
                        tool_name=tool_name,
                        tool_call_id=tool_call.id,
                        content=(
                            f"Error: Unknown tool "
                            f"'{tool_name}'."
                        ),
                    )
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

            # CHANGED: use make_tool_msg() instead of
            #   hardcoded {"role": "tool", "tool_name": ...}
            working_messages.append(
                make_tool_msg(
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    content=str(result),
                )
            )

    # If we exhausted all rounds, do one final LLM call
    # without tools to force a text answer.

    logger.info(
        "[%s] Max tool rounds reached, forcing answer",
        request_id,
    )

    # CHANGED: was ollama.chat(), now llm_chat()
    #   No tools passed → LLM must give a text answer.
    final_response = llm_chat(
        messages=working_messages,
    )

    total_time = perf_counter() - start_time

    logger.info(
        "[%s] Forced final answer: %.2fs",
        request_id,
        total_time,
    )

    # CHANGED: was final_response.message.content
    return final_response.content


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

    # query_embedding is used for cache lookup AND cache
    # storage later. Initialized to None; set to the actual
    # embedding vector if cache lookup succeeds. If it stays
    # None, we skip cache storage (e.g., if the embedding
    # generation failed or the question was empty).
    query_embedding = None

    # -------------------------------------------------------
    # STEP 0.5: Extract the user's latest question
    # -------------------------------------------------------
    #
    # WHY EXTRACT EARLY?
    #
    #   We need the question text BEFORE forced retrieval
    #   so we can check the semantic cache first. If the
    #   cache has an answer, we skip retrieval AND the LLM
    #   call entirely — that's where the big speed gain is.
    #
    #   Previously, latest_question was extracted inside
    #   the retrieval step. Moving it here lets us use it
    #   for both cache lookup AND retrieval.

    latest_question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_question = msg.get("content", "")
            break

    # -------------------------------------------------------
    # STEP 0.6: Semantic cache lookup
    # -------------------------------------------------------
    #
    # HOW THIS WORKS:
    #
    #   1. Generate an embedding vector for the user's
    #      question using the SAME model (nomic-embed-text)
    #      used for document chunks. This ensures the
    #      vectors are in the same "meaning space."
    #
    #   2. Compare this vector against all cached question
    #      embeddings (for the same source_filter) using
    #      cosine similarity.
    #
    #   3. If a cached question has similarity >= 0.92,
    #      it means the same thing → return cached answer.
    #
    # EXAMPLE:
    #
    #   "how many pages?"           → embedding → [0.12, ...]
    #   "total number of pages?"    → embedding → [0.11, ...]
    #   cosine_similarity = 0.94 → CACHE HIT → instant answer!
    #
    # WHY CHECK BEFORE RETRIEVAL?
    #
    #   Forced retrieval calls ChromaDB + BM25 (~0.3s) and
    #   the LLM call takes 10-30s. If we have a cached
    #   answer, we skip ALL of that and return in <1s.
    #   The embedding generation (~0.5s) is the only cost.

    if latest_question:
        try:
            # Generate embedding for the user's question.
            # generate_embeddings() expects a LIST of texts
            # and returns a LIST of vectors. We pass one
            # question and take the first (only) vector.
            query_embedding = generate_embeddings(
                [latest_question]
            )[0]

            # Check the cache for a similar question
            cache_result = lookup_cache(
                query_embedding=query_embedding,
                source_filter=source_filter,
            )

            if cache_result is not None:
                # -----------------------------------------
                # CACHE HIT — return cached answer instantly
                # -----------------------------------------
                #
                # The cache stores both plain text answers
                # AND interactive content (flashcards, etc.)
                # with their HTML. We need to handle both.

                logger.info(
                    "[%s] Cache HIT (similarity=%.4f): "
                    "'%.50s'",
                    request_id,
                    cache_result["similarity"],
                    cache_result["question"],
                )

                # Tell the frontend this is a cached answer
                yield {
                    "type": "status",
                    "status": "cache_hit",
                    "message": (
                        f"Found cached answer "
                        f"(similarity: "
                        f"{cache_result['similarity']:.0%})"
                    ),
                    "elapsed_seconds": round(
                        perf_counter() - start_time, 2,
                    ),
                    "request_id": request_id,
                }

                cached_content_type = cache_result.get(
                    "content_type"
                )
                cached_html = cache_result.get(
                    "html_content"
                )

                if cached_content_type and cached_html:
                    # Interactive content (flashcards, quiz,
                    # summary, etc.) — send as html_content
                    # event so frontend renders in iframe

                    yield {
                        "type": "status",
                        "status": "first_token",
                        "message": (
                            f"{cached_content_type.title()}"
                            f" ready! (cached)"
                        ),
                        "first_token_seconds": round(
                            perf_counter() - start_time, 2,
                        ),
                        "request_id": request_id,
                    }

                    yield {
                        "type": "html_content",
                        "html": cached_html,
                        "content_type": cached_content_type,
                    }

                    total_time = perf_counter() - start_time

                    yield {
                        "type": "completed",
                        "response_time_seconds": round(
                            total_time, 2,
                        ),
                        "assistant_content": (
                            f"[Generated "
                            f"{cached_content_type}]"
                        ),
                        "html_content": cached_html,
                        "content_type": cached_content_type,
                        "cached": True,
                        "request_id": request_id,
                    }

                else:
                    # Plain text answer — stream as tokens
                    cached_answer = cache_result["answer"]

                    yield {
                        "type": "status",
                        "status": "first_token",
                        "message": (
                            "Generating answer... (cached)"
                        ),
                        "first_token_seconds": round(
                            perf_counter() - start_time, 2,
                        ),
                        "request_id": request_id,
                    }

                    yield {
                        "type": "token",
                        "content": cached_answer,
                    }

                    total_time = perf_counter() - start_time

                    yield {
                        "type": "completed",
                        "response_time_seconds": round(
                            total_time, 2,
                        ),
                        "assistant_content": cached_answer,
                        "cached": True,
                        "request_id": request_id,
                    }

                # Return early — skip retrieval + LLM
                return

        except Exception as exc:
            # If cache lookup fails (e.g., Ollama not running
            # for embeddings, or SQLite error), log it and
            # continue with the normal flow. The cache is a
            # performance OPTIMIZATION — if it breaks, the
            # app should still work, just slower.
            logger.warning(
                "[%s] Cache lookup failed, continuing "
                "without cache: %s",
                request_id,
                exc,
            )
            # Set query_embedding to None so we know not to
            # try storing in cache later
            query_embedding = None

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
    # STEP 1.5: Check for content generation request
    # -------------------------------------------------------
    #
    # PHASE 3: INTERACTIVE CONTENT GENERATION
    #
    #   Before the normal Q&A flow, we check if the user
    #   wants to GENERATE content (quiz, summary, article,
    #   Q&A, flashcards). If yes, we:
    #
    #   1. Use a special structured prompt that tells the
    #      LLM to output JSON (not plain text)
    #   2. Parse the JSON response
    #   3. Convert JSON → rich HTML via content_renderer
    #   4. Send the HTML to the frontend for rendering
    #
    #   This bypasses the normal tool loop because content
    #   generation doesn't need filter_rows or aggregate_data.
    #   It only needs the document context.
    # -------------------------------------------------------

    content_type = _detect_content_request(latest_question)

    if content_type is not None:
        logger.info(
            "[%s] Content generation mode: %s",
            request_id,
            content_type,
        )

        yield {
            "type": "status",
            "status": "thinking",
            "message": f"Generating {content_type}...",
            "elapsed_seconds": round(
                perf_counter() - start_time, 2,
            ),
            "request_id": request_id,
        }

        # ---------------------------------------------------
        # OPTIMIZATION 3: Trim context for content generation
        #
        #   When the user asks "create flashcards from page 2",
        #   the full context includes page-2 chunks PLUS
        #   semantic results from pages 1, 5, etc. Those
        #   extras are noise for content generation — they
        #   make the LLM read more text and generate slower.
        #
        #   If a specific page was requested, we rebuild a
        #   TRIMMED context with only that page's chunks.
        #   This means:
        #     - Less input tokens → faster processing
        #     - No off-topic content → better output quality
        #
        #   For non-page queries (e.g. "summarize the whole
        #   document"), we use the full context as before.
        # ---------------------------------------------------
        gen_context = context  # default: use full context

        requested_page = _detect_page_query(latest_question)
        if requested_page is not None:
            # Fetch only the page-specific chunks
            page_only = get_page_chunks(
                requested_page,
                source_filter=source_filter,
            )
            if page_only:
                # Merge overlapping chunks into clean text
                merged = _merge_overlapping_chunks(page_only)
                source_name = page_only[0].source
                gen_context = (
                    f"\n\nDOCUMENT CONTEXT:\n"
                    f"--- Page {requested_page} "
                    f"(Source: {source_name}) ---\n"
                    f"Content:\n{merged}\n"
                )
                logger.info(
                    "[%s] Trimmed context for content gen: "
                    "%d → %d chars (page %d only)",
                    request_id,
                    len(context),
                    len(gen_context),
                    requested_page,
                )

        # Build the structured generation prompt
        gen_prompt = _build_generation_prompt(
            content_type=content_type,
            question=latest_question,
            context=gen_context,
            source=source_filter or "Document",
        )

        # Send to LLM as a single-turn request
        # (no tool loop needed for content generation)
        gen_messages = [
            {"role": "system", "content": gen_prompt},
            {"role": "user", "content": latest_question},
        ]

        # ---------------------------------------------------
        # OPTIMIZATION 4: Show "generating" status with a
        # friendly label so the user knows the system is
        # actively working, not frozen.
        #
        #   Content generation can take 30-60s on free-tier
        #   models. Without this, the user stares at
        #   "Generating flashcards..." the whole time.
        #   This update changes the status to a "generating"
        #   state with the elapsed time, so the frontend
        #   shows the writing animation.
        # ---------------------------------------------------
        yield {
            "type": "status",
            "status": "generating",
            "message": (
                f"Writing {content_type} — this may take "
                f"30-60s on free models..."
            ),
            "elapsed_seconds": round(
                perf_counter() - start_time, 2,
            ),
            "request_id": request_id,
        }

        try:
            # Use non-streaming for content generation
            # because we need the FULL JSON to parse it
            llm_start = perf_counter()
            response = llm_chat(
                messages=gen_messages,
                tools=None,  # no tools for generation
            )
            raw_output = response.content or ""

            logger.info(
                "[%s] LLM generation took %.1fs",
                request_id,
                perf_counter() - llm_start,
            )

        except Exception as exc:
            logger.error(
                "[%s] Content generation LLM call failed: %s",
                request_id,
                exc,
            )
            yield {
                "type": "error",
                "message": f"Generation failed: {exc}",
                "error_type": type(exc).__name__,
                "elapsed_seconds": round(
                    perf_counter() - start_time, 2,
                ),
                "request_id": request_id,
            }
            return

        # Parse the JSON from LLM output
        parsed_data = _parse_llm_json(raw_output)

        if parsed_data is None:
            # JSON parsing failed — fall back to showing
            # raw LLM output as plain text
            logger.warning(
                "[%s] JSON parsing failed, showing raw "
                "output as plain text",
                request_id,
            )
            yield {
                "type": "status",
                "status": "first_token",
                "message": "Generating answer...",
                "first_token_seconds": round(
                    perf_counter() - start_time, 2,
                ),
                "request_id": request_id,
            }
            yield {"type": "token", "content": raw_output}
            total_time = perf_counter() - start_time
            yield {
                "type": "completed",
                "response_time_seconds": round(
                    total_time, 2,
                ),
                "assistant_content": raw_output,
                "request_id": request_id,
            }
            return

        # Render JSON → HTML using the template engine
        html_output = render_content(parsed_data)

        if html_output is None:
            # Rendering failed — show raw JSON
            logger.warning(
                "[%s] HTML rendering failed",
                request_id,
            )
            fallback_text = _json.dumps(
                parsed_data, indent=2,
            )
            yield {
                "type": "status",
                "status": "first_token",
                "message": "Generating answer...",
                "first_token_seconds": round(
                    perf_counter() - start_time, 2,
                ),
                "request_id": request_id,
            }
            yield {"type": "token", "content": fallback_text}
            total_time = perf_counter() - start_time
            yield {
                "type": "completed",
                "response_time_seconds": round(
                    total_time, 2,
                ),
                "assistant_content": fallback_text,
                "request_id": request_id,
            }
            return

        # SUCCESS — send HTML to frontend
        total_time = perf_counter() - start_time

        logger.info(
            "[%s] Content generated: %s (%d chars HTML, "
            "%.2fs)",
            request_id,
            content_type,
            len(html_output),
            total_time,
        )

        yield {
            "type": "status",
            "status": "first_token",
            "message": f"{content_type.title()} ready!",
            "first_token_seconds": round(
                perf_counter() - start_time, 2,
            ),
            "request_id": request_id,
        }

        # Send HTML as a special "html_content" event
        # The frontend will render this in an iframe
        yield {
            "type": "html_content",
            "html": html_output,
            "content_type": content_type,
        }

        # -------------------------------------------------
        # Completed event — includes HTML for persistence
        # -------------------------------------------------
        #
        # assistant_content is the SHORT label saved as the
        # message's "content" column (what shows in plain
        # text contexts like exports or search).
        #
        # html_content is the FULL interactive HTML saved
        # in the "html_content" column — the frontend uses
        # this to re-render the iframe on page reload.
        #
        # Previously we only sent assistant_content and the
        # HTML was lost after the stream ended. Now both
        # travel in the completed event so main.py can
        # persist them to the database.

        yield {
            "type": "completed",
            "response_time_seconds": round(
                total_time, 2,
            ),
            "assistant_content": f"[Generated {content_type}]",
            "html_content": html_output,
            "content_type": content_type,
            "request_id": request_id,
        }

        # -------------------------------------------------
        # Store content generation result in cache
        # -------------------------------------------------
        #
        # WHY CACHE INTERACTIVE CONTENT?
        #
        #   Content generation (flashcards, quizzes, etc.)
        #   is the SLOWEST operation — 30-60s on free-tier
        #   models. Caching it gives the biggest speedup.
        #
        #   We store both the text label ("[Generated
        #   flashcards]") AND the full HTML, so on cache
        #   hit the frontend gets the complete iframe
        #   content instantly.

        if query_embedding is not None:
            try:
                store_in_cache(
                    question=latest_question,
                    query_embedding=query_embedding,
                    answer=f"[Generated {content_type}]",
                    source_filter=source_filter,
                    content_type=content_type,
                    html_content=html_output,
                )
            except Exception as cache_exc:
                # Cache store failure is non-fatal — the
                # answer was already sent to the user.
                logger.warning(
                    "[%s] Failed to cache content: %s",
                    request_id,
                    cache_exc,
                )

        # -------------------------------------------------
        # Generate follow-up suggestions for content
        # -------------------------------------------------
        try:
            suggestions = generate_followup_suggestions(
                question=latest_question,
                answer=f"[Generated {content_type}]",
                content_type=content_type,
            )
            if suggestions:
                yield {
                    "type": "suggestions",
                    "questions": suggestions,
                    "request_id": request_id,
                }
        except Exception as sug_exc:
            logger.warning(
                "[%s] Suggestion generation failed: %s",
                request_id,
                sug_exc,
            )

        return

    # -------------------------------------------------------
    # STEP 2: Build system prompt (normal Q&A mode)
    # -------------------------------------------------------

    system_content = SYSTEM_PROMPT + context

    if source_filter:
        # Single-document mode
        system_content += (
            f"\nYou are answering questions about: "
            f"{source_filter}\n"
        )
    else:
        # Multi-document mode (Phase 4) — tell LLM to
        # cite sources when answering across documents
        system_content += (
            "\nYou are searching across ALL uploaded "
            "documents. When answering, always cite which "
            "document (Source filename) each piece of "
            "information comes from.\n"
        )

    # Pick tools based on document type AND question.
    # Simple lookup questions ("who is the teacher?")
    # get NO tools — the model answers from context.
    # Filtering/counting questions get tools.
    # In multi-doc mode (source_filter=None), no tools.
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

                # CHANGED: was ollama.chat(), now llm_chat()
                #   Routes to Ollama or OpenRouter based on
                #   LLM_PROVIDER env var. Same unified
                #   response format either way.
                response = llm_chat(
                    messages=working_messages,
                    tools=active_tools,
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

            # CHANGED: was response.message.tool_calls
            #   Now uses unified LLMResponse.tool_calls
            if not response.tool_calls:
                # LLM decided to answer — but we got
                # the answer non-streamed. Emit it as
                # one big token block.

                # CHANGED: was response.message.content
                content = response.content or ""

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

                # Cache the tool-assisted answer
                if query_embedding is not None:
                    try:
                        store_in_cache(
                            question=latest_question,
                            query_embedding=query_embedding,
                            answer=content,
                            source_filter=source_filter,
                        )
                    except Exception as cache_exc:
                        logger.warning(
                            "[%s] Failed to cache "
                            "tool answer: %s",
                            request_id,
                            cache_exc,
                        )

                # Generate follow-up suggestions
                try:
                    suggestions = generate_followup_suggestions(
                        question=latest_question,
                        answer=content,
                    )
                    if suggestions:
                        yield {
                            "type": "suggestions",
                            "questions": suggestions,
                            "request_id": request_id,
                        }
                except Exception as sug_exc:
                    logger.warning(
                        "[%s] Suggestion generation "
                        "failed: %s",
                        request_id,
                        sug_exc,
                    )

                return

            # More tool calls — execute and loop
            # CHANGED: was response.message (Ollama-specific)
            #   Now uses make_assistant_msg() to build a
            #   provider-neutral message dict.
            working_messages.append(
                make_assistant_msg(
                    content="",
                    tool_calls=response.tool_calls,
                )
            )

            # CHANGED: was response.message.tool_calls
            for tc in response.tool_calls:
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

            # CHANGED: was ollama.chat(stream=True)
            #   Now llm_stream() returns LLMChunk objects.
            #   For Ollama: wraps ollama.chat(stream=True)
            #   For OpenRouter: accumulates streaming
            #     tool calls from multiple delta chunks.
            stream = llm_stream(
                messages=working_messages,
                tools=active_tools,
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

            # CHANGED: was chunk.message.content
            #   LLMChunk uses .content directly
            if chunk.content:

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

                # CHANGED: was chunk.message.content
                full_content_parts.append(
                    chunk.content,
                )

                yield {
                    "type": "token",
                    "content": chunk.content,
                }

            # CHANGED: was chunk.message.tool_calls
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls

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

            streamed_answer = "".join(full_content_parts)

            yield {
                "type": "completed",
                "response_time_seconds": round(
                    total_time, 2,
                ),
                "assistant_content": streamed_answer,
                "request_id": request_id,
            }

            # -----------------------------------------
            # Store streamed answer in cache
            # -----------------------------------------
            #
            # This is the most common path — a normal
            # Q&A answer streamed token by token. We
            # join all the token parts into the full
            # answer text and cache it for next time.

            if query_embedding is not None:
                try:
                    store_in_cache(
                        question=latest_question,
                        query_embedding=query_embedding,
                        answer=streamed_answer,
                        source_filter=source_filter,
                    )
                except Exception as cache_exc:
                    logger.warning(
                        "[%s] Failed to cache answer: %s",
                        request_id,
                        cache_exc,
                    )

            # Generate follow-up suggestions
            try:
                suggestions = generate_followup_suggestions(
                    question=latest_question,
                    answer=streamed_answer,
                )
                if suggestions:
                    yield {
                        "type": "suggestions",
                        "questions": suggestions,
                        "request_id": request_id,
                    }
            except Exception as sug_exc:
                logger.warning(
                    "[%s] Suggestion generation failed: %s",
                    request_id,
                    sug_exc,
                )

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

        # CHANGED: was Message(role="assistant", ...)
        #   Ollama's Message class is Ollama-specific.
        #   make_assistant_msg() builds a provider-neutral
        #   dict that works with both Ollama and OpenRouter.
        working_messages.append(
            make_assistant_msg(
                content="",
                tool_calls=tool_calls,
            )
        )

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

        # CHANGED: was ollama.chat(stream=True)
        #   No tools passed → LLM must give a text answer.
        final_stream = llm_stream(
            messages=working_messages,
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

        # CHANGED: was chunk.message.content
        if not chunk.content:
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

        # CHANGED: was chunk.message.content
        final_parts.append(chunk.content)

        yield {
            "type": "token",
            "content": chunk.content,
        }

    total_time = perf_counter() - start_time

    logger.info(
        "[%s] Completed (max rounds, tools=[%s]) "
        "in %.2fs",
        request_id,
        ", ".join(tools_used),
        total_time,
    )

    final_answer = "".join(final_parts)

    yield {
        "type": "completed",
        "response_time_seconds": round(
            total_time, 2,
        ),
        "assistant_content": final_answer,
        "tools_used": tools_used,
        "request_id": request_id,
    }

    # Cache the max-rounds answer
    if query_embedding is not None:
        try:
            store_in_cache(
                question=latest_question,
                query_embedding=query_embedding,
                answer=final_answer,
                source_filter=source_filter,
            )
        except Exception as cache_exc:
            logger.warning(
                "[%s] Failed to cache final answer: %s",
                request_id,
                cache_exc,
            )

    # Generate follow-up suggestions
    try:
        suggestions = generate_followup_suggestions(
            question=latest_question,
            answer=final_answer,
        )
        if suggestions:
            yield {
                "type": "suggestions",
                "questions": suggestions,
                "request_id": request_id,
            }
    except Exception as sug_exc:
        logger.warning(
            "[%s] Suggestion generation failed: %s",
            request_id,
            sug_exc,
        )


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

    # CHANGED: was tool_call.function.name / .arguments
    #   Ollama nests under .function, but our unified
    #   ToolCall dataclass puts them at top level.
    tool_name = tool_call.name
    arguments = tool_call.arguments

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

        # CHANGED: was hardcoded {"role": "tool", "tool_name": ...}
        #   make_tool_msg() builds the right format for
        #   whichever provider is active (Ollama uses
        #   tool_name, OpenRouter uses tool_call_id).
        working_messages.append(
            make_tool_msg(
                tool_name=tool_name,
                tool_call_id=getattr(tool_call, "id", ""),
                content=f"Error: Unknown tool '{tool_name}'.",
            )
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

    # CHANGED: was hardcoded {"role": "tool", "tool_name": ...}
    #   make_tool_msg() ensures the tool result message
    #   has the right fields for the active provider.
    working_messages.append(
        make_tool_msg(
            tool_name=tool_name,
            tool_call_id=getattr(tool_call, "id", ""),
            content=str(result),
        )
    )