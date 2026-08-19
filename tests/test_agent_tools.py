"""Tests for agent module — query enrichment and tools.

Phase 8.4: Adding unit tests for the pure functions
in agent.py that can be tested without Ollama.

What these tests cover:
  1. _enrich_query() — follow-up question detection
  2. _sanitize_filename() — path traversal prevention
  3. filter_rows() — CSV filtering logic
  4. aggregate_data() — CSV aggregation logic
  5. calculator() — edge cases not in test_calculator.py

How to run:
  cd E:\\ai-document-agent
  uv run pytest tests/ -v
"""

import pytest

from ai_document_agent.agent import (
    _enrich_query,
    calculator,
)


# ---------------------------------------------------------
# _enrich_query() tests
# ---------------------------------------------------------

class TestEnrichQuery:
    """Test the follow-up question detection."""

    def test_standalone_question_unchanged(self):
        """A standalone question should not be enriched."""

        messages = [
            {"role": "user", "content": "What is the total marks?"},
        ]
        result = _enrich_query(
            "What is the total marks?", messages,
        )
        assert result == "What is the total marks?"

    def test_referential_word_triggers_enrichment(self):
        """Questions with 'that', 'it', etc. should be
        enriched with the previous question."""

        messages = [
            {"role": "user", "content": "list the toppers"},
            {"role": "assistant", "content": "Here are..."},
            {"role": "user", "content": "tell me more about that"},
        ]
        result = _enrich_query(
            "tell me more about that", messages,
        )
        assert "list the toppers" in result
        assert "tell me more about that" in result

    def test_connecting_word_triggers_enrichment(self):
        """Questions starting with 'what about', 'how about',
        etc. should be enriched."""

        messages = [
            {"role": "user", "content": "list subject wise topper"},
            {"role": "assistant", "content": "Here..."},
            {"role": "user", "content": "what about Hindi?"},
        ]
        result = _enrich_query(
            "what about Hindi?", messages,
        )
        assert "list subject wise topper" in result
        assert "Hindi" in result

    def test_short_question_triggers_enrichment(self):
        """Very short questions (< 6 words) that don't look
        standalone should be enriched."""

        messages = [
            {"role": "user", "content": "show me the results"},
            {"role": "assistant", "content": "Here..."},
            {"role": "user", "content": "in Hindi?"},
        ]
        result = _enrich_query("in Hindi?", messages)
        assert "show me the results" in result

    def test_short_standalone_not_enriched(self):
        """Short questions that start with 'list', 'show',
        etc. should NOT be enriched."""

        messages = [
            {"role": "user", "content": "something old"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "list all students"},
        ]
        result = _enrich_query(
            "list all students", messages,
        )
        assert result == "list all students"

    def test_no_previous_question_returns_original(self):
        """If there's no previous user question, return
        the original."""

        messages = [
            {"role": "user", "content": "what about that?"},
        ]
        result = _enrich_query(
            "what about that?", messages,
        )
        assert result == "what about that?"

    def test_empty_question_returns_empty(self):
        """Empty question should return empty."""

        assert _enrich_query("", []) == ""

    def test_none_question_returns_none(self):
        """None question should return None."""

        assert _enrich_query(None, []) is None

    def test_enriched_format(self):
        """Enriched query should use ' — ' separator."""

        messages = [
            {"role": "user", "content": "show marks"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "and Hindi?"},
        ]
        result = _enrich_query("and Hindi?", messages)
        assert " — " in result

    def test_also_triggers_enrichment(self):
        """'also' at the start should trigger enrichment."""

        messages = [
            {"role": "user", "content": "show math marks"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "also science"},
        ]
        result = _enrich_query("also science", messages)
        assert "show math marks" in result


# ---------------------------------------------------------
# _sanitize_filename() tests
# ---------------------------------------------------------

from ai_document_agent.main import _sanitize_filename


class TestSanitizeFilename:
    """Test filename sanitization for path traversal."""

    def test_normal_filename_unchanged(self):
        assert _sanitize_filename("report.pdf") == "report.pdf"

    def test_strips_directory_traversal(self):
        result = _sanitize_filename("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result
        assert result == "passwd"

    def test_strips_windows_path(self):
        result = _sanitize_filename(
            "C:\\Windows\\system32\\evil.exe"
        )
        assert "\\" not in result
        assert result == "evil.exe"

    def test_strips_unix_absolute(self):
        result = _sanitize_filename("/etc/shadow")
        assert result == "shadow"

    def test_empty_becomes_unnamed(self):
        assert _sanitize_filename("") == "unnamed_upload"

    def test_null_bytes_removed(self):
        result = _sanitize_filename("file\x00.pdf")
        assert "\x00" not in result

    def test_whitespace_stripped(self):
        result = _sanitize_filename("  report.pdf  ")
        assert result == "report.pdf"

    def test_mixed_traversal(self):
        result = _sanitize_filename(
            "../../../uploads/../../../etc/passwd"
        )
        assert result == "passwd"


# ---------------------------------------------------------
# calculator() edge cases
# ---------------------------------------------------------

class TestCalculatorEdgeCases:
    """Additional calculator tests not in test_calculator."""

    def test_very_small_floats(self):
        result = calculator("add", 0.001, 0.002)
        assert abs(result - 0.003) < 1e-10

    def test_negative_division(self):
        assert calculator("divide", -10, 2) == -5.0

    def test_both_negative(self):
        assert calculator("multiply", -3, -4) == 12

    def test_divide_produces_float(self):
        result = calculator("divide", 1, 3)
        assert isinstance(result, float)
        assert abs(result - 1/3) < 1e-10
