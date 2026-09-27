"""Tests for agent module — query enrichment and tools.

Phase 8.4: Adding unit tests for the pure functions
in agent.py that can be tested without Ollama.

What these tests cover:
  1. _enrich_query() — follow-up question detection
  2. _sanitize_filename() — path traversal prevention
  3. filter_rows() — CSV filtering logic
  4. aggregate_data() — CSV aggregation logic

How to run:
  uv run pytest tests/ -v
"""

import pytest

from ai_document_agent.agent import (
    _enrich_query,
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

from ai_document_agent.shared import (
    sanitize_filename as _sanitize_filename,
)


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

    def test_dot_dot_alone_is_rejected(self):
        assert _sanitize_filename("..") == "unnamed_upload"
        assert _sanitize_filename("../..") == "unnamed_upload"

    def test_mixed_separators(self):
        assert _sanitize_filename("a\\b/../c\\report.pdf") == "report.pdf"

    def test_mixed_traversal(self):
        result = _sanitize_filename(
            "../../../uploads/../../../etc/passwd"
        )
        assert result == "passwd"


# ---------------------------------------------------------
# Agent tools: filter_rows() and aggregate_data()
# ---------------------------------------------------------
#
# These are the functions the LLM can call (tool calling)
# when a question needs exact answers from tabular data.

from unittest.mock import patch  # noqa: E402

from ai_document_agent import agent as _agent  # noqa: E402

SAMPLE_CSV = (
    "name,department,salary\n"
    "Asha,Engineering,120000\n"
    "Ravi,Engineering,95000\n"
    "Meera,Sales,70000\n"
    "Kabir,Sales,85000\n"
)


@pytest.fixture
def csv_source(tmp_path):
    path = tmp_path / "staff.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    with patch.object(_agent, "_find_csv_for_source", return_value=path):
        yield "staff.csv"


class TestFilterRows:
    def test_exact_match_is_case_insensitive(self, csv_source):
        out = _agent.filter_rows("department", "sales", csv_source)
        assert "2 rows" in out
        assert "Meera" in out and "Kabir" in out
        assert "Asha" not in out

    def test_numeric_comparison(self, csv_source):
        out = _agent.filter_rows("salary", ">90000", csv_source)
        assert "Asha" in out and "Ravi" in out
        assert "Meera" not in out

    def test_unknown_column_lists_available_columns(self, csv_source):
        out = _agent.filter_rows("age", "30", csv_source)
        assert out.startswith("Error")
        assert "salary" in out

    def test_no_match_reports_total_rows(self, csv_source):
        out = _agent.filter_rows("department", "HR", csv_source)
        assert "No rows found" in out and "4" in out

    def test_non_tabular_document_returns_guidance(self):
        with patch.object(_agent, "_find_csv_for_source", return_value=None):
            out = _agent.filter_rows("x", "y", "notes.txt")
        assert "only works on tabular data" in out


class TestAggregateData:
    def test_average(self, csv_source):
        out = _agent.aggregate_data("salary", "average", None, csv_source)
        assert "92500" in out.replace(",", "")

    def test_sum_grouped_by_department(self, csv_source):
        out = _agent.aggregate_data(
            "salary", "sum", "department", csv_source,
        ).replace(",", "")
        assert "Engineering" in out and "215000" in out
        assert "Sales" in out and "155000" in out

    def test_bad_group_by_column(self, csv_source):
        out = _agent.aggregate_data("salary", "sum", "team", csv_source)
        assert out.startswith("Error")


class TestToolSelectionForSmallPdfTables:
    """Small PDF tables get tools only for arithmetic questions."""

    def _tools(self, question, n_rows=5):
        rows = [{"Region": f"R{i}", "Revenue": str(i)} for i in range(n_rows)]
        with patch.object(_agent, "find_extracted_table_csv", return_value="t.csv"), \
             patch.object(_agent, "load_csv", return_value=(["Region", "Revenue"], rows)):
            return _agent._get_tools_for_source("report.pdf", question)

    def test_arithmetic_question_gets_tools(self):
        assert self._tools("What is the average revenue?") == _agent.CSV_TOOLS

    def test_lookup_question_gets_no_tools(self):
        assert self._tools("Who is the chairman?") == _agent.NO_TOOLS
