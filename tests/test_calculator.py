"""Tests for the calculator tool.

Why test the calculator separately?
  The calculator is a pure function — it takes inputs
  and returns outputs with no side effects. This makes
  it perfect for unit testing. We can test every edge
  case quickly without needing Ollama running.

How to run:
  cd E:\\ai-document-agent
  uv run pytest tests/ -v
"""

import pytest

from ai_document_agent.agent import calculator


# ---------------------------------------------------------
# Basic operations
# ---------------------------------------------------------

class TestBasicOperations:
    """Test that each operation produces correct results."""

    def test_add(self):
        assert calculator("add", 2, 3) == 5

    def test_subtract(self):
        assert calculator("subtract", 10, 3) == 7

    def test_multiply(self):
        assert calculator("multiply", 4, 5) == 20

    def test_divide(self):
        assert calculator("divide", 10, 2) == 5.0

    def test_divide_with_decimals(self):
        assert calculator("divide", 7, 2) == 3.5

    def test_negative_numbers(self):
        assert calculator("add", -5, 3) == -2

    def test_zero(self):
        assert calculator("multiply", 100, 0) == 0

    def test_large_numbers(self):
        assert calculator("add", 1_000_000, 2_000_000) == 3_000_000

    def test_float_precision(self):
        result = calculator("add", 0.1, 0.2)
        assert abs(result - 0.3) < 1e-10


# ---------------------------------------------------------
# Synonym handling
# ---------------------------------------------------------

class TestSynonyms:
    """Test that all operation synonyms work.

    Why test synonyms?
      The LLM might send any of these. If a synonym
      breaks, the user sees an error for a valid question.
    """

    # Addition synonyms
    def test_sum(self):
        assert calculator("sum", 1, 2) == 3

    def test_plus(self):
        assert calculator("plus", 1, 2) == 3

    def test_plus_symbol(self):
        assert calculator("+", 1, 2) == 3

    def test_addition(self):
        assert calculator("addition", 1, 2) == 3

    # Subtraction synonyms
    def test_sub(self):
        assert calculator("sub", 5, 2) == 3

    def test_minus(self):
        assert calculator("minus", 5, 2) == 3

    def test_minus_symbol(self):
        assert calculator("-", 5, 2) == 3

    def test_subtraction(self):
        assert calculator("subtraction", 5, 2) == 3

    # Multiplication synonyms
    def test_mul(self):
        assert calculator("mul", 3, 4) == 12

    def test_times(self):
        assert calculator("times", 3, 4) == 12

    def test_star_symbol(self):
        assert calculator("*", 3, 4) == 12

    def test_x_symbol(self):
        assert calculator("x", 3, 4) == 12

    def test_multiplication(self):
        assert calculator("multiplication", 3, 4) == 12

    # Division synonyms
    def test_div(self):
        assert calculator("div", 10, 2) == 5

    def test_slash_symbol(self):
        assert calculator("/", 10, 2) == 5

    def test_division(self):
        assert calculator("division", 10, 2) == 5


# ---------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------

class TestCaseInsensitivity:
    """Test that operation names work in any case.

    Why?
      LLMs sometimes capitalize randomly.
      "ADD", "Add", "add" should all work.
    """

    def test_uppercase(self):
        assert calculator("ADD", 1, 2) == 3

    def test_mixed_case(self):
        assert calculator("Multiply", 3, 4) == 12

    def test_with_whitespace(self):
        assert calculator("  add  ", 1, 2) == 3


# ---------------------------------------------------------
# Error handling
# ---------------------------------------------------------

class TestErrors:
    """Test that errors are raised correctly."""

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="divide by zero"):
            calculator("divide", 10, 0)

    def test_unknown_operation(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            calculator("modulo", 10, 3)

    def test_unknown_operation_message_is_helpful(self):
        """Error message should tell the user what's supported."""
        with pytest.raises(ValueError, match="Supported"):
            calculator("power", 2, 8)
