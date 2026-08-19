"""Tests for the FastAPI endpoints.

Why test the API?
  The API is the contract between your frontend and
  backend. If the response shape changes accidentally,
  the browser UI breaks. Tests catch that before you
  notice it manually.

How these work:
  FastAPI has a built-in TestClient that simulates HTTP
  requests without starting a real server. We also mock
  Ollama so tests run instantly without needing the model.

How to run:
  cd E:\\ai-document-agent
  uv run pytest tests/ -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_document_agent.main import app, sessions


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------
#
# What is a fixture?
#   A reusable piece of test setup. @pytest.fixture
#   runs before each test that uses it.
#
# Why clear sessions?
#   Tests should be independent — one test's session
#   data shouldn't leak into another test.

@pytest.fixture(autouse=True)
def clear_sessions():
    """Reset session store before each test."""
    sessions.clear()
    yield
    sessions.clear()


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------
# Helper to mock Ollama responses
# ---------------------------------------------------------

def make_mock_response(content="Hello!", tool_calls=None):
    """Create a fake Ollama chat response.

    Why mock Ollama?
      1. Tests run without Ollama/Qwen installed.
      2. Tests run in milliseconds, not 30 seconds.
      3. Tests are deterministic — same input, same output.
      4. We can simulate errors that are hard to trigger
         with a real model.
    """

    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    response = MagicMock()
    response.message = message

    return response


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

class TestHealth:

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

class TestRoot:

    def test_root_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------
# POST /chat
# ---------------------------------------------------------

class TestChatEndpoint:

    @patch("ai_document_agent.main.ask_agent")
    def test_chat_returns_answer(
        self, mock_ask, client
    ):
        mock_ask.return_value = "42"

        response = client.post(
            "/chat",
            json={
                "question": "What is the answer?",
                "session_id": "test-session-1",
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["answer"] == "42"
        assert data["session_id"] == "test-session-1"
        assert "request_id" in data

    @patch("ai_document_agent.main.ask_agent")
    def test_chat_saves_to_session(
        self, mock_ask, client
    ):
        """After a successful chat, both user and
        assistant messages should be in the session."""

        mock_ask.return_value = "Paris"

        client.post(
            "/chat",
            json={
                "question": "What is the capital?",
                "session_id": "sess-1",
            },
        )

        history = sessions["sess-1"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Paris"

    @patch("ai_document_agent.main.ask_agent")
    def test_chat_error_returns_500(
        self, mock_ask, client
    ):
        """If the agent crashes, the API should return
        a structured error, not a raw exception."""

        mock_ask.side_effect = ConnectionError(
            "Ollama not running"
        )

        response = client.post(
            "/chat",
            json={
                "question": "Hello",
                "session_id": "sess-err",
            },
        )

        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert "Ollama" in data["error"]
        assert "request_id" in data

    @patch("ai_document_agent.main.ask_agent")
    def test_chat_error_cleans_session(
        self, mock_ask, client
    ):
        """If the agent crashes, the user message should
        be removed from session so history stays clean."""

        mock_ask.side_effect = Exception("boom")

        client.post(
            "/chat",
            json={
                "question": "Hello",
                "session_id": "sess-clean",
            },
        )

        # Session should be empty — the dangling
        # user message was removed
        history = sessions.get("sess-clean", [])
        assert len(history) == 0

    def test_chat_requires_question(self, client):
        """Missing 'question' field should return 422."""

        response = client.post(
            "/chat",
            json={"session_id": "test"},
        )

        assert response.status_code == 422


# ---------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------

class TestChatStreamEndpoint:

    @patch("ai_document_agent.main.stream_agent")
    def test_stream_returns_ndjson(
        self, mock_stream, client
    ):
        """Stream endpoint should return NDJSON events."""

        mock_stream.return_value = iter(
            [
                {
                    "type": "status",
                    "status": "thinking",
                    "message": "...",
                    "elapsed_seconds": 0,
                    "request_id": "req_test",
                },
                {
                    "type": "token",
                    "content": "Hi there!",
                },
                {
                    "type": "completed",
                    "response_time_seconds": 1.5,
                    "assistant_content": "Hi there!",
                    "request_id": "req_test",
                },
            ]
        )

        response = client.post(
            "/chat/stream",
            json={
                "question": "Hello",
                "session_id": "stream-1",
            },
        )

        assert response.status_code == 200
        assert "ndjson" in response.headers[
            "content-type"
        ]

        # Parse all NDJSON events
        lines = response.text.strip().split("\n")
        events = [
            json.loads(line) for line in lines
        ]

        # Should have 3 events
        assert len(events) == 3
        assert events[0]["type"] == "status"
        assert events[1]["type"] == "token"
        assert events[2]["type"] == "completed"

    @patch("ai_document_agent.main.stream_agent")
    def test_stream_saves_to_session(
        self, mock_stream, client
    ):
        """After a successful stream, the assistant
        reply should be saved to session history."""

        mock_stream.return_value = iter(
            [
                {
                    "type": "completed",
                    "response_time_seconds": 1.0,
                    "assistant_content": "Hello!",
                    "request_id": "req_test",
                },
            ]
        )

        client.post(
            "/chat/stream",
            json={
                "question": "Hi",
                "session_id": "stream-save",
            },
        )

        history = sessions["stream-save"]

        # User message + assistant reply
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello!"

    @patch("ai_document_agent.main.stream_agent")
    def test_stream_error_cleans_session(
        self, mock_stream, client
    ):
        """If stream_agent raises, the dangling user
        message should be cleaned from history."""

        mock_stream.side_effect = Exception(
            "Ollama crashed"
        )

        response = client.post(
            "/chat/stream",
            json={
                "question": "Hello",
                "session_id": "stream-err",
            },
        )

        # The error event should be in the response
        lines = response.text.strip().split("\n")
        last_event = json.loads(lines[-1])
        assert last_event["type"] == "error"

        # Session should be clean
        history = sessions.get("stream-err", [])
        assert len(history) == 0


# ---------------------------------------------------------
# Session management
# ---------------------------------------------------------

class TestSessions:

    @patch("ai_document_agent.main.ask_agent")
    def test_auto_generates_session_id(
        self, mock_ask, client
    ):
        """If no session_id is provided, one should
        be auto-generated."""

        mock_ask.return_value = "ok"

        response = client.post(
            "/chat",
            json={"question": "Hello"},
        )

        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    @patch("ai_document_agent.main.ask_agent")
    def test_conversation_continuity(
        self, mock_ask, client
    ):
        """Multiple requests with the same session_id
        should accumulate history."""

        mock_ask.return_value = "answer"

        for i in range(3):
            client.post(
                "/chat",
                json={
                    "question": f"Q{i}",
                    "session_id": "persist",
                },
            )

        # 3 questions × 2 messages each = 6
        history = sessions["persist"]
        assert len(history) == 6

    @patch("ai_document_agent.main.ask_agent")
    def test_different_sessions_are_isolated(
        self, mock_ask, client
    ):
        """Different session_ids should have separate
        conversation histories."""

        mock_ask.return_value = "answer"

        client.post(
            "/chat",
            json={
                "question": "Q1",
                "session_id": "alice",
            },
        )

        client.post(
            "/chat",
            json={
                "question": "Q2",
                "session_id": "bob",
            },
        )

        assert len(sessions["alice"]) == 2
        assert len(sessions["bob"]) == 2
        assert (
            sessions["alice"][0]["content"] !=
            sessions["bob"][0]["content"]
        )
