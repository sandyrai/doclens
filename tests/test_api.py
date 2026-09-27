"""API tests for the FastAPI app.

These use FastAPI's TestClient (no real server) and mock
the agent, so no LLM or Ollama is needed. Conversation
history is stored in SQLite; conftest.py points it at a
temporary database.

How to run:
  uv run pytest tests/ -v
"""

import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ai_document_agent.database import get_session_messages
from ai_document_agent.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _sid() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------
# Health and frontend
# ---------------------------------------------------------

class TestHealth:
    def test_health_reports_status(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] in {"healthy", "degraded"}

    def test_every_response_has_request_id(self, client):
        res = client.get("/health")
        assert res.headers.get("X-Request-ID", "").startswith("req_")

    def test_root_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]


# ---------------------------------------------------------
# POST /chat
# ---------------------------------------------------------

class TestChat:
    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_returns_answer(self, mock_agent, client):
        mock_agent.return_value = "The report has 12 pages."
        sid = _sid()
        res = client.post(
            "/chat",
            json={"question": "How many pages?", "session_id": sid},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["answer"] == "The report has 12 pages."
        assert body["session_id"] == sid
        assert body["request_id"].startswith("req_")

    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_saves_both_messages(self, mock_agent, client):
        mock_agent.return_value = "Answer one"
        sid = _sid()
        client.post("/chat", json={"question": "Q1", "session_id": sid})
        history = get_session_messages(sid)
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "Q1"
        assert history[1]["content"] == "Answer one"

    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_conversation_history_is_passed_to_agent(
        self, mock_agent, client,
    ):
        mock_agent.return_value = "ok"
        sid = _sid()
        client.post("/chat", json={"question": "First", "session_id": sid})
        client.post("/chat", json={"question": "Second", "session_id": sid})
        messages = mock_agent.call_args.args[0]
        contents = [m["content"] for m in messages]
        assert "First" in contents and "Second" in contents

    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_sessions_are_isolated(self, mock_agent, client):
        mock_agent.return_value = "ok"
        alice, bob = _sid(), _sid()
        client.post("/chat", json={"question": "Alice Q", "session_id": alice})
        client.post("/chat", json={"question": "Bob Q", "session_id": bob})
        assert get_session_messages(alice)[0]["content"] == "Alice Q"
        assert get_session_messages(bob)[0]["content"] == "Bob Q"
        assert len(get_session_messages(alice)) == 2

    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_agent_failure_returns_500_with_request_id(
        self, mock_agent, client,
    ):
        mock_agent.side_effect = RuntimeError("LLM unreachable")
        res = client.post(
            "/chat", json={"question": "Q", "session_id": _sid()},
        )
        assert res.status_code == 500
        body = res.json()
        assert body["error_type"] == "RuntimeError"
        assert body["request_id"].startswith("req_")

    def test_question_is_required(self, client):
        res = client.post("/chat", json={"session_id": _sid()})
        assert res.status_code == 422

    @patch("ai_document_agent.routes.chat.check_and_increment_question")
    def test_rate_limited_returns_429(self, mock_check, client):
        mock_check.return_value = {
            "allowed": False,
            "used": 15,
            "limit": 15,
            "message": "Daily question limit reached.",
        }
        res = client.post(
            "/chat", json={"question": "Q", "session_id": _sid()},
        )
        assert res.status_code == 429
        assert res.json()["usage"]["questions_limit"] == 15


# ---------------------------------------------------------
# POST /chat/stream (NDJSON)
# ---------------------------------------------------------

def _ndjson(res) -> list[dict]:
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


class TestChatStream:
    @patch("ai_document_agent.routes.chat.stream_agent")
    def test_streams_tokens_then_completed(self, mock_stream, client):
        mock_stream.return_value = iter([
            {"type": "token", "content": "Hel"},
            {"type": "token", "content": "lo"},
            {"type": "completed", "assistant_content": "Hello"},
        ])
        sid = _sid()
        res = client.post(
            "/chat/stream", json={"question": "Hi", "session_id": sid},
        )
        assert res.status_code == 200
        assert "ndjson" in res.headers["content-type"]
        events = _ndjson(res)
        assert [e["type"] for e in events][-1] == "completed"
        assert "".join(
            e["content"] for e in events if e["type"] == "token"
        ) == "Hello"

    @patch("ai_document_agent.routes.chat.stream_agent")
    def test_completed_answer_is_saved(self, mock_stream, client):
        mock_stream.return_value = iter([
            {"type": "completed", "assistant_content": "Saved answer"},
        ])
        sid = _sid()
        client.post("/chat/stream", json={"question": "Q", "session_id": sid})
        history = get_session_messages(sid)
        assert history[-1]["role"] == "assistant"
        assert history[-1]["content"] == "Saved answer"

    @patch("ai_document_agent.routes.chat.stream_agent")
    def test_exception_becomes_error_event(self, mock_stream, client):
        def boom(*args, **kwargs):
            yield {"type": "token", "content": "partial"}
            raise RuntimeError("stream broke")

        mock_stream.side_effect = boom
        res = client.post(
            "/chat/stream", json={"question": "Q", "session_id": _sid()},
        )
        events = _ndjson(res)
        assert events[-1]["type"] == "error"
        assert events[-1]["request_id"].startswith("req_")


# ---------------------------------------------------------
# Client IP handling (rate-limit spoofing protection)
# ---------------------------------------------------------

class TestClientIp:
    def test_forwarded_header_only_trusted_when_enabled(self):
        from starlette.requests import Request

        from ai_document_agent import shared

        scope = {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"1.2.3.4")],
            "client": ("10.0.0.9", 1234),
        }
        with patch.object(shared, "TRUST_PROXY_HEADERS", False):
            assert shared.get_client_ip(Request(scope)) == "10.0.0.9"
        with patch.object(shared, "TRUST_PROXY_HEADERS", True):
            assert shared.get_client_ip(Request(scope)) == "1.2.3.4"
