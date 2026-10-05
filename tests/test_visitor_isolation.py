"""Visitor isolation tests (no database or LLM needed).

With VISITOR_ISOLATION on, each browser gets its own
visitor cookie, and sessions, uploads, cached answers and
suggestions are private to that visitor. These tests use
two TestClients (two cookie jars) as two visitors.

How to run:
  uv run pytest tests/test_visitor_isolation.py -v
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ai_document_agent import tenancy
from ai_document_agent.main import app
from ai_document_agent.shared import upload_tasks


@pytest.fixture
def isolation(monkeypatch):
    monkeypatch.setattr(tenancy, "VISITOR_ISOLATION", True)


@pytest.fixture
def alice(isolation):
    with TestClient(app) as c:
        c.get("/")  # first request issues the visitor cookie
        yield c


@pytest.fixture
def bob(isolation):
    with TestClient(app) as c:
        c.get("/")
        yield c


def _sid() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------
# Cookie / visitor identity
# ---------------------------------------------------------

class TestVisitorCookie:
    def test_first_request_sets_httponly_cookie(self, isolation):
        with TestClient(app) as c:
            res = c.get("/health")
        cookie = res.headers["set-cookie"]
        assert cookie.startswith(f"{tenancy.COOKIE_NAME}=")
        assert "HttpOnly" in cookie
        assert "SameSite=Lax" in cookie

    def test_cookie_is_not_reissued(self, alice):
        res = alice.get("/health")
        assert "set-cookie" not in res.headers

    def test_secure_flag_behind_https_proxy(self, isolation, monkeypatch):
        mw = next(
            m for m in app.user_middleware
            if m.cls is tenancy.VisitorMiddleware
        )
        monkeypatch.setitem(mw.kwargs, "trust_proxy_headers", True)
        app.middleware_stack = None  # rebuild with the new kwargs
        try:
            with TestClient(app) as c:
                res = c.get(
                    "/health",
                    headers={"X-Forwarded-Proto": "https"},
                )
            assert "Secure" in res.headers["set-cookie"]
        finally:
            app.middleware_stack = None

    def test_malformed_cookie_gets_a_fresh_identity(self, isolation):
        with TestClient(app) as c:
            c.cookies.set(tenancy.COOKIE_NAME, "../../etc/passwd")
            res = c.get("/health")
        assert "set-cookie" in res.headers

    def test_off_by_default_no_cookie(self, monkeypatch):
        monkeypatch.setattr(tenancy, "VISITOR_ISOLATION", False)
        with TestClient(app) as c:
            res = c.get("/health")
        assert "set-cookie" not in res.headers


# ---------------------------------------------------------
# Scoped keys and folders
# ---------------------------------------------------------

class TestScoping:
    def test_default_visitor_keys_unchanged(self):
        assert tenancy.scoped_key("report.pdf") == "report.pdf"
        assert tenancy.scoped_key(None) is None
        assert tenancy.document_hash_salt() == b""

    def test_visitor_keys_are_prefixed(self):
        token = tenancy.set_visitor("v_abc")
        try:
            assert tenancy.scoped_key("report.pdf") == "v_abc|report.pdf"
            assert tenancy.scoped_key(None) == "v_abc|"
            assert tenancy.document_hash_salt() != b""
            assert tenancy.visitor_upload_dir().name == "v_abc"
        finally:
            tenancy.reset_visitor(token)

    def test_run_with_context_carries_visitor_into_threads(self):
        import concurrent.futures

        token = tenancy.set_visitor("v_thread")
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                plain = pool.submit(tenancy.current_visitor).result()
                wrapped = pool.submit(
                    tenancy.run_with_context(tenancy.current_visitor)
                ).result()
        finally:
            tenancy.reset_visitor(token)
        assert plain == tenancy.DEFAULT_VISITOR
        assert wrapped == "v_thread"


# ---------------------------------------------------------
# Sessions
# ---------------------------------------------------------

class TestSessionIsolation:
    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_other_visitor_cannot_list_or_read_session(
        self, mock_agent, alice, bob,
    ):
        mock_agent.return_value = "secret answer"
        sid = _sid()
        alice.post("/chat", json={"question": "my secret", "session_id": sid})

        alice_ids = [s["id"] for s in alice.get("/sessions").json()["sessions"]]
        bob_ids = [s["id"] for s in bob.get("/sessions").json()["sessions"]]
        assert sid in alice_ids
        assert sid not in bob_ids

        bob_view = bob.get(f"/sessions/{sid}/messages")
        assert "my secret" not in bob_view.text
        alice_view = alice.get(f"/sessions/{sid}/messages")
        assert "my secret" in alice_view.text

    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_reusing_session_id_does_not_leak_history(
        self, mock_agent, alice, bob,
    ):
        mock_agent.return_value = "ok"
        sid = _sid()
        alice.post("/chat", json={"question": "alice private", "session_id": sid})
        bob.post("/chat", json={"question": "bob question", "session_id": sid})

        sent_to_llm = [m["content"] for m in mock_agent.call_args.args[0]]
        assert "bob question" in sent_to_llm
        assert "alice private" not in sent_to_llm

    @patch("ai_document_agent.routes.chat.ask_agent")
    def test_other_visitor_cannot_delete_session(
        self, mock_agent, alice, bob,
    ):
        mock_agent.return_value = "ok"
        sid = _sid()
        alice.post("/chat", json={"question": "keep me", "session_id": sid})
        bob.delete(f"/sessions/{sid}")
        assert "keep me" in alice.get(f"/sessions/{sid}/messages").text


# ---------------------------------------------------------
# Uploads and images
# ---------------------------------------------------------

class TestUploadIsolation:
    def test_upload_status_hidden_from_other_visitor(self, alice, bob):
        res = alice.get("/health")  # bind a request so we can read the id
        assert res.status_code == 200

        # Register a task as Alice would: owned by her visitor ID.
        alice_cookie = alice.cookies.get(tenancy.COOKIE_NAME)
        alice_visitor = tenancy.visitor_id_from_token(alice_cookie)
        task_id = "task_" + uuid.uuid4().hex
        upload_tasks[task_id] = {
            "task_id": task_id,
            "visitor": alice_visitor,
            "status": "processing",
            "stage": "extracting",
            "filename": "alice-contract.pdf",
            "progress_pct": 20,
        }
        try:
            assert alice.get(f"/upload/status/{task_id}").status_code == 200
            res = bob.get(f"/upload/status/{task_id}")
            assert res.status_code == 404
            assert "alice-contract" not in res.text
        finally:
            upload_tasks.pop(task_id, None)

    def test_image_route_rejects_path_traversal(self, alice):
        res = alice.get("/documents/../images/x.png")
        assert res.status_code in (400, 404)
        res = alice.get("/documents/..%2F..%2Fuploads/images/x.png")
        assert res.status_code in (400, 404)
