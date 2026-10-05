"""PostgreSQL + pgvector store tests.

These need a real database with the pgvector extension,
so they are skipped unless TEST_DATABASE_URL is set:

  docker run -d --name doclens-test-db -p 5433:5432 \\
      -e POSTGRES_PASSWORD=test pgvector/pgvector:pg17
  TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:5433/postgres \\
      uv run pytest tests/test_vector_store.py -v

CI runs them against a pgvector service container.
"""

import os

import pytest

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set (needs Postgres + pgvector)",
)

from ai_document_agent import tenancy, vector_store  # noqa: E402

DIM = vector_store.EMBEDDING_DIM


@pytest.fixture(autouse=True)
def store(monkeypatch):
    monkeypatch.setattr(vector_store, "DATABASE_URL", TEST_DATABASE_URL)
    vector_store.close()
    with vector_store._get_pool().connection() as conn:
        conn.execute("TRUNCATE chunks")
    yield
    vector_store.close()


def _vec(*hot: int) -> list[float]:
    """A unit-ish vector with 1.0 at the given positions."""
    v = [0.0] * DIM
    for i in hot:
        v[i] = 1.0
    return v


def _chunk(text, source="report.pdf", page=1, idx=0):
    return {
        "text": text,
        "metadata": {"source": source, "page": page, "chunk_index": idx},
    }


def _as(visitor):
    """Context manager: act as `visitor`."""
    class _Ctx:
        def __enter__(self):
            self.token = tenancy.set_visitor(visitor)

        def __exit__(self, *exc):
            tenancy.reset_visitor(self.token)

    return _Ctx()


def _seed():
    """Alice and Bob both upload a 'report.pdf' with the same doc id."""
    with _as("v_alice"):
        vector_store.store_chunks(
            [
                _chunk("Alice revenue grew to 5 crore", page=1, idx=0),
                _chunk("Alice board meeting minutes", page=2, idx=1),
            ],
            [_vec(0), _vec(1)],
            "doc1",
        )
    with _as("v_bob"):
        vector_store.store_chunks(
            [_chunk("Bob salary slip for March", page=1, idx=0)],
            [_vec(0)],
            "doc1",
        )


class TestIsolation:
    def test_list_documents_only_shows_own(self):
        _seed()
        with _as("v_alice"):
            docs = vector_store.list_documents()
        assert docs == [{
            "document_id": "doc1", "filename": "report.pdf",
            "chunks": 2, "pages": 2,
        }]
        with _as("v_bob"):
            assert vector_store.list_documents()[0]["chunks"] == 1
        with _as("v_carol"):
            assert vector_store.list_documents() == []

    def test_semantic_search_never_returns_other_visitors_chunks(self):
        _seed()
        with _as("v_bob"):
            hits = vector_store.semantic_search(_vec(0), 10)
        assert [h["text"] for h in hits] == ["Bob salary slip for March"]

    def test_keyword_search_scoped_and_or_matching(self):
        _seed()
        with _as("v_alice"):
            hits = vector_store.keyword_search(
                "what was the revenue and salary?", 10,
            )
        texts = [h["text"] for h in hits]
        # "revenue" matches Alice's chunk even though not every
        # word does; Bob's "salary" chunk must never appear.
        assert texts == ["Alice revenue grew to 5 crore"]

    def test_keyword_search_treats_input_as_plain_text(self):
        _seed()
        with _as("v_alice"):
            # tsquery operators must not be interpreted
            hits = vector_store.keyword_search("revenue & !(board) | :*", 10)
        assert "Alice revenue grew to 5 crore" in [h["text"] for h in hits]

    def test_page_chunks_and_counts_scoped(self):
        _seed()
        with _as("v_alice"):
            assert [c["text"] for c in vector_store.page_chunks(1)] == [
                "Alice revenue grew to 5 crore"
            ]
            assert vector_store.count_chunks() == 2
            assert vector_store.count_chunks("report.pdf") == 2
            assert vector_store.count_chunks("other.pdf") == 0
        with _as("v_bob"):
            assert vector_store.count_chunks() == 1

    def test_cannot_delete_other_visitors_document(self):
        _seed()
        with _as("v_bob"):
            assert vector_store.delete_document("doc1") == 1
            assert vector_store.delete_document("doc1") == 0
        with _as("v_alice"):
            assert vector_store.count_chunks() == 2

    def test_restore_replaces_instead_of_duplicating(self):
        _seed()
        with _as("v_alice"):
            vector_store.store_chunks(
                [_chunk("Alice v2", idx=0)], [_vec(2)], "doc1",
            )
            assert vector_store.count_chunks() == 1


class TestRetention:
    def test_purge_removes_old_visitor_chunks_but_not_local(self):
        _seed()
        with _as(tenancy.DEFAULT_VISITOR):
            vector_store.store_chunks(
                [_chunk("local note")], [_vec(3)], "local1",
            )
        with vector_store._get_pool().connection() as conn:
            conn.execute(
                "UPDATE chunks SET created_at = now() - interval '30 days'"
            )

        removed = vector_store.purge_expired_chunks(days=7)

        assert removed == 3
        with _as(tenancy.DEFAULT_VISITOR):
            assert vector_store.count_chunks() == 1
