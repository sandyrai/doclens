# ---------------------------------------------------------
# vector_store.py — Document chunks in PostgreSQL + pgvector
# ---------------------------------------------------------
#
# WHY POSTGRES INSTEAD OF CHROMADB:
#
#   ChromaDB's embedded client is a single-process, on-disk
#   store with no real multi-user story, and the BM25 index
#   that sat next to it lived in memory and was rebuilt from
#   every chunk on every upload. Postgres gives us:
#
#     - one table where every chunk carries its visitor_id,
#       so isolation is a WHERE clause enforced in one place
#     - full-text search (tsvector + GIN index) in place of
#       the in-memory BM25 index — persistent, incremental
#     - transactions, connection pooling, normal backups
#       (pg_dump), and safe use from several app workers
#
# SEARCH DESIGN:
#
#   Semantic search is an EXACT nearest-neighbour scan over
#   the visitor's own rows (found through the visitor_id
#   index). We deliberately do not add an HNSW index: an
#   approximate index ranks across ALL visitors first and
#   filters afterwards, which can return too few (or zero)
#   results for one visitor. Each visitor only has a few
#   documents, so the exact scan is fast and always correct.
#
#   Keyword search ORs the query's words together, like
#   BM25 did: a chunk matching some of the words still
#   ranks, instead of requiring every word (which is what
#   plainto_tsquery would do).
#
# CONNECTION:
#
#   DATABASE_URL, e.g.
#     postgresql://doclens:secret@127.0.0.1:5432/doclens
#   The pool opens lazily on first use, so importing this
#   module (in tests, or the CLI) needs no database.
# ---------------------------------------------------------

from __future__ import annotations

import logging
import os
import re
import threading

from ai_document_agent.tenancy import (
    DEFAULT_VISITOR,
    VISITOR_RETENTION_DAYS,
    current_visitor,
)

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://doclens:doclens@127.0.0.1:5432/doclens",
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
# Seconds to wait for a connection before giving up.
DB_TIMEOUT = float(os.getenv("DB_TIMEOUT", "10"))

# Text search configuration. "english" stems words
# (revenue/revenues) and drops stop words (the, of).
FTS_CONFIG = os.getenv("FTS_CONFIG", "english")

_pool = None
_pool_lock = threading.Lock()

_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    visitor_id   TEXT        NOT NULL,
    document_id  TEXT        NOT NULL,
    source       TEXT        NOT NULL,
    page         INTEGER     NOT NULL,
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    embedding    vector({EMBEDDING_DIM}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS
                     (to_tsvector('{FTS_CONFIG}', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (visitor_id, document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_visitor_source_page
    ON chunks (visitor_id, source, page);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_created
    ON chunks (created_at);
"""


def _get_pool():
    """Open the connection pool (and schema) on first use."""
    global _pool

    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is None:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                DATABASE_URL,
                min_size=1,
                max_size=DB_POOL_SIZE,
                open=False,
                timeout=DB_TIMEOUT,
            )
            try:
                pool.open(wait=True, timeout=DB_TIMEOUT)
                with pool.connection() as conn:
                    conn.execute(_SCHEMA)
            except Exception:
                # Don't leave a pool reconnecting in the
                # background; the next call retries cleanly.
                pool.close()
                raise
            _pool = pool
            logger.info("Vector store ready (PostgreSQL + pgvector)")

    return _pool


def close() -> None:
    """Close the pool (used at shutdown and in tests)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def _vector_literal(embedding: list[float]) -> str:
    """Format an embedding as pgvector's text input."""
    return "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"


def _source_clause(source: str | None) -> tuple[str, tuple]:
    if source:
        return " AND source = %s", (source,)
    return "", ()


# ---------------------------------------------------------
# Writes
# ---------------------------------------------------------

def store_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: str,
) -> int:
    """Replace a document's chunks for the current visitor.

    Delete + insert in one transaction, so re-processing a
    document never leaves a mix of old and new chunks.
    """
    visitor = current_visitor()
    rows = [
        (
            visitor,
            document_id,
            chunk["metadata"]["source"],
            chunk["metadata"]["page"],
            chunk["metadata"]["chunk_index"],
            chunk["text"],
            _vector_literal(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    with _get_pool().connection() as conn:
        with conn.transaction():
            conn.execute(
                "DELETE FROM chunks "
                "WHERE visitor_id = %s AND document_id = %s",
                (visitor, document_id),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (visitor_id, document_id, "
                    "source, page, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::vector)",
                    rows,
                )

    return len(rows)


def delete_document(document_id: str) -> int:
    """Delete one of the current visitor's documents.

    Returns the number of chunks removed (0 if the visitor
    has no such document — including when it belongs to
    someone else).
    """
    with _get_pool().connection() as conn:
        cur = conn.execute(
            "DELETE FROM chunks "
            "WHERE visitor_id = %s AND document_id = %s",
            (current_visitor(), document_id),
        )
        return cur.rowcount


def purge_expired_chunks(
    days: int = VISITOR_RETENTION_DAYS,
) -> int:
    """Delete visitor chunks older than `days`.

    Never touches the default (local) visitor's documents.
    """
    with _get_pool().connection() as conn:
        cur = conn.execute(
            "DELETE FROM chunks WHERE visitor_id <> %s "
            "AND created_at < now() - make_interval(days => %s)",
            (DEFAULT_VISITOR, days),
        )
        if cur.rowcount:
            logger.info(
                "Purged %d expired visitor chunk(s)", cur.rowcount,
            )
        return cur.rowcount


# ---------------------------------------------------------
# Reads
# ---------------------------------------------------------

def count_chunks(source: str | None = None) -> int:
    clause, params = _source_clause(source)
    with _get_pool().connection() as conn:
        row = conn.execute(
            "SELECT count(*) FROM chunks WHERE visitor_id = %s"
            + clause,
            (current_visitor(), *params),
        ).fetchone()
    return row[0]


def list_documents() -> list[dict]:
    """The current visitor's documents, oldest first."""
    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT document_id, min(source), count(*), "
            "count(DISTINCT page) FROM chunks "
            "WHERE visitor_id = %s "
            "GROUP BY document_id ORDER BY min(created_at)",
            (current_visitor(),),
        ).fetchall()

    return [
        {
            "document_id": doc_id,
            "filename": source,
            "chunks": chunks,
            "pages": pages,
        }
        for doc_id, source, chunks, pages in rows
    ]


def semantic_search(
    query_embedding: list[float],
    n_results: int,
    source: str | None = None,
) -> list[dict]:
    """Nearest chunks by cosine distance (exact scan)."""
    clause, params = _source_clause(source)
    vector = _vector_literal(query_embedding)

    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT content, source, page, "
            "embedding <=> %s::vector AS distance "
            "FROM chunks WHERE visitor_id = %s" + clause
            + " ORDER BY distance LIMIT %s",
            (vector, current_visitor(), *params, n_results),
        ).fetchall()

    return [
        {"text": text, "source": src, "page": page, "distance": dist}
        for text, src, page, dist in rows
    ]


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def keyword_search(
    query: str,
    n_results: int,
    source: str | None = None,
) -> list[dict]:
    """Full-text search; chunks matching ANY query word rank."""
    words = []
    for word in _WORD_RE.findall(query.lower()):
        word = word.strip("_")
        if len(word) >= 2 and word not in words:
            words.append(word)
    if not words:
        return []

    # Each word goes through plainto_tsquery on its own, so
    # user text is never parsed as tsquery syntax; the
    # results are then OR-ed together.
    or_query = " || ".join(
        [f"plainto_tsquery('{FTS_CONFIG}', %s)"] * len(words)
    )
    clause, params = _source_clause(source)

    with _get_pool().connection() as conn:
        rows = conn.execute(
            f"WITH q AS (SELECT {or_query} AS query) "
            "SELECT content, source, page, "
            "ts_rank_cd(tsv, q.query) AS score "
            "FROM chunks, q "
            "WHERE visitor_id = %s AND tsv @@ q.query" + clause
            + " ORDER BY score DESC LIMIT %s",
            (*words, current_visitor(), *params, n_results),
        ).fetchall()

    return [
        {
            "text": text,
            "source": src,
            "page": page,
            "bm25_score": round(float(score), 4),
        }
        for text, src, page, score in rows
    ]


def page_chunks(
    page: int,
    source: str | None = None,
) -> list[dict]:
    """Every chunk on one page, in reading order."""
    clause, params = _source_clause(source)
    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT content, source, page FROM chunks "
            "WHERE visitor_id = %s AND page = %s" + clause
            + " ORDER BY source, chunk_index",
            (current_visitor(), page, *params),
        ).fetchall()

    return [
        {"text": text, "source": src, "page": pg}
        for text, src, pg in rows
    ]
