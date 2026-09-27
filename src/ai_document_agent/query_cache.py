# ---------------------------------------------------------
# Semantic Query Cache — Smart Answer Caching
# ---------------------------------------------------------
#
# WHY THIS EXISTS:
#
#   When a user asks the same question (or a semantically
#   similar one), we re-run the full pipeline every time:
#
#     1. Generate query embedding     (~0.5s)
#     2. Search ChromaDB + BM25       (~0.3s)
#     3. Call the LLM for an answer   (~10-30s)
#
#   That's 10-30 seconds wasted for a question we've
#   already answered. With caching, the same question
#   returns in <0.5 seconds.
#
# HOW IT WORKS (Semantic Cache — Option B):
#
#   Unlike a simple hash cache (Option A), which only
#   matches IDENTICAL questions, semantic caching uses
#   embedding similarity to detect questions that MEAN
#   the same thing:
#
#     "how many pages?"           → cached answer
#     "How many pages are there?" → SAME cached answer
#     "total number of pages?"    → SAME cached answer
#     "what's on page 3?"         → DIFFERENT question
#
#   Steps:
#     1. On first question: embed it, get the answer from
#        the LLM, store both in the cache.
#     2. On next question: embed it, compare cosine
#        similarity to all cached entries (same document
#        filter). If similarity > THRESHOLD → cache hit.
#     3. On document upload/delete: invalidate ALL cache
#        entries (the answers may no longer be correct).
#
# SIMILARITY THRESHOLD:
#
#   0.92 is conservative — it requires questions to be
#   very similar in meaning. This prevents false matches
#   like "who is the teacher?" matching "what is taught?".
#
#   You can lower it to 0.88 for more aggressive caching
#   (more hits, but risk of wrong answers), or raise it
#   to 0.95 for extra safety (fewer hits, more LLM calls).
#
# STORAGE:
#
#   We use the same SQLite database as chat history
#   (data/chat_history.db). Embeddings are stored as
#   JSON-encoded lists of floats. This isn't the most
#   space-efficient, but for a single-user app with
#   hundreds of cached queries, it's perfectly fine.
#
#   For a production system with millions of queries,
#   you'd use a vector database (like ChromaDB itself,
#   or Pinecone/Qdrant) for faster similarity search.
#
# CACHE INVALIDATION:
#
#   The cache is invalidated (all entries deleted) when:
#     - A new document is uploaded
#     - A document is deleted
#
#   This is intentionally aggressive. If the document
#   collection changes, any cached answer could be wrong
#   because it was based on the OLD set of documents.
#   Better to re-compute than serve stale answers.
# ---------------------------------------------------------

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Cosine similarity threshold for a cache hit.
#
# WHAT DOES THIS NUMBER MEAN?
#
#   Cosine similarity ranges from -1 to 1:
#     1.0  = identical meaning
#     0.95 = very similar (paraphrase)
#     0.90 = similar topic, same intent
#     0.80 = related topic, different question
#     0.50 = vaguely related
#     0.0  = unrelated
#
#   At 0.92, we match questions like:
#     "how many pages?" ↔ "total number of pages?"  (~0.94)
#     "what is data science?" ↔ "define data science" (~0.93)
#
#   But NOT:
#     "how many pages?" ↔ "what's on page 3?"  (~0.78)
#     "who is the teacher?" ↔ "what is taught?" (~0.72)

SIMILARITY_THRESHOLD = 0.92

# Maximum number of cached entries to keep.
# When exceeded, oldest entries are deleted first.
# 500 is plenty for a single-user app — that's 500
# unique questions cached.
MAX_CACHE_ENTRIES = 500

# Cache TTL in seconds (24 hours).
# Entries older than this are considered stale and
# will be re-computed on next query.
CACHE_TTL_SECONDS = 86400  # 24 hours


# ---------------------------------------------------------
# Database path — reuse the same DB as chat history
# ---------------------------------------------------------

from ai_document_agent.database import (
    DB_PATH,
    DATA_DIR,
)


def _get_connection() -> sqlite3.Connection:
    """Get a connection to the cache database.

    Uses the same database file as chat history
    (data/chat_history.db) but a separate table.
    """

    DATA_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    return conn


# ---------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------
#
# TABLE: query_cache
#
#   id:            Auto-incrementing row ID
#   question:      The original question text (for display)
#   source_filter: Which document filter was active
#                  (NULL = all documents)
#   embedding:     JSON-encoded list of floats (768 dims)
#                  The question's embedding vector, used
#                  for cosine similarity matching.
#   answer:        The cached answer text
#   content_type:  NULL for text answers, "flashcards"/etc.
#                  for interactive content
#   html_content:  NULL for text answers, full HTML string
#                  for interactive content
#   created_at:    When this entry was cached
#
# WHY A SEPARATE TABLE (not reusing messages)?
#
#   1. Messages table stores conversation history.
#      Cache is a performance optimization — different
#      lifecycle, different access patterns.
#   2. We need the embedding vector for similarity search.
#      Messages don't have embeddings.
#   3. Cache gets bulk-deleted on document changes.
#      We don't want to wipe chat history too.

def init_query_cache() -> None:
    """Create the query_cache table if it doesn't exist.

    Called once at server startup, alongside init_db().
    Safe to call multiple times.
    """

    conn = _get_connection()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                question      TEXT NOT NULL,
                source_filter TEXT,
                embedding     TEXT NOT NULL,
                answer        TEXT NOT NULL,
                content_type  TEXT,
                html_content  TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for quick lookups by source_filter
        # (we only compare embeddings within the same filter)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_source
                ON query_cache(source_filter)
        """)

        conn.commit()

        # Count existing entries for logging
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM query_cache"
        ).fetchone()["cnt"]

        logger.info(
            "Query cache initialized (%d entries)", count
        )

    finally:
        conn.close()


# ---------------------------------------------------------
# Cosine similarity — the math behind semantic matching
# ---------------------------------------------------------
#
# WHAT IS COSINE SIMILARITY?
#
#   Imagine each embedding as an arrow pointing in
#   768-dimensional space. Cosine similarity measures
#   the ANGLE between two arrows:
#
#     - Same direction (angle ≈ 0°) → similarity ≈ 1.0
#     - Perpendicular (angle ≈ 90°) → similarity ≈ 0.0
#     - Opposite (angle ≈ 180°)     → similarity ≈ -1.0
#
#   The key insight: MEANING is encoded as DIRECTION.
#   Similar questions point in similar directions,
#   regardless of exact wording.
#
# THE FORMULA:
#
#   cosine_sim(A, B) = (A · B) / (|A| × |B|)
#
#   Where:
#     A · B = sum of element-wise products
#     |A|   = length (norm) of vector A
#     |B|   = length (norm) of vector B
#
# WHY NOT USE A LIBRARY?
#
#   numpy or scipy could do this in one line, but we
#   already have them as indirect dependencies and
#   the pure-Python version is fast enough for our
#   use case (comparing against ~500 cached embeddings).
#   For 10,000+ entries, you'd switch to numpy.

def _cosine_similarity(
    vec_a: list[float],
    vec_b: list[float],
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector (768 floats).
        vec_b: Second embedding vector (768 floats).

    Returns:
        Similarity score between -1.0 and 1.0.
        Higher = more similar meaning.
    """

    # Dot product: sum of element-wise multiplication
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Magnitude (length) of each vector
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    # Guard against zero-length vectors (shouldn't happen
    # with real embeddings, but be safe)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------
# Cache lookup — find a cached answer for a similar question
# ---------------------------------------------------------

def lookup_cache(
    query_embedding: list[float],
    source_filter: str | None = None,
) -> dict | None:
    """Search the cache for a semantically similar question.

    Steps:
      1. Fetch all cache entries with the same source_filter
      2. For each entry, compute cosine similarity between
         the query embedding and the cached embedding
      3. If the best match exceeds SIMILARITY_THRESHOLD,
         return the cached answer
      4. Otherwise return None (cache miss)

    Args:
        query_embedding: The new question's embedding vector.
        source_filter:   Which document filter is active
                         (must match cached entries exactly).

    Returns:
        Dict with cached answer details if hit, None if miss.
        On hit: {"question", "answer", "content_type",
                 "html_content", "similarity", "cached": True}
    """

    conn = _get_connection()

    try:
        # Fetch all cached entries for this source_filter.
        #
        # WHY FETCH ALL instead of a WHERE similarity > X?
        #   SQLite doesn't have vector similarity operators.
        #   We compute similarity in Python. For ~500 entries,
        #   this takes <10ms — totally acceptable.
        #
        # We also filter out stale entries (older than TTL).

        if source_filter:
            rows = conn.execute(
                """
                SELECT * FROM query_cache
                WHERE source_filter = ?
                  AND created_at > datetime('now',
                      ? || ' seconds')
                """,
                (source_filter, f"-{CACHE_TTL_SECONDS}"),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM query_cache
                WHERE source_filter IS NULL
                  AND created_at > datetime('now',
                      ? || ' seconds')
                """,
                (f"-{CACHE_TTL_SECONDS}",),
            ).fetchall()

        if not rows:
            return None

        # Find the best match by cosine similarity
        best_match = None
        best_similarity = 0.0

        for row in rows:

            # Decode the stored embedding from JSON
            cached_embedding = json.loads(row["embedding"])

            # Compute similarity
            similarity = _cosine_similarity(
                query_embedding, cached_embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = row

        # Check if the best match exceeds the threshold
        if (
            best_match is not None
            and best_similarity >= SIMILARITY_THRESHOLD
        ):
            logger.info(
                "Cache HIT: '%.50s' ≈ '%.50s' "
                "(similarity=%.4f)",
                best_match["question"],
                best_match["question"],  # logged for debug
                best_similarity,
            )

            return {
                "question": best_match["question"],
                "answer": best_match["answer"],
                "content_type": best_match["content_type"],
                "html_content": best_match["html_content"],
                "similarity": round(best_similarity, 4),
                "cached": True,
            }

        # Log near-misses for debugging threshold tuning
        if best_match and best_similarity > 0.85:
            logger.info(
                "Cache NEAR-MISS: '%.50s' "
                "(similarity=%.4f, threshold=%.2f)",
                best_match["question"],
                best_similarity,
                SIMILARITY_THRESHOLD,
            )

        return None

    finally:
        conn.close()


# ---------------------------------------------------------
# Cache storage — save a new answer to the cache
# ---------------------------------------------------------

def store_in_cache(
    question: str,
    query_embedding: list[float],
    answer: str,
    source_filter: str | None = None,
    content_type: str | None = None,
    html_content: str | None = None,
) -> None:
    """Store a question-answer pair in the cache.

    Also enforces MAX_CACHE_ENTRIES by deleting the oldest
    entries if the cache is full.

    Args:
        question:        The original question text.
        query_embedding: The question's embedding vector.
        answer:          The LLM's answer text.
        source_filter:   Active document filter (or None).
        content_type:    None for text, "flashcards"/etc.
        html_content:    None for text, full HTML string.
    """

    conn = _get_connection()

    try:
        # Serialize the embedding as JSON
        embedding_json = json.dumps(query_embedding)

        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT INTO query_cache
                (question, source_filter, embedding,
                 answer, content_type, html_content,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question, source_filter, embedding_json,
                answer, content_type, html_content, now,
            ),
        )
        conn.commit()

        logger.info(
            "Cached answer for: '%.60s' "
            "(filter=%s, type=%s)",
            question,
            source_filter or "all",
            content_type or "text",
        )

        # Enforce max cache size — delete oldest entries
        # if we've exceeded the limit.
        #
        # WHY DELETE OLDEST?
        #   Simple and effective. Old entries are less
        #   likely to be asked again. FIFO eviction.
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM query_cache"
        ).fetchone()["cnt"]

        if count > MAX_CACHE_ENTRIES:
            excess = count - MAX_CACHE_ENTRIES
            conn.execute(
                """
                DELETE FROM query_cache
                WHERE id IN (
                    SELECT id FROM query_cache
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                """,
                (excess,),
            )
            conn.commit()

            logger.info(
                "Evicted %d old cache entries "
                "(total was %d, max %d)",
                excess, count, MAX_CACHE_ENTRIES,
            )

    finally:
        conn.close()


# ---------------------------------------------------------
# Cache invalidation — clear when documents change
# ---------------------------------------------------------

def invalidate_cache(
    source_filter: str | None = None,
) -> int:
    """Clear cached entries when documents change.

    Called when:
      - A new document is uploaded
      - A document is deleted
      - User explicitly requests cache clear

    Args:
        source_filter: If provided, only clear entries for
                       this specific document. If None,
                       clear ALL cached entries (nuclear
                       option — safest when unsure).

    Returns:
        Number of entries deleted.
    """

    conn = _get_connection()

    try:
        if source_filter:
            # Clear entries for a specific document
            cursor = conn.execute(
                "DELETE FROM query_cache WHERE source_filter = ?",
                (source_filter,),
            )
        else:
            # Clear ALL entries — safest approach
            cursor = conn.execute(
                "DELETE FROM query_cache"
            )

        conn.commit()
        deleted = cursor.rowcount

        if deleted > 0:
            logger.info(
                "Invalidated %d cache entries (filter=%s)",
                deleted,
                source_filter or "ALL",
            )

        return deleted

    finally:
        conn.close()


def get_cache_stats() -> dict:
    """Get cache statistics for debugging/monitoring.

    Returns:
        Dict with total entries, oldest entry age,
        and entries per source_filter.
    """

    conn = _get_connection()

    try:
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM query_cache"
        ).fetchone()["cnt"]

        oldest = conn.execute(
            "SELECT MIN(created_at) as oldest "
            "FROM query_cache"
        ).fetchone()["oldest"]

        # Count per source_filter
        by_source = conn.execute(
            """
            SELECT
                COALESCE(source_filter, 'all') as source,
                COUNT(*) as cnt
            FROM query_cache
            GROUP BY source_filter
            """
        ).fetchall()

        return {
            "total_entries": total,
            "max_entries": MAX_CACHE_ENTRIES,
            "oldest_entry": oldest,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "by_source": {
                row["source"]: row["cnt"]
                for row in by_source
            },
        }

    finally:
        conn.close()
