# ---------------------------------------------------------
# Chat History Database — SQLite Storage
# ---------------------------------------------------------
#
# WHY SQLITE?
#
#   Before this, chat history lived in a Python dict
#   (in-memory). That meant:
#     1. All conversations vanished on server restart
#     2. No way to browse or resume old chats
#     3. Memory grew with every message (no cleanup)
#
#   SQLite fixes all three:
#     - Data persists on disk in a single file
#     - We can query, list, and delete conversations
#     - Old sessions can be archived or cleaned up
#
# WHY NOT POSTGRES / MYSQL?
#
#   For a single-user local app, SQLite is perfect:
#     - Zero setup (no server to install or configure)
#     - Single file (easy to backup or move)
#     - Built into Python's standard library
#     - Fast enough for thousands of conversations
#
#   If you later deploy this as a multi-user web app,
#   you'd switch to PostgreSQL for concurrent writes.
#
# DATABASE FILE LOCATION:
#
#   The .db file lives in the project's "data" folder:
#     ai-document-agent/data/chat_history.db
#
#   This keeps it separate from code (src/) and uploads.
#   The "data" folder is git-ignored to avoid committing
#   personal conversation data to version control.
# ---------------------------------------------------------

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ai_document_agent.tenancy import current_visitor, is_default_visitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Database path
# ---------------------------------------------------------
#
# Path(__file__) = this file (database.py)
# .resolve()    = absolute path
# .parents[2]   = go up 2 levels:
#   src/ai_document_agent/database.py
#   → src/ai_document_agent/
#   → src/
#   → project root (ai-document-agent/)
#
# Then we create a "data" subfolder for the database file.

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "chat_history.db"


def _get_connection() -> sqlite3.Connection:
    """Create a database connection with optimal settings.

    WHY THESE SETTINGS:

    row_factory = sqlite3.Row:
        Lets us access columns by name (row["title"])
        instead of index (row[0]). Much more readable.

    isolation_level = None:
        Enables "autocommit" mode — each statement runs
        immediately without needing explicit commit().
        For a single-user app, this is simpler and safer
        (no forgotten commits = no lost data).

    WAL mode (Write-Ahead Logging):
        Allows reading while writing. Without this,
        a long write would block all reads. WAL mode
        is the recommended default for most SQLite apps.
    """

    # Ensure the data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))

    # Access columns by name instead of index
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for better concurrent read/write
    conn.execute("PRAGMA journal_mode=WAL")

    # Enable foreign key enforcement
    # (SQLite has foreign keys OFF by default!)
    conn.execute("PRAGMA foreign_keys=ON")

    return conn


# ---------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------
#
# TWO TABLES:
#
# 1. sessions — one row per conversation
#    - id:            unique identifier (UUID)
#    - title:         display name (auto-generated from
#                     first message, editable later)
#    - source_filter: which document this chat is about
#                     (NULL = all documents)
#    - created_at:    when the conversation started
#    - updated_at:    when the last message was sent
#                     (used for sorting: newest first)
#
# 2. messages — one row per chat message
#    - id:            auto-incrementing integer
#    - session_id:    links to sessions.id (foreign key)
#    - role:          "user" or "assistant"
#    - content:       the message text
#    - created_at:    when this message was sent
#
# WHY SEPARATE TABLES?
#
#   Normalisation. A session has metadata (title, dates)
#   and many messages. Putting them in one table would
#   duplicate the title/dates on every message row.
#   Two tables = clean, efficient, standard.
#
# ON DELETE CASCADE:
#   When you delete a session, ALL its messages are
#   automatically deleted too. No orphaned messages.
# ---------------------------------------------------------

def init_db() -> None:
    """Create the database tables if they don't exist.

    Called once at server startup. If the tables already
    exist, this does nothing (CREATE TABLE IF NOT EXISTS).
    Safe to call multiple times.
    """

    conn = _get_connection()

    try:
        conn.executescript("""
            -- Sessions table: one row per conversation
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New Chat',
                source_filter TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Messages table: stores every user/assistant message
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- -------------------------------------------------
                -- NEW COLUMNS for interactive content persistence
                -- -------------------------------------------------
                --
                -- WHY THESE COLUMNS?
                --
                --   When the agent generates interactive content
                --   (flashcards, quizzes, summaries), the HTML is
                --   rendered in a sandboxed iframe during the live
                --   stream. But on page reload, we need that HTML
                --   again to re-render the content.
                --
                --   Previously, assistant_content was saved as
                --   "[Generated flashcards]" — a placeholder that
                --   lost the actual interactive content.
                --
                -- content_type:
                --   What kind of content this is. NULL for normal
                --   text messages. Values like "flashcards",
                --   "quiz", "summary", "article", "qa" for
                --   interactive content.
                --
                -- html_content:
                --   The full HTML string for interactive content.
                --   NULL for normal text messages. This is what
                --   gets loaded into the iframe's srcdoc attribute
                --   when restoring from history.
                --
                content_type  TEXT,
                html_content  TEXT,

                -- Foreign key: links each message to its session.
                -- ON DELETE CASCADE means deleting a session
                -- automatically deletes all its messages.
                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
                    ON DELETE CASCADE
            );

            -- Index for fast message lookups by session
            -- Without this, every "get messages for session X"
            -- would scan the entire messages table.
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);

            -- Index for sorting sessions by most recent activity
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC);
        """)

        # -------------------------------------------------
        # Schema migration: add new columns to existing DBs
        # -------------------------------------------------
        #
        # CREATE TABLE IF NOT EXISTS won't add new columns
        # to a table that already exists. For users who have
        # an existing database (from before this update), we
        # need ALTER TABLE to add the new columns.
        #
        # SQLite's ALTER TABLE ADD COLUMN is safe to call —
        # it fails silently if the column already exists.
        # We catch the error and move on.
        #
        # WHY NOT A MIGRATION FRAMEWORK?
        #   For a single-user local app, this is simpler
        #   than adding Alembic or a migrations table.
        #   We only have 2 new columns to add — once.

        for col_sql in [
            "ALTER TABLE messages ADD COLUMN content_type TEXT",
            "ALTER TABLE messages ADD COLUMN html_content TEXT",
        ]:
            try:
                conn.execute(col_sql)
                logger.info("Migration: added column — %s", col_sql)
            except sqlite3.OperationalError:
                # Column already exists — that's fine
                pass

        logger.info(
            "Database initialized at %s", DB_PATH
        )

    finally:
        conn.close()


# ---------------------------------------------------------
# Session operations
# ---------------------------------------------------------


# ---------------------------------------------------------
# Per-visitor session IDs
# ---------------------------------------------------------
#
# Session IDs come from the browser, so with visitor
# isolation on we store them prefixed with the visitor ID.
# A visitor who sends someone else's session ID simply gets
# a different, empty session of their own: there is no way
# to name another visitor's row. The default (local)
# visitor keeps unprefixed IDs, so existing history works.

def _scope_session_id(session_id: str) -> str:
    if is_default_visitor():
        return session_id
    return f"{current_visitor()}:{session_id}"


def _unscope_session_id(session_id: str) -> str:
    if is_default_visitor():
        return session_id
    prefix = f"{current_visitor()}:"
    if session_id.startswith(prefix):
        return session_id[len(prefix):]
    return session_id


def create_session(
    session_id: str | None = None,
    title: str = "New Chat",
    source_filter: str | None = None,
) -> dict:
    """Create a new chat session.

    Args:
        session_id:    Optional custom ID. If None, a UUID
                       is generated automatically.
        title:         Display name for the conversation.
                       Defaults to "New Chat" and gets
                       updated to the first question text.
        source_filter: Which document to search. None means
                       search all documents.

    Returns:
        Dict with the new session's id, title, source_filter,
        created_at, and updated_at.
    """

    if session_id is None:
        session_id = str(uuid.uuid4())

    public_id = session_id
    session_id = _scope_session_id(session_id)

    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()

    try:
        conn.execute(
            """
            INSERT INTO sessions (id, title, source_filter, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, title, source_filter, now, now),
        )
        conn.commit()

        logger.info(
            "Created session %s: '%s'",
            session_id[:8],
            title[:50],
        )

        return {
            "id": public_id,
            "title": title,
            "source_filter": source_filter,
            "created_at": now,
            "updated_at": now,
        }

    finally:
        conn.close()


def get_session(session_id: str) -> dict | None:
    """Get a single session by ID.

    Returns:
        Session dict, or None if not found.
    """

    session_id = _scope_session_id(session_id)


    conn = _get_connection()

    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

        if row is None:
            return None

        session = dict(row)
        session["id"] = _unscope_session_id(session["id"])
        return session

    finally:
        conn.close()


def list_sessions(limit: int = 50) -> list[dict]:
    """List all chat sessions, newest first.

    Args:
        limit: Maximum number of sessions to return.
               Defaults to 50 (enough for a sidebar list).

    Returns:
        List of session dicts, ordered by updated_at DESC
        (most recently active conversation first).

    WHY ORDER BY updated_at?
        The user wants to see their most recent
        conversation at the top — not the one they
        created first. updated_at changes every time
        a new message is sent, so active chats float
        to the top automatically.
    """

    conn = _get_connection()

    try:
        # Only this visitor's sessions (all of them for the
        # default visitor, whose prefix is empty).
        prefix = "" if is_default_visitor() else f"{current_visitor()}:"
        rows = conn.execute(
            """
            SELECT s.*,
                   (SELECT COUNT(*) FROM messages m
                    WHERE m.session_id = s.id) as message_count
            FROM sessions s
            WHERE substr(s.id, 1, ?) = ?
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (len(prefix), prefix, limit),
        ).fetchall()

        sessions = [dict(row) for row in rows]
        for session in sessions:
            session["id"] = _unscope_session_id(session["id"])
        return sessions

    finally:
        conn.close()


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages.

    Thanks to ON DELETE CASCADE, deleting the session row
    automatically removes all associated messages. No
    need to manually delete from both tables.

    Returns:
        True if the session was found and deleted,
        False if it didn't exist.
    """

    session_id = _scope_session_id(session_id)


    conn = _get_connection()

    try:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE id = ?",
            (session_id,),
        )
        conn.commit()

        deleted = cursor.rowcount > 0

        if deleted:
            logger.info(
                "Deleted session %s and its messages",
                session_id[:8],
            )
        else:
            logger.warning(
                "Session %s not found for deletion",
                session_id[:8],
            )

        return deleted

    finally:
        conn.close()


def update_session_title(
    session_id: str,
    title: str,
) -> bool:
    """Update a session's title.

    This is called automatically after the first user
    message to replace "New Chat" with a meaningful
    title derived from their question.

    Args:
        session_id: The session to update.
        title:      New display title (truncated to 100 chars).

    Returns:
        True if updated, False if session not found.
    """

    session_id = _scope_session_id(session_id)


    # Truncate to 100 chars for display
    title = title[:100].strip()

    conn = _get_connection()

    try:
        cursor = conn.execute(
            """
            UPDATE sessions
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                datetime.now(timezone.utc).isoformat(),
                session_id,
            ),
        )
        conn.commit()

        return cursor.rowcount > 0

    finally:
        conn.close()


# ---------------------------------------------------------
# Message operations
# ---------------------------------------------------------

def save_message(
    session_id: str,
    role: str,
    content: str,
    content_type: str | None = None,
    html_content: str | None = None,
) -> dict:
    """Save a message to the database.

    Also updates the session's updated_at timestamp so
    it floats to the top of the session list.

    If this is the FIRST user message in the session,
    we auto-update the session title from the question
    text. This gives conversations meaningful names
    like "What are the Q3 revenue numbers?" instead
    of "New Chat".

    Args:
        session_id:   Which conversation this belongs to.
        role:         "user" or "assistant".
        content:      The message text (for normal messages)
                      or a short label like "[Generated flashcards]"
                      for interactive content.
        content_type: None for normal text messages.
                      "flashcards", "quiz", "summary", etc.
                      for interactive content that has HTML.
        html_content: None for normal text messages.
                      The full HTML string for interactive
                      content — this is what gets rendered
                      in an iframe when restoring from history.

    Returns:
        Dict with the saved message's id, session_id,
        role, content, content_type, html_content,
        and created_at.
    """

    public_id = session_id
    session_id = _scope_session_id(session_id)


    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()

    try:
        # Insert the message
        # content_type and html_content are NULL for normal
        # text messages, populated only for interactive
        # content (flashcards, quizzes, summaries, etc.)
        cursor = conn.execute(
            """
            INSERT INTO messages
                (session_id, role, content, content_type,
                 html_content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content, content_type,
             html_content, now),
        )

        message_id = cursor.lastrowid

        # Update session's updated_at timestamp
        conn.execute(
            """
            UPDATE sessions
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, session_id),
        )

        # Auto-title: if this is the first user message,
        # use it as the session title instead of "New Chat"
        if role == "user":
            msg_count = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM messages
                WHERE session_id = ? AND role = 'user'
                """,
                (session_id,),
            ).fetchone()["cnt"]

            if msg_count == 1:
                # This is the first user message —
                # generate a title from the question
                title = _generate_title(content)
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )

        conn.commit()

        return {
            "id": message_id,
            "session_id": public_id,
            "role": role,
            "content": content,
            "content_type": content_type,
            "html_content": html_content,
            "created_at": now,
        }

    finally:
        conn.close()


def get_session_messages(
    session_id: str,
    limit: int = 50,
) -> list[dict]:
    """Get all messages for a session, oldest first.

    Args:
        session_id: Which conversation to load.
        limit:      Max messages to return (default 50).
                    Oldest messages are dropped first if
                    the conversation exceeds this limit.

    Returns:
        List of message dicts, ordered chronologically
        (oldest first) so they display in chat order.

    WHY OLDEST FIRST?
        Chat UIs show messages top-to-bottom in time
        order. The database returns them in creation
        order so the frontend can just append them.

    WHY LIMIT?
        Very long conversations would be slow to load
        and use lots of memory. We keep the most recent
        50 messages, which is enough context for the
        LLM and the user. Older messages are still in
        the database — just not loaded into the UI.
    """

    public_id = session_id
    session_id = _scope_session_id(session_id)


    conn = _get_connection()

    try:
        rows = conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

        messages = [dict(row) for row in rows]
        for message in messages:
            message["session_id"] = public_id
        return messages

    finally:
        conn.close()


def get_messages_for_llm(
    session_id: str,
    limit: int = 50,
) -> list[dict]:
    """Get messages formatted for the LLM (role + content only).

    The LLM expects a list of {"role": "user", "content": "..."}
    dicts. This strips out the database metadata (id, timestamps)
    that the LLM doesn't need.

    Args:
        session_id: Which conversation to load.
        limit:      Max messages (most recent kept).

    Returns:
        List of {"role": ..., "content": ...} dicts,
        ready to pass directly to the LLM as conversation
        history.
    """

    messages = get_session_messages(session_id, limit)

    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

# ---------------------------------------------------------
# API Client operations (Phase 3: App Authentication)
# ---------------------------------------------------------
#
# WHY IN THIS FILE?
#
#   API clients are stored in the SAME SQLite database as
#   sessions and messages (chat_history.db). Keeping all
#   database operations in one module avoids:
#     1. Duplicate _get_connection() functions
#     2. Multiple modules managing the same DB file
#     3. Confusion about "which module handles the DB?"
#
# WHAT ARE API CLIENTS?
#
#   Each wrapper application (PHP education system, future
#   business apps) gets an API key to authenticate its
#   WebSocket connection to DocAgent. The api_clients table
#   stores these keys (hashed — NEVER plaintext).
#
#   DocAgent authenticates APPS, not end-users. The wrapper
#   handles its own user auth (students, teachers, etc.).
#   DocAgent only cares: "is this connection from a known,
#   authorized application?"


def init_api_clients() -> None:
    """Create the api_clients table if it doesn't exist.

    Called once at server startup, after init_db().

    TABLE STRUCTURE:

        id          — UUID primary key (e.g., "app_abc123")
        name        — Human-readable app name ("Education AI")
        key_hash    — SHA-256 hash of the API key (NEVER the
                      key itself — if the DB is stolen, the
                      actual keys remain secret)
        is_active   — 1 = active, 0 = revoked. Revoked apps
                      can't connect but their record stays for
                      audit trails.
        created_at  — When the key was generated
        last_seen   — Last WebSocket connection time (NULL if
                      never connected). Useful for finding
                      unused/orphaned keys.

    SECURITY DECISIONS:

        1. Keys are stored as SHA-256 hashes — the actual key
           is shown ONCE during creation and never stored.

        2. is_active flag instead of DELETE — revoking keeps
           the record for audit. You can see "Education AI was
           created on Aug 1, revoked on Aug 15" instead of it
           vanishing from history.

        3. UNIQUE on key_hash — prevents two apps from
           accidentally getting the same key (astronomically
           unlikely with 32 hex chars, but defense in depth).

        4. UNIQUE on name — prevents confusing duplicates
           like two apps both called "Education AI".
    """

    conn = _get_connection()

    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_clients (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                key_hash    TEXT NOT NULL UNIQUE,
                is_active   INTEGER DEFAULT 1,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen   TIMESTAMP
            );

            -- Fast lookup by key_hash (checked on every
            -- WebSocket connection handshake)
            CREATE INDEX IF NOT EXISTS idx_api_clients_hash
                ON api_clients(key_hash);
        """)

        logger.info("API clients table initialized")

    finally:
        conn.close()


def create_api_client(
    name: str,
    key_hash: str,
    client_id: str | None = None,
) -> dict:
    """Register a new API client (wrapper application).

    Called by the manage_keys.py CLI when an admin creates
    a new API key for a wrapper app.

    Args:
        name:       Human-readable name (e.g., "Education AI").
        key_hash:   SHA-256 hash of the generated API key.
        client_id:  Optional custom ID. Auto-generated if None.

    Returns:
        Dict with the client's id, name, is_active, created_at.

    Raises:
        sqlite3.IntegrityError: If name or key_hash already
            exists (UNIQUE constraint violation).
    """

    if client_id is None:
        client_id = "app_" + uuid.uuid4().hex[:12]

    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()

    try:
        conn.execute(
            """
            INSERT INTO api_clients
                (id, name, key_hash, is_active, created_at)
            VALUES (?, ?, ?, 1, ?)
            """,
            (client_id, name, key_hash, now),
        )
        conn.commit()

        logger.info(
            "Created API client '%s' (id=%s)",
            name,
            client_id,
        )

        return {
            "id": client_id,
            "name": name,
            "is_active": True,
            "created_at": now,
        }

    finally:
        conn.close()


def get_api_client_by_key_hash(key_hash: str) -> dict | None:
    """Look up an API client by its key hash.

    Called during WebSocket handshake to validate the
    provided API key:
      1. Hash the incoming key with SHA-256
      2. Call this function with the hash
      3. If found AND is_active → connection allowed
      4. If not found or inactive → reject connection

    Args:
        key_hash: SHA-256 hash of the provided API key.

    Returns:
        Client dict if found, None if not found.
        Includes is_active flag — caller must check this!
    """

    conn = _get_connection()

    try:
        row = conn.execute(
            """
            SELECT id, name, is_active, created_at, last_seen
            FROM api_clients
            WHERE key_hash = ?
            """,
            (key_hash,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        conn.close()


def list_api_clients() -> list[dict]:
    """List all registered API clients.

    Used by the manage_keys.py CLI to show which wrapper
    apps are registered and their status.

    Returns:
        List of client dicts, ordered by creation time.
        Does NOT include key_hash (security — even hashes
        shouldn't be casually displayed).
    """

    conn = _get_connection()

    try:
        rows = conn.execute(
            """
            SELECT id, name, is_active, created_at, last_seen
            FROM api_clients
            ORDER BY created_at DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


def revoke_api_client(name: str) -> bool:
    """Revoke an API client's access by name.

    Sets is_active to 0. The client record stays in the
    database for audit purposes, but the key can no longer
    be used to connect.

    WHY NOT DELETE?

        If an admin revokes "Education AI" and later wants
        to know when it was originally created or when it
        last connected, that info is still available. Hard
        deleting loses this history.

    Args:
        name: The app name to revoke (e.g., "Education AI").

    Returns:
        True if found and revoked, False if not found.
    """

    conn = _get_connection()

    try:
        cursor = conn.execute(
            """
            UPDATE api_clients
            SET is_active = 0
            WHERE name = ?
            """,
            (name,),
        )
        conn.commit()

        revoked = cursor.rowcount > 0

        if revoked:
            logger.info("Revoked API client '%s'", name)
        else:
            logger.warning(
                "API client '%s' not found for revocation",
                name,
            )

        return revoked

    finally:
        conn.close()


def update_api_client_last_seen(client_id: str) -> None:
    """Update the last_seen timestamp for an API client.

    Called when a wrapper app successfully connects via
    WebSocket. Useful for identifying inactive/orphaned
    keys that should be revoked.

    Args:
        client_id: The client's ID (e.g., "app_abc123").
    """

    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()

    try:
        conn.execute(
            """
            UPDATE api_clients
            SET last_seen = ?
            WHERE id = ?
            """,
            (now, client_id),
        )
        conn.commit()

    finally:
        conn.close()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def _generate_title(question: str) -> str:
    """Generate a short title from the first question.

    Takes the user's first question and creates a concise
    title for the conversation sidebar. Rules:

    1. Strip leading/trailing whitespace
    2. Take the first line only (ignore multi-line)
    3. Truncate to 60 characters
    4. Add "..." if truncated
    5. Fall back to "New Chat" if empty

    Examples:
        "What are the Q3 revenue numbers?"
            → "What are the Q3 revenue numbers?"
        "Can you summarize the entire document including all sections..."
            → "Can you summarize the entire document including al..."
        "" → "New Chat"
    """

    # Take first line, strip whitespace
    title = question.strip().split("\n")[0].strip()

    if not title:
        return "New Chat"

    # Truncate long titles
    if len(title) > 60:
        title = title[:57] + "..."

    return title


# ---------------------------------------------------------
# Upload Tasks — Database-Backed (Phase 5)
# ---------------------------------------------------------
#
# WHAT CHANGED IN PHASE 5:
#
#   Before: upload_tasks lived in a Python dict (shared.py).
#   That meant:
#     1. All task state vanished on server restart
#     2. An upload processing when the server restarted
#        was lost — the browser polling /upload/status
#        would get 404 forever
#     3. No history of past uploads
#
#   Now: upload_tasks are stored in SQLite alongside sessions
#   and messages. Benefits:
#     1. Tasks survive server restarts
#     2. The browser can resume polling after a restart
#     3. We can query upload history for diagnostics
#     4. Stale task cleanup uses SQL instead of Python loops
#
# WHY KEEP THE IN-MEMORY DICT TOO?
#
#   We DON'T — Phase 5 replaces the dict entirely.
#   The shared.py upload_tasks dict is kept but only used
#   as a fast cache. The database is the source of truth.
#   If the server restarts, the dict is empty but the
#   database still has the tasks.
#
# TABLE STRUCTURE:
#
#   task_id      — "task_abc123" format (primary key)
#   status       — "processing" | "ready" | "error"
#   stage        — "saving" | "extracting" | "processing" | "done" | "failed"
#   filename     — sanitized upload filename
#   document_id  — SHA-256 content hash (links to ChromaDB)
#   request_id   — correlates with server logs
#   progress_pct — 0-100 integer for progress bar
#   result_json  — JSON string of processing result (when done)
#   error        — error message (when failed)
#   created_at   — when the upload started
#   updated_at   — last progress update

def init_upload_tasks() -> None:
    """Create the upload_tasks table if it doesn't exist.

    Called once at server startup, after init_db().
    """

    conn = _get_connection()

    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS upload_tasks (
                task_id      TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'processing',
                stage        TEXT NOT NULL DEFAULT 'saving',
                filename     TEXT NOT NULL,
                document_id  TEXT NOT NULL,
                request_id   TEXT NOT NULL,
                progress_pct INTEGER DEFAULT 0,
                result_json  TEXT,
                error        TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Index for finding tasks by document_id
            -- (useful for checking if a document is still processing)
            CREATE INDEX IF NOT EXISTS idx_upload_tasks_document
                ON upload_tasks(document_id);

            -- Index for cleanup queries (find old tasks)
            CREATE INDEX IF NOT EXISTS idx_upload_tasks_created
                ON upload_tasks(created_at);
        """)

        # Mark any "processing" tasks as "error" — they were
        # interrupted by a server restart and will never finish.
        # This prevents the browser from polling forever.
        interrupted = conn.execute(
            """
            UPDATE upload_tasks
            SET status = 'error',
                stage = 'failed',
                error = 'Server restarted during processing',
                updated_at = ?
            WHERE status = 'processing'
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()

        if interrupted.rowcount > 0:
            logger.warning(
                "Marked %d interrupted upload tasks as "
                "failed (server restart)",
                interrupted.rowcount,
            )

        logger.info("Upload tasks table initialized")

    finally:
        conn.close()


def create_upload_task(
    task_id: str,
    filename: str,
    document_id: str,
    request_id: str,
) -> dict:
    """Create a new upload task record.

    Called when POST /upload accepts a file and starts
    background processing.

    Args:
        task_id:     "task_abc123" format.
        filename:    Sanitized upload filename.
        document_id: SHA-256 content hash.
        request_id:  For log correlation.

    Returns:
        Dict with the task's initial state.
    """

    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()

    try:
        conn.execute(
            """
            INSERT INTO upload_tasks
                (task_id, status, stage, filename,
                 document_id, request_id, progress_pct,
                 created_at, updated_at)
            VALUES (?, 'processing', 'saving', ?, ?, ?, 10, ?, ?)
            """,
            (task_id, filename, document_id,
             request_id, now, now),
        )
        conn.commit()

        return {
            "task_id": task_id,
            "status": "processing",
            "stage": "saving",
            "filename": filename,
            "document_id": document_id,
            "request_id": request_id,
            "progress_pct": 10,
            "result": None,
            "error": None,
            "created_at": now,
        }

    finally:
        conn.close()


def update_upload_task(
    task_id: str,
    status: str | None = None,
    stage: str | None = None,
    progress_pct: int | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """Update an upload task's progress.

    Called by _process_in_background() at each processing
    stage. Only updates the fields you pass — omitted fields
    stay unchanged.

    Args:
        task_id:      Which task to update.
        status:       New status ("processing", "ready", "error").
        stage:        New stage name.
        progress_pct: New progress percentage (0-100).
        result:       Processing result dict (stored as JSON).
        error:        Error message (when status="error").
    """

    import json as _json

    now = datetime.now(timezone.utc).isoformat()

    # Build SET clause dynamically — only update provided fields
    #
    # WHY DYNAMIC SQL?
    #   If we always SET all fields, a None value would
    #   overwrite a previously-set value. By building the
    #   query dynamically, we only touch what changed.

    updates = ["updated_at = ?"]
    params: list = [now]

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if stage is not None:
        updates.append("stage = ?")
        params.append(stage)
    if progress_pct is not None:
        updates.append("progress_pct = ?")
        params.append(progress_pct)
    if result is not None:
        updates.append("result_json = ?")
        params.append(_json.dumps(result))
    if error is not None:
        updates.append("error = ?")
        params.append(error)

    params.append(task_id)

    conn = _get_connection()

    try:
        conn.execute(
            f"UPDATE upload_tasks SET {', '.join(updates)} "
            f"WHERE task_id = ?",
            params,
        )
        conn.commit()

    finally:
        conn.close()


def get_upload_task(task_id: str) -> dict | None:
    """Get an upload task by its ID.

    Called by GET /upload/status/{task_id} to report
    progress to the polling browser.

    Returns:
        Task dict with all fields, or None if not found.
        result_json is parsed back into a dict.
    """

    import json as _json

    conn = _get_connection()

    try:
        row = conn.execute(
            "SELECT * FROM upload_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()

        if row is None:
            return None

        task = dict(row)

        # Parse result_json back into a dict
        if task.get("result_json"):
            task["result"] = _json.loads(task["result_json"])
        else:
            task["result"] = None

        return task

    finally:
        conn.close()


def cleanup_old_upload_tasks(hours: int = 24) -> int:
    """Delete upload tasks older than the given hours.

    Called periodically to prevent the table from growing
    indefinitely. Tasks older than 24 hours are unlikely
    to be polled again.

    Args:
        hours: Delete tasks older than this many hours.

    Returns:
        Number of tasks deleted.
    """

    conn = _get_connection()

    try:
        cursor = conn.execute(
            """
            DELETE FROM upload_tasks
            WHERE created_at < datetime('now', ? || ' hours')
            """,
            (f"-{hours}",),
        )
        conn.commit()

        deleted = cursor.rowcount

        if deleted > 0:
            logger.info(
                "Cleaned up %d upload tasks older than %d hours",
                deleted,
                hours,
            )

        return deleted

    finally:
        conn.close()
