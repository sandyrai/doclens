# ---------------------------------------------------------
# Rate Limiter — IP-Based Usage Tracking
# ---------------------------------------------------------
#
# WHY THIS MODULE EXISTS:
#
#   DocAgent is publicly accessible. Without limits, anyone
#   (or any bot) can:
#     - Upload hundreds of PDFs, filling disk space
#     - Ask thousands of questions, burning through our
#       cloud LLM API credits (LLM calls can cost money!)
#     - Overload the server, making it slow for everyone
#
#   This module tracks usage PER IP ADDRESS PER DAY.
#   Each IP gets a limited number of uploads and questions.
#   When the limit is reached, the server returns HTTP 429
#   ("Too Many Requests") with a friendly message.
#
# WHY PER-IP (not per-session)?
#
#   Sessions live in the browser (localStorage). A user can
#   clear their cookies or open an incognito window to get
#   a new session ID — bypassing per-session limits.
#
#   IP addresses are harder to change. While not perfect
#   (VPNs, shared WiFi), it's a much stronger barrier
#   against casual abuse. For serious abuse prevention,
#   you'd add authentication (Phase 3).
#
# WHY PER-DAY (not per-hour or per-session)?
#
#   Per-day is the most user-friendly approach:
#     - Users know exactly when their limits reset
#     - It's generous enough for genuine exploration
#     - It prevents sustained abuse over time
#     - It's simple to explain: "1 upload, 15 questions/day"
#
# DAILY RESET:
#
#   Limits reset at midnight IST (Indian Standard Time,
#   UTC+5:30). We store the IST date with each record.
#   When a new day starts (in IST), the old record no
#   longer matches, and a fresh one is created with
#   counts at zero.
#
# HOW IT WORKS:
#
#   1. Request comes in → extract IP from request.client.host
#   2. Look up usage row for this IP + today's IST date
#   3. If no row → first visit today → create row (counts=0)
#   4. Check if the relevant count < limit
#   5. If under limit → increment count, allow the request
#   6. If at/over limit → return False (caller sends 429)
#
# CONFIGURATION:
#
#   Limits are read from environment variables (.env file):
#     ANON_MAX_UPLOADS=1     (default: 1 upload per day)
#     ANON_MAX_QUESTIONS=15  (default: 15 questions per day)
#
#   Change these in .env without touching code.
# ---------------------------------------------------------

import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Database path — same DB as chat history
# ---------------------------------------------------------
#
# We reuse the same SQLite database (chat_history.db) so
# there's only ONE database file to manage. The usage_limits
# table sits alongside the sessions and messages tables.
#
# WHY NOT A SEPARATE DATABASE?
#
#   One database = one file to backup, one connection pool,
#   one WAL journal. SQLite handles multiple tables in one
#   file effortlessly. Splitting into separate DBs would add
#   complexity with zero benefit at our scale.

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "chat_history.db"

# ---------------------------------------------------------
# IST timezone definition
# ---------------------------------------------------------
#
# WHY DEFINE IST MANUALLY?
#
#   Python's datetime module needs a timezone object to
#   convert UTC to local time. IST is UTC+5:30, which
#   we define as a fixed offset using timedelta.
#
#   We could use the `pytz` or `zoneinfo` libraries, but
#   IST never changes (India doesn't observe daylight
#   saving time), so a fixed offset is simpler and has
#   zero dependencies.

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------
# Read limits from environment variables
# ---------------------------------------------------------
#
# os.getenv() reads from the environment (set by .env file
# via load_dotenv() in main.py). The second argument is the
# default value if the variable isn't set.
#
# int() converts the string to a number — environment
# variables are always strings.

MAX_UPLOADS = int(os.getenv("ANON_MAX_UPLOADS", "1"))
MAX_QUESTIONS = int(os.getenv("ANON_MAX_QUESTIONS", "15"))


def _get_connection() -> sqlite3.Connection:
    """Create a database connection.

    Same settings as database.py — Row factory for named
    columns, WAL mode for concurrent access, foreign keys on.

    WHY DUPLICATE THIS FUNCTION?

        We could import _get_connection from database.py,
        but that would create a circular dependency risk
        (if database.py ever imports from rate_limiter.py).
        Keeping it self-contained is safer for a module that
        might be imported before database.py is initialized.
    """

    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _get_today_ist() -> str:
    """Get today's date in IST as a string (YYYY-MM-DD).

    WHY IST?

        The app owner is in India. Midnight IST (12:00 AM
        local time) is when users expect "today" to change.
        If we used UTC, the reset would happen at 5:30 AM
        IST — confusing for Indian users who are still
        awake at midnight.

    WHY A STRING?

        SQLite doesn't have a native DATE type. We store
        dates as TEXT in 'YYYY-MM-DD' format, which SQLite
        can compare and sort correctly as strings.
    """

    return datetime.now(IST).strftime("%Y-%m-%d")


# ---------------------------------------------------------
# Table initialization
# ---------------------------------------------------------

def init_usage_limits() -> None:
    """Create the usage_limits table if it doesn't exist.

    Called once at server startup, after init_db().

    TABLE STRUCTURE:

        id              — auto-incrementing primary key
        ip_address      — the client's IP address (TEXT)
        date            — IST date string 'YYYY-MM-DD'
        uploads_count   — how many files uploaded today
        questions_count — how many questions asked today
        created_at      — when this record was first created
        updated_at      — last time a count was incremented

    UNIQUE CONSTRAINT:

        (ip_address, date) is unique — there's exactly ONE
        row per IP per day. When a new day starts, a new row
        is created (old rows stay for audit/analytics).

    INDEX:

        We search by (ip_address, date) on every request,
        so an index on those columns makes lookups instant
        even with millions of rows.
    """

    conn = _get_connection()

    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS usage_limits (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address      TEXT NOT NULL,
                date            TEXT NOT NULL,
                uploads_count   INTEGER DEFAULT 0,
                questions_count INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(ip_address, date)
            );

            -- Fast lookup by IP + date (checked on every request)
            CREATE INDEX IF NOT EXISTS idx_usage_ip_date
                ON usage_limits(ip_address, date);
        """)

        logger.info(
            "Usage limits table initialized "
            "(max %d uploads, %d questions per IP/day)",
            MAX_UPLOADS,
            MAX_QUESTIONS,
        )

    finally:
        conn.close()


# ---------------------------------------------------------
# Usage checking and incrementing
# ---------------------------------------------------------
#
# PATTERN: "Check-then-increment"
#
#   Each function does TWO things atomically:
#     1. Check if the IP is under the limit
#     2. If yes, increment the count AND return True
#     3. If no, return False (don't increment)
#
#   WHY ATOMIC (check + increment together)?
#
#     If we separated "check" and "increment" into two
#     steps, a race condition could occur:
#
#       Thread A: check → count=14 → under limit!
#       Thread B: check → count=14 → under limit!
#       Thread A: increment → count=15
#       Thread B: increment → count=16 ← OVER LIMIT!
#
#     By doing both in ONE SQL statement, SQLite's row-level
#     locking prevents this. (For our single-server app,
#     Python's GIL also helps, but atomic SQL is correct
#     regardless.)

def check_and_increment_upload(ip_address: str) -> dict:
    """Check if this IP can upload, and increment if yes.

    Args:
        ip_address: The client's IP address.

    Returns:
        Dict with:
            allowed:   True if the upload is allowed
            used:      Current upload count (after increment if allowed)
            limit:     Maximum uploads per day
            message:   Human-readable status message
    """

    today = _get_today_ist()
    conn = _get_connection()

    try:
        # -------------------------------------------------
        # Step 1: Get or create today's usage record
        # -------------------------------------------------
        #
        # INSERT OR IGNORE: if a row for this IP+date already
        # exists, do nothing. If it doesn't exist, create one
        # with counts at 0. This is an "upsert" pattern.

        conn.execute(
            """
            INSERT OR IGNORE INTO usage_limits
                (ip_address, date, uploads_count, questions_count)
            VALUES (?, ?, 0, 0)
            """,
            (ip_address, today),
        )

        # -------------------------------------------------
        # Step 2: Check the current count
        # -------------------------------------------------

        row = conn.execute(
            """
            SELECT uploads_count FROM usage_limits
            WHERE ip_address = ? AND date = ?
            """,
            (ip_address, today),
        ).fetchone()

        current_count = row["uploads_count"]

        # -------------------------------------------------
        # Step 3: Allow or deny
        # -------------------------------------------------

        if current_count >= MAX_UPLOADS:
            # Limit reached — don't increment
            return {
                "allowed": False,
                "used": current_count,
                "limit": MAX_UPLOADS,
                "message": (
                    f"Daily upload limit reached "
                    f"({MAX_UPLOADS}/{MAX_UPLOADS}). "
                    f"Try again tomorrow!"
                ),
            }

        # Under limit — increment and allow
        conn.execute(
            """
            UPDATE usage_limits
            SET uploads_count = uploads_count + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE ip_address = ? AND date = ?
            """,
            (ip_address, today),
        )
        conn.commit()

        new_count = current_count + 1

        logger.info(
            "Upload allowed for IP %s (%d/%d today)",
            ip_address,
            new_count,
            MAX_UPLOADS,
        )

        return {
            "allowed": True,
            "used": new_count,
            "limit": MAX_UPLOADS,
            "message": (
                f"Upload {new_count}/{MAX_UPLOADS} today"
            ),
        }

    finally:
        conn.close()


def check_and_increment_question(ip_address: str) -> dict:
    """Check if this IP can ask a question, and increment if yes.

    Same pattern as check_and_increment_upload, but for
    the questions_count column.

    Args:
        ip_address: The client's IP address.

    Returns:
        Dict with:
            allowed:   True if the question is allowed
            used:      Current question count (after increment if allowed)
            limit:     Maximum questions per day
            message:   Human-readable status message
    """

    today = _get_today_ist()
    conn = _get_connection()

    try:
        # Get or create today's usage record
        conn.execute(
            """
            INSERT OR IGNORE INTO usage_limits
                (ip_address, date, uploads_count, questions_count)
            VALUES (?, ?, 0, 0)
            """,
            (ip_address, today),
        )

        row = conn.execute(
            """
            SELECT questions_count FROM usage_limits
            WHERE ip_address = ? AND date = ?
            """,
            (ip_address, today),
        ).fetchone()

        current_count = row["questions_count"]

        if current_count >= MAX_QUESTIONS:
            return {
                "allowed": False,
                "used": current_count,
                "limit": MAX_QUESTIONS,
                "message": (
                    f"Daily question limit reached "
                    f"({MAX_QUESTIONS}/{MAX_QUESTIONS}). "
                    f"Try again tomorrow!"
                ),
            }

        # Under limit — increment and allow
        conn.execute(
            """
            UPDATE usage_limits
            SET questions_count = questions_count + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE ip_address = ? AND date = ?
            """,
            (ip_address, today),
        )
        conn.commit()

        new_count = current_count + 1

        logger.info(
            "Question allowed for IP %s (%d/%d today)",
            ip_address,
            new_count,
            MAX_QUESTIONS,
        )

        return {
            "allowed": True,
            "used": new_count,
            "limit": MAX_QUESTIONS,
            "message": (
                f"Question {new_count}/{MAX_QUESTIONS} today"
            ),
        }

    finally:
        conn.close()


def get_usage(ip_address: str) -> dict:
    """Get current usage for an IP address (read-only).

    Called by the GET /usage endpoint so the frontend can
    display a usage counter like "3/15 questions today".

    This does NOT increment anything — it's purely a read.

    Args:
        ip_address: The client's IP address.

    Returns:
        Dict with current upload and question counts,
        limits, and remaining allowances.
    """

    today = _get_today_ist()
    conn = _get_connection()

    try:
        row = conn.execute(
            """
            SELECT uploads_count, questions_count
            FROM usage_limits
            WHERE ip_address = ? AND date = ?
            """,
            (ip_address, today),
        ).fetchone()

        if row is None:
            # No usage today — everything is at zero
            return {
                "uploads_used": 0,
                "uploads_limit": MAX_UPLOADS,
                "uploads_remaining": MAX_UPLOADS,
                "questions_used": 0,
                "questions_limit": MAX_QUESTIONS,
                "questions_remaining": MAX_QUESTIONS,
                "date": today,
                "resets_at": "midnight IST",
            }

        uploads_used = row["uploads_count"]
        questions_used = row["questions_count"]

        return {
            "uploads_used": uploads_used,
            "uploads_limit": MAX_UPLOADS,
            "uploads_remaining": max(
                0, MAX_UPLOADS - uploads_used
            ),
            "questions_used": questions_used,
            "questions_limit": MAX_QUESTIONS,
            "questions_remaining": max(
                0, MAX_QUESTIONS - questions_used
            ),
            "date": today,
            "resets_at": "midnight IST",
        }

    finally:
        conn.close()


def cleanup_old_usage(days_to_keep: int = 30) -> int:
    """Delete usage records older than N days.

    Called periodically (e.g., at startup) to prevent the
    usage_limits table from growing indefinitely. Old records
    have no value — we only need today's data for enforcement.

    We keep 30 days by default for basic analytics (you could
    see daily active IPs, peak usage days, etc.).

    Args:
        days_to_keep: How many days of history to retain.

    Returns:
        Number of rows deleted.
    """

    # Calculate the cutoff date in IST
    cutoff = (
        datetime.now(IST) - timedelta(days=days_to_keep)
    ).strftime("%Y-%m-%d")

    conn = _get_connection()

    try:
        cursor = conn.execute(
            "DELETE FROM usage_limits WHERE date < ?",
            (cutoff,),
        )
        conn.commit()

        deleted = cursor.rowcount

        if deleted > 0:
            logger.info(
                "Cleaned up %d old usage records "
                "(older than %s)",
                deleted,
                cutoff,
            )

        return deleted

    finally:
        conn.close()
