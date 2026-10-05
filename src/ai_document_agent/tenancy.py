# ---------------------------------------------------------
# tenancy.py — Per-visitor isolation
# ---------------------------------------------------------
#
# WHY THIS EXISTS:
#
#   DocLens started as a single-user tool: every uploaded
#   document, chat session and cached answer was visible to
#   whoever opened the app. That's fine on your own laptop,
#   but on a public URL one visitor could read, search or
#   delete another visitor's files.
#
#   With VISITOR_ISOLATION=true, every request is tied to a
#   visitor ID and everything that stores user data is
#   scoped to it:
#
#     - document chunks (vector_store.py, visitor_id column)
#     - uploaded files, table CSVs and images (one folder
#       per visitor under uploads/visitors/)
#     - chat sessions (database.py, scoped session IDs)
#     - cached answers and document suggestions (scoped keys)
#
# HOW A VISITOR IS IDENTIFIED:
#
#   Browsers get a random 128-bit token in an HttpOnly
#   cookie on their first request. The visitor ID is a hash
#   of that token, so the raw token never appears in the
#   database, folder names or logs.
#
#   WebSocket clients (external apps) are identified by a
#   hash of their API key instead, so each app gets its own
#   private document space.
#
# HOW THE ID REACHES THE CODE THAT NEEDS IT:
#
#   VisitorMiddleware stores the ID in a ContextVar before
#   the request is handled. Everything that runs for that
#   request — the route, streamed responses and background
#   upload processing — inherits the same context, so
#   storage code calls current_visitor() instead of every
#   function growing a new parameter.
#
#   Threads started with loop.run_in_executor() do NOT
#   inherit context; use run_with_context() for those.
#
# WITH ISOLATION OFF (the default, for local use):
#
#   Every request belongs to DEFAULT_VISITOR and storage
#   behaves exactly as before: one shared document space.
# ---------------------------------------------------------

from __future__ import annotations

import contextvars
import functools
import hashlib
import logging
import os
import re
import secrets
import time
from http.cookies import SimpleCookie
from pathlib import Path

logger = logging.getLogger(__name__)

VISITOR_ISOLATION = os.getenv(
    "VISITOR_ISOLATION", "false"
).strip().lower() in {"1", "true", "yes"}

# Days to keep a visitor's documents before they are
# purged. Only applies with isolation on; local
# single-user data is never purged automatically.
VISITOR_RETENTION_DAYS = int(
    os.getenv("VISITOR_RETENTION_DAYS", "7")
)

DEFAULT_VISITOR = "local"

COOKIE_NAME = "doclens_vid"
COOKIE_MAX_AGE = 365 * 24 * 60 * 60
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
VISITORS_DIR = UPLOAD_DIR / "visitors"

_visitor: contextvars.ContextVar[str] = contextvars.ContextVar(
    "doclens_visitor", default=DEFAULT_VISITOR,
)


# ---------------------------------------------------------
# Reading / setting the current visitor
# ---------------------------------------------------------

def current_visitor() -> str:
    """The visitor ID for the request being handled."""
    return _visitor.get()


def set_visitor(visitor_id: str) -> contextvars.Token:
    """Bind a visitor ID to the current context.

    Used by the middleware, and by tests to act as a
    specific visitor.
    """
    return _visitor.set(visitor_id)


def reset_visitor(token: contextvars.Token) -> None:
    _visitor.reset(token)


def is_default_visitor() -> bool:
    return current_visitor() == DEFAULT_VISITOR


def visitor_id_from_token(token: str) -> str:
    """Hash a browser token into a visitor ID."""
    return "v_" + hashlib.sha256(token.encode()).hexdigest()[:24]


def visitor_id_from_api_key(api_key: str) -> str:
    """Hash an API key into a visitor ID for WebSocket apps."""
    return "k_" + hashlib.sha256(api_key.encode()).hexdigest()[:24]


def run_with_context(fn, *args, **kwargs):
    """Wrap a call so it runs inside the current context.

    loop.run_in_executor() runs functions in a thread pool
    WITHOUT copying ContextVars, so the worker would see
    the default visitor. Use:

        loop.run_in_executor(None, run_with_context(fn, a, b))
    """
    ctx = contextvars.copy_context()
    return functools.partial(ctx.run, fn, *args, **kwargs)


# ---------------------------------------------------------
# Scoped storage locations and keys
# ---------------------------------------------------------

def visitor_upload_dir() -> Path:
    """Folder holding the current visitor's uploaded files.

    The default visitor keeps the original uploads/ folder,
    so existing local installs keep working unchanged.
    """
    if is_default_visitor():
        path = UPLOAD_DIR
    else:
        path = VISITORS_DIR / current_visitor()
    path.mkdir(parents=True, exist_ok=True)
    return path


def visitor_images_dir() -> Path:
    """Folder holding images extracted from the visitor's PDFs."""
    return visitor_upload_dir() / "images"


def scoped_key(key: str | None) -> str | None:
    """Prefix an in-memory or cache key with the visitor ID.

    Keys such as filenames are only unique per visitor:
    two visitors can both upload "report.pdf". The default
    visitor's keys are left unchanged.
    """
    if is_default_visitor():
        return key
    return f"{current_visitor()}|{key or ''}"


def visitor_key_prefix() -> str:
    """Prefix shared by every scoped_key() of this visitor."""
    return f"{current_visitor()}|"


def document_hash_salt() -> bytes:
    """Salt for content-hash document IDs.

    Without it, two visitors uploading the same file would
    get the same document_id and share image folders and
    table CSVs. The default visitor keeps unsalted IDs.
    """
    if is_default_visitor():
        return b""
    return current_visitor().encode() + b"\x00"


# ---------------------------------------------------------
# Retention
# ---------------------------------------------------------

def purge_expired_visitor_files(
    days: int = VISITOR_RETENTION_DAYS,
) -> int:
    """Delete visitor files older than `days`.

    Files are written when a document is uploaded, so this
    expires the same documents whose rows
    vector_store.purge_expired_chunks() removes. Folders
    left empty are removed too. Returns the number of files
    deleted.
    """
    if not VISITORS_DIR.exists():
        return 0

    cutoff = time.time() - days * 24 * 60 * 60
    removed = 0

    for path in VISITORS_DIR.rglob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1

    # Deepest folders first, so parents empty out too.
    for folder in sorted(
        (p for p in VISITORS_DIR.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            folder.rmdir()  # only succeeds when empty
        except OSError:
            pass

    if removed:
        logger.info(
            "Purged %d expired visitor file(s)", removed,
        )
    return removed


# ---------------------------------------------------------
# Middleware
# ---------------------------------------------------------

def _cookie_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name != b"cookie":
            continue
        cookie = SimpleCookie()
        try:
            cookie.load(value.decode("latin-1"))
        except Exception:  # noqa: BLE001 — malformed cookie header
            continue
        morsel = cookie.get(COOKIE_NAME)
        if morsel and _TOKEN_RE.match(morsel.value):
            return morsel.value
    return None


def _is_https(scope: dict, trust_proxy_headers: bool) -> bool:
    if scope.get("scheme") in ("https", "wss"):
        return True
    if trust_proxy_headers:
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-proto":
                return value.decode("latin-1").strip() == "https"
    return False


def _api_key_from_query(scope: dict) -> str | None:
    from urllib.parse import parse_qs

    query = scope.get("query_string", b"").decode("latin-1")
    values = parse_qs(query).get("api_key")
    return values[0] if values else None


class VisitorMiddleware:
    """Pure ASGI middleware that binds each request to a visitor.

    Pure ASGI (not BaseHTTPMiddleware) so the ContextVar is
    set in the same context that later runs the route,
    streaming body and background tasks.
    """

    def __init__(self, app, trust_proxy_headers: bool = False):
        self.app = app
        self.trust_proxy_headers = trust_proxy_headers

    async def __call__(self, scope, receive, send):
        if not VISITOR_ISOLATION or scope["type"] not in (
            "http", "websocket",
        ):
            await self.app(scope, receive, send)
            return

        token = _cookie_token(scope.get("headers", []))
        new_token = None

        if scope["type"] == "websocket":
            api_key = _api_key_from_query(scope)
            if api_key:
                visitor_id = visitor_id_from_api_key(api_key)
            elif token:
                visitor_id = visitor_id_from_token(token)
            else:
                # No identity at all: a private, throwaway
                # space for this one connection.
                visitor_id = visitor_id_from_token(
                    secrets.token_hex(16)
                )
        else:
            if token is None:
                new_token = secrets.token_hex(16)
                token = new_token
            visitor_id = visitor_id_from_token(token)

        ctx_token = set_visitor(visitor_id)

        if new_token is None:
            send_wrapper = send
        else:
            secure = _is_https(scope, self.trust_proxy_headers)
            cookie = (
                f"{COOKIE_NAME}={new_token}; Path=/; "
                f"Max-Age={COOKIE_MAX_AGE}; HttpOnly; SameSite=Lax"
                + ("; Secure" if secure else "")
            )

            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append(
                        (b"set-cookie", cookie.encode("latin-1"))
                    )
                    message = {**message, "headers": headers}
                await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_visitor(ctx_token)
