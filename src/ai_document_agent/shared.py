# ---------------------------------------------------------
# shared.py — Shared state, helpers, and models (Phase 2)
# ---------------------------------------------------------
#
# WHY THIS MODULE EXISTS:
#
#   When we split main.py into separate route files (Phase 2),
#   several things are needed by MORE THAN ONE route module:
#
#     - upload_tasks dict   → used by upload.py (write) and
#                              upload status polling (read)
#     - document_suggestions → used by upload.py (write after
#                              processing) and suggestions.py (read)
#     - make_request_id()   → used by every route for logging
#     - _get_client_ip()    → used by chat.py and upload.py
#                              for rate limiting
#     - _sanitize_filename()→ used by upload.py
#     - ChatRequest model   → used by chat.py
#     - Constants           → BASE_DIR, UPLOAD_DIR, etc.
#
#   Instead of each route file importing from every other
#   route file (circular imports!), we put shared state here.
#   Every route imports from shared.py — clean, one-way deps.
#
# CIRCULAR IMPORT PREVENTION:
#
#   Without shared.py, you'd get this:
#     upload.py imports from chat.py (for ChatRequest)
#     chat.py imports from upload.py (for upload_tasks)
#     → Python raises ImportError: circular import
#
#   With shared.py:
#     upload.py imports from shared.py ✓
#     chat.py imports from shared.py   ✓
#     → No circular dependency
#
# ---------------------------------------------------------

import re
import uuid
from pathlib import Path
from time import time as _time

from fastapi import Request
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Path constants
# ---------------------------------------------------------
#
# BASE_DIR points to the project root (two levels up from
# this file: src/ai_document_agent/shared.py → project root).
#
# UPLOAD_DIR is where uploaded files are saved on disk
# before processing. We create it if it doesn't exist.

BASE_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

MAX_MESSAGES_PER_SESSION = 50

# Maximum upload file size: 20 MB
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------
#
# WHY MODULE-LEVEL DICTS?
#
#   Python modules are singletons — importing shared.py from
#   two different files gives them the SAME dict object.
#   So upload.py can write to upload_tasks and the status
#   endpoint can read from it, because they share the same
#   dict instance.
#
#   This is safe for a single-server app. For multiple
#   servers, you'd replace these with Redis or a database.

# Tracks background upload processing tasks.
# Key: task_id (str), Value: dict with status, stage, etc.
upload_tasks: dict[str, dict] = {}

# Stores AI-generated starter questions per document.
# Key: filename (str), Value: list of question strings.
document_suggestions: dict[str, list[str]] = {}


# ---------------------------------------------------------
# Request ID generation
# ---------------------------------------------------------
#
# Every API call gets a unique request_id like "req_a1b2".
# This ID appears in:
#   - every log line for this request
#   - every streaming event sent to the browser
#   - error responses
#
# Why?
#   When something goes wrong, you search your terminal
#   for the request_id and see the complete trace —
#   which session, what the user asked, which tool was
#   called, where it failed. Without this, debugging
#   concurrent requests is nearly impossible.

def make_request_id() -> str:
    """Generate a short, unique request ID."""
    return "req_" + uuid.uuid4().hex[:8]


# ---------------------------------------------------------
# IP address extraction
# ---------------------------------------------------------

def get_client_ip(request: Request) -> str:
    """Extract the client's IP address from a request.

    WHY A HELPER FUNCTION?

        FastAPI gives us the client IP via request.client.host.
        But when the app runs behind a reverse proxy (nginx,
        Cloudflare, AWS ALB), request.client.host is the
        PROXY's IP, not the real user's IP.

        Reverse proxies set the X-Forwarded-For header with
        the original client IP. We check that header first,
        falling back to request.client.host for direct
        connections (like local development).

    X-FORWARDED-FOR FORMAT:

        X-Forwarded-For: client_ip, proxy1_ip, proxy2_ip

        The FIRST value is the original client. We take that.

    NAMING NOTE (Phase 2):

        Previously called _get_client_ip() with a leading
        underscore (Python convention for "private"). Now
        that it's in a shared module imported by multiple
        route files, it's a public API — so we dropped the
        underscore. Same function, better name for its new
        role.
    """

    # Check for proxy header first
    forwarded_for = request.headers.get("x-forwarded-for")

    if forwarded_for:
        # Take the first IP (the original client)
        return forwarded_for.split(",")[0].strip()

    # Direct connection — use the socket's remote address
    if request.client:
        return request.client.host

    # Fallback (should never happen in practice)
    return "unknown"


# ---------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------

def sanitize_filename(raw_name: str) -> str:
    """Sanitize an uploaded filename to prevent
    path traversal attacks.

    Strips directory components, replaces dangerous
    characters, and ensures a safe basename remains.

    NAMING NOTE (Phase 2):

        Previously called _sanitize_filename() with a
        leading underscore. Now public since upload.py
        imports it from shared.py.
    """

    # Take only the final path component — kills
    # "../../etc/passwd" and "C:\\Windows\\system32\\x"
    name = Path(raw_name).name

    # Remove any remaining path separators
    name = name.replace("/", "_").replace("\\", "_")

    # Remove null bytes and other control characters
    name = re.sub(r'[\x00-\x1f]', '', name)

    # Collapse whitespace
    name = name.strip()

    if not name:
        name = "unnamed_upload"

    return name


# ---------------------------------------------------------
# Request / response models
# ---------------------------------------------------------
#
# WHY PUT MODELS HERE (not in chat.py)?
#
#   ChatRequest is used by both POST /chat and POST
#   /chat/stream in chat.py. If we later add a /chat/retry
#   or /chat/edit endpoint in another file, it would need
#   ChatRequest too. Keeping models in shared.py prevents
#   circular imports and makes them easy to find.

class ChatRequest(BaseModel):
    """Request body for chat endpoints.

    Fields:
        question:      The user's question text.
        session_id:    Browser session ID. Auto-generated
                       if not provided.
        source_filter: Optional filename to scope retrieval
                       to a single document.
    """

    question: str
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Browser session ID. If not provided, "
            "a new session is created."
        ),
    )
    source_filter: str | None = Field(
        default=None,
        description=(
            "Optional filename to scope retrieval "
            "to a single document. If None, searches "
            "all uploaded documents."
        ),
    )
