# ---------------------------------------------------------
# routes/health.py — Health check, root page, usage (Phase 5)
# ---------------------------------------------------------
#
# WHAT'S IN THIS FILE:
#
#   GET /       → Serves the frontend HTML page
#   GET /health → Health check (is the server alive?)
#   GET /usage  → Current rate limit usage for this IP
#
# WHAT CHANGED IN PHASE 5:
#
#   The /health endpoint was upgraded from a simple status
#   flag to a comprehensive system health report:
#     - Server uptime (how long since last restart)
#     - Version number (for debugging)
#     - WebSocket connections (from the connection manager)
#     - Document count (from pdf_processor)
#     - Database status (quick query to verify SQLite works)
#     - OCR availability (unchanged from before)
#
#   This gives monitoring tools and load balancers much more
#   to work with than just "healthy" / not responding.
# ---------------------------------------------------------

import logging
from time import time as _time

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from ai_document_agent.pdf_processor import (
    OCR_AVAILABLE,
    list_documents,
)
from ai_document_agent.rate_limiter import get_usage
from ai_document_agent.shared import BASE_DIR, get_client_ip

router = APIRouter(tags=["health"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Server startup timestamp
# ---------------------------------------------------------
#
# Captured once when this module is imported (= server start).
# Used to calculate uptime in the /health endpoint.
#
# WHY MODULE-LEVEL?
#
#   This code runs ONCE when Python imports health.py —
#   which happens at server startup. The value never changes
#   during the server's lifetime, so it's effectively a
#   constant that records "when did the server start?"

_server_start_time = _time()


# ---------------------------------------------------------
# GET / — Serve the frontend
# ---------------------------------------------------------

@router.get("/")
def root():
    """Serve the main HTML page.

    FastAPI serves index.html as a regular file response.
    The browser then loads app.js and styles.css from
    /static/ (mounted separately in main.py).
    """
    return FileResponse(
        BASE_DIR / "static" / "index.html"
    )


# ---------------------------------------------------------
# GET /health — Enhanced health check (Phase 5)
# ---------------------------------------------------------

@router.get("/health")
def health():
    """Comprehensive health check endpoint.

    Returns system status information useful for:
      - Monitoring tools (Prometheus, Datadog, UptimeRobot)
      - Load balancers (health check probes)
      - Frontend diagnostics panel
      - Debugging deployment issues

    WHAT CHANGED IN PHASE 5:

        Before: {"status": "healthy", "ocr_available": true}

        After:  Much richer response with uptime, version,
                WebSocket stats, document count, and DB status.

    WHY CHECK THE DATABASE?

        SQLite can occasionally lock up or corrupt. A quick
        SELECT 1 verifies the database is actually responding.
        If it fails, the health check reports "degraded"
        instead of "healthy" — a load balancer can route
        traffic elsewhere.

    WHY LAZY IMPORT OF GATEWAY?

        The WebSocket manager lives in gateway.py, which
        imports from handlers.py, which imports from agent.py.
        If we imported gateway at the top level, we'd pull
        in the entire agent/LLM stack during import. Lazy
        importing avoids this and keeps the health module
        lightweight.
    """

    # -------------------------------------------------
    # Calculate uptime
    # -------------------------------------------------

    uptime_seconds = _time() - _server_start_time

    # Format as human-readable: "2h 15m 30s"
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    # -------------------------------------------------
    # WebSocket connection stats
    # -------------------------------------------------
    #
    # Lazy import to avoid pulling in the entire
    # WebSocket/agent stack at module load time.

    try:
        from ai_document_agent.websocket.gateway import (
            manager,
        )
        ws_stats = manager.get_stats()
    except Exception:
        ws_stats = {"total_connections": 0, "apps": {}}

    # -------------------------------------------------
    # Document count
    # -------------------------------------------------

    try:
        doc_count = len(list_documents())
    except Exception:
        doc_count = -1  # -1 signals an error

    # -------------------------------------------------
    # Database status
    # -------------------------------------------------
    #
    # A quick "SELECT 1" verifies SQLite is responding.
    # If this fails, something is seriously wrong (disk
    # full, file locked, corruption).

    db_status = "ok"

    try:
        from ai_document_agent.database import (
            _get_connection,
        )
        conn = _get_connection()
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        db_status = f"error: {e}"

    # -------------------------------------------------
    # Overall status
    # -------------------------------------------------
    #
    # "healthy"  = everything works
    # "degraded" = DB has issues but server is responding

    overall = (
        "healthy" if db_status == "ok"
        else "degraded"
    )

    return {
        "status": overall,
        "version": "0.5.0",
        "uptime": uptime_str,
        "uptime_seconds": round(uptime_seconds),
        "ocr_available": OCR_AVAILABLE,
        "documents": doc_count,
        "database": db_status,
        "websocket": ws_stats,
    }


# ---------------------------------------------------------
# GET /usage — Rate limit usage (Phase 1)
# ---------------------------------------------------------

@router.get("/usage")
def get_usage_endpoint(request: Request):
    """Get current usage limits for this IP address.

    Returns upload and question counts, limits, and
    remaining allowances for today.

    The frontend uses this to show a usage counter
    near the chat input and upload button.

    WHY 'request: Request' HERE?

        We need the client's IP to look up their usage.
        FastAPI injects the HTTP request object when it
        sees the Request type hint — no body parsing needed.
    """

    ip = get_client_ip(request)

    return get_usage(ip)
