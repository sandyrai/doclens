# ---------------------------------------------------------
# main.py — Application entry point (Phase 5: Hardening)
# ---------------------------------------------------------
#
# WHAT CHANGED IN PHASE 5:
#
#   1. REQUEST ID MIDDLEWARE — every HTTP request now gets a
#      unique ID automatically (RequestIDMiddleware). No more
#      calling make_request_id() in each route handler.
#
#   2. DATABASE-BACKED UPLOAD TASKS — upload tasks are now
#      persisted to SQLite so they survive server restarts.
#      init_upload_tasks() creates the table at startup.
#
#   3. GRACEFUL SHUTDOWN — the server notifies all connected
#      WebSocket wrappers before shutting down (close code
#      1001 = "going away"). This is done via FastAPI's
#      lifespan/shutdown event system.
#
#   4. OLD TASK CLEANUP — stale upload tasks older than 24
#      hours are pruned at startup.
#
#   5. VERSION BUMP — 0.4.0 → 0.5.0
#
# WHAT CHANGED IN PHASE 4:
#
#   Added the WebSocket gateway — wrapper applications can
#   now connect via ws://host/ws?api_key=dak_xxx for real-
#   time bidirectional communication with DocLens.
#
# WHAT CHANGED IN PHASE 2:
#
#   BEFORE (Phase 1): main.py was a ~1900-line monolith
#   containing ALL 16+ route handlers, background processing,
#   request models, helper functions, and constants.
#
#   AFTER (Phase 2): main.py is a slim ~150-line file that
#   only does setup and wiring:
#
#     1. Load .env (MUST be first — see explanation below)
#     2. Configure logging
#     3. Create FastAPI app
#     4. Set up CORS + Request ID middleware
#     5. Mount static files
#     6. Initialize databases (SQLite, cache, usage, tasks)
#     7. Wire up route modules (include_router)
#     8. Register shutdown handler (Phase 5)
#
#   All route logic moved to:
#     routes/health.py      → GET /, /health, /usage
#     routes/chat.py        → POST /chat, /chat/stream
#     routes/upload.py      → POST /upload, /upload/status
#     routes/documents.py   → GET/DELETE /documents, images
#     routes/sessions.py    → CRUD /sessions
#     routes/suggestions.py → GET /suggestions
#
#   Shared state and helpers moved to:
#     shared.py → upload_tasks, document_suggestions,
#                 ChatRequest, etc.
#
#   Middleware organized in:
#     middleware/cors.py       → CORS configuration
#     middleware/request_id.py → Request ID (Phase 5)
#     middleware/auth.py       → API key auth (Phase 3)
#
# WHY THIS STRUCTURE?
#
#   1. READABILITY — you can understand any feature by reading
#      ONE file instead of scrolling through 1900 lines.
#
#   2. TEAMWORK — two developers can edit chat.py and
#      upload.py at the same time without merge conflicts.
#
#   3. TESTING — you can test routes/chat.py independently
#      by importing its router into a test FastAPI app.
#
#   4. NAVIGATION — "where's the upload logic?" → upload.py.
#      No more Ctrl+F through a giant file.
#
#   5. MAINTENANCE — adding a new endpoint? Create a new
#      route file, add a router, include it here. Done.
# ---------------------------------------------------------


# ---------------------------------------------------------
# Load .env FIRST — before ANY other imports
# ---------------------------------------------------------
#
# WHY THIS MUST BE AT THE VERY TOP:
#
#   When Python imports a module, it runs ALL the code at
#   the module level immediately. Our llm_provider.py reads
#   LLM_PROVIDER, LLM_MODELS and API keys from environment
#   variables at import time.
#
#   If we import agent.py (which imports llm_provider.py)
#   BEFORE calling load_dotenv(), the .env file hasn't been
#   loaded yet, so os.getenv("LLM_PROVIDER") returns None
#   and defaults to "ollama" — even if .env says "gemini".
#
#   Order matters:
#     1. load_dotenv()         <- reads .env into os.environ
#     2. import routes/...     <- routes import agent/llm_provider
#     3. llm_provider reads   <- NOW sees your LLM_PROVIDER

from dotenv import load_dotenv
load_dotenv()  # Must happen before importing routes (which import agent)

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------
# Import initialization functions
# ---------------------------------------------------------
#
# These create database tables at startup. We call them
# here (not in the route files) because:
#   1. They should run ONCE at server start
#   2. They must run BEFORE any request arrives
#   3. main.py is the single entry point — the right place
#      for one-time setup

from ai_document_agent.database import (
    init_db,
    init_api_clients,
    init_upload_tasks,
    cleanup_old_upload_tasks,
)
from ai_document_agent.query_cache import init_query_cache
from ai_document_agent.rate_limiter import (
    cleanup_old_usage,
    init_usage_limits,
)

# Import shared constants (BASE_DIR for static files)
from ai_document_agent.shared import BASE_DIR

# Import CORS setup from middleware
from ai_document_agent.middleware.cors import setup_cors

# Import Request ID middleware (Phase 5)
from ai_document_agent.middleware.request_id import (
    RequestIDMiddleware,
)

# Import all route modules at once via the routes package.
# See routes/__init__.py for how all_routers is assembled.
from ai_document_agent.routes import all_routers

# Import WebSocket gateway (Phase 4)
# The ws_router provides the /ws endpoint for wrapper apps.
# Also import the manager for graceful shutdown (Phase 5).
from ai_document_agent.websocket import ws_router
from ai_document_agent.websocket.gateway import manager


# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
#
# Why use logging instead of print()?
#
# 1. TIMESTAMPS — every log line shows when it happened.
# 2. LOG LEVELS — INFO for normal flow, ERROR for failures.
# 3. SOURCE — shows which module the log came from.
# 4. FILTERING — turn off noisy modules without code changes.
# 5. PRODUCTION-READY — logs can go to files, Datadog, etc.

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)-5s %(name)s  "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="DocLens",
    description=(
        "AI-powered document analysis engine with real-time "
        "WebSocket API for wrapper applications"
    ),
    version="0.5.0",  # Bumped for Phase 5: Production Hardening
)


# ---------------------------------------------------------
# Middleware setup
# ---------------------------------------------------------
#
# Middleware runs on EVERY request. We configure it here,
# right after creating the app, before mounting routes.
#
# Order matters for middleware — they execute in REVERSE
# order of how they're added:
#
#   Added first  → runs LAST  (outermost wrapper)
#   Added last   → runs FIRST (innermost wrapper)
#
# So the order below means:
#   1. CORS is added first → runs last → always adds
#      CORS headers, even on error responses.
#   2. RequestIDMiddleware is added second → runs first →
#      every request gets an ID before any route sees it.
#
# Visually:
#   Client → [CORS outer] → [RequestID inner] → Route → back out

setup_cors(app)

# ---------------------------------------------------------
# Request ID middleware (Phase 5)
# ---------------------------------------------------------
#
# Generates a unique "req_xxxxxxxx" ID for every HTTP
# request and stores it in request.state.request_id.
# Also adds an X-Request-ID header to every response.
#
# Before Phase 5, each route called make_request_id()
# manually — easy to forget on new endpoints. Now it's
# automatic and consistent.

app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------
# Static files
# ---------------------------------------------------------
#
# Mount the /static directory so the browser can load
# app.js, styles.css, and other frontend assets.

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)


# ---------------------------------------------------------
# Database initialization
# ---------------------------------------------------------
#
# All three init functions create their respective tables
# IF they don't already exist. Safe to call multiple times.
#
# Order:
#   1. init_db()          → sessions + messages tables
#   2. init_query_cache() → query_cache table (same DB)
#   3. init_usage_limits()→ usage_limits table (same DB)
#   4. cleanup_old_usage()→ delete records older than 30 days

# ---------------------------------------------------------
# Order matters — tables must exist before anything else
# tries to query them:
#   1. init_db()                 → sessions + messages tables
#   2. init_api_clients()        → api_clients table (Phase 3)
#   3. init_query_cache()        → query_cache table
#   4. init_usage_limits()       → usage_limits table
#   5. init_upload_tasks()       → upload_tasks table (Phase 5)
#   6. cleanup_old_usage()       → prune stale usage records
#   7. cleanup_old_upload_tasks()→ prune stale upload tasks

init_db()
init_api_clients()
init_query_cache()
init_usage_limits()
init_upload_tasks()
cleanup_old_usage(days_to_keep=30)

# ---------------------------------------------------------
# Clean up old upload tasks (Phase 5)
# ---------------------------------------------------------
#
# Upload tasks older than 24 hours are deleted. These are
# tasks from previous sessions that no browser is polling
# anymore. Keeps the database tidy.

cleaned = cleanup_old_upload_tasks(hours=24)
if cleaned > 0:
    logger.info(
        "Cleaned up %d stale upload task(s) at startup",
        cleaned,
    )


# ---------------------------------------------------------
# Wire up all route modules
# ---------------------------------------------------------
#
# This is the key step of Phase 2: instead of defining
# routes directly on `app`, we import routers from the
# routes package and include them. Each router brings its
# own endpoints, and they all get mounted on the main app.
#
# app.include_router(router) takes all the @router.get(),
# @router.post(), etc. decorators from that router and
# registers them on the real app. It's like copy-pasting
# the route definitions here, but without the clutter.
#
# You could also add prefixes here if needed:
#   app.include_router(chat_router, prefix="/api/v1")
# But for now we keep the same URL structure as before.

for router in all_routers:
    app.include_router(router)

# ---------------------------------------------------------
# WebSocket gateway (Phase 4)
# ---------------------------------------------------------
#
# The WebSocket route lives alongside the HTTP routes on
# the same server and port. When a wrapper app connects to
# ws://host/ws?api_key=dak_xxx, it goes through the ws_router
# which handles authentication, message routing, and the
# connection lifecycle.
#
# This is separate from the HTTP routers above because:
#   1. WebSocket has a fundamentally different lifecycle
#      (persistent connection vs request/response)
#   2. It uses API key auth, not IP-based rate limiting
#   3. It's used by wrapper apps, not the web UI

app.include_router(ws_router)


# ---------------------------------------------------------
# Shutdown event handler (Phase 5)
# ---------------------------------------------------------
#
# WHAT IS A SHUTDOWN EVENT?
#
#   FastAPI (via Starlette) lets you register functions
#   that run when the server is shutting down — triggered
#   by SIGTERM (Docker stop), SIGINT (Ctrl+C), or uvicorn's
#   graceful shutdown.
#
#   This is the place to clean up resources:
#     - Close database connections
#     - Flush caches
#     - Notify connected clients
#
# WHY @app.on_event("shutdown")?
#
#   This decorator registers an async function that FastAPI
#   calls during the shutdown sequence. It runs AFTER the
#   server stops accepting new connections but BEFORE
#   existing connections are forcibly terminated.
#
#   This gives us a window to send close frames to WebSocket
#   clients so they know the shutdown is intentional.
#
# NOTE: @app.on_event() is the older API. FastAPI 0.93+
#   introduced "lifespan" as the newer pattern. We use
#   on_event() here because it's simpler to understand
#   and works fine for our single-event use case.

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close all WebSocket connections on shutdown.

    This runs when the server receives SIGTERM or SIGINT.
    It notifies all connected wrapper apps that the server
    is shutting down (close code 1001 = "Going Away"),
    giving them a chance to reconnect gracefully.
    """

    logger.info("DocLens shutting down...")

    await manager.shutdown_all()

    logger.info("DocLens shutdown complete.")


logger.info(
    "DocLens started (Phase 5: Production Hardening) "
    "— %d HTTP route modules + WebSocket gateway loaded",
    len(all_routers),
)
