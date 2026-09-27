# ---------------------------------------------------------
# middleware/request_id.py — Request ID Middleware (Phase 5)
# ---------------------------------------------------------
#
# WHAT THIS FILE DOES:
#
#   Adds a unique request ID to EVERY HTTP request, automatically.
#   Before Phase 5, each route handler called make_request_id()
#   manually. Now the middleware does it once, centrally.
#
# WHY MIDDLEWARE IS BETTER THAN PER-ROUTE:
#
#   1. CONSISTENCY — every request gets an ID, even if the
#      developer forgets to add it. With per-route calls,
#      a new endpoint might skip make_request_id().
#
#   2. RESPONSE HEADER — the middleware adds X-Request-ID to
#      the HTTP response. Clients (browser, wrapper apps) can
#      read this header for debugging without parsing the body.
#
#   3. LOGGING — the request_id is available via request.state
#      throughout the entire request lifecycle, including in
#      error handlers and middleware that runs AFTER the route.
#
# HOW IT WORKS:
#
#   FastAPI middleware wraps every request:
#
#     Client → Middleware (before) → Route Handler → Middleware (after) → Client
#
#   Our middleware:
#     BEFORE: generates request_id, stores in request.state
#     AFTER:  adds X-Request-ID header to response
#
# WHAT IS request.state?
#
#   FastAPI/Starlette provides request.state as a place to
#   attach custom data to a request. It's like a dict that
#   travels with the request through middleware and handlers.
#
#   We store request_id there so any route handler can access
#   it via: request.state.request_id
#
#   This is the standard pattern — Django uses request.META,
#   Express uses req.locals, FastAPI uses request.state.
#
# X-REQUEST-ID HEADER:
#
#   Many production APIs include a request ID in the response
#   headers (Stripe, GitHub, AWS). This lets:
#     - Frontend developers see the ID in browser DevTools
#     - API consumers log it for their own debugging
#     - Support teams correlate user reports with server logs
# ---------------------------------------------------------

import uuid

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every HTTP request.

    The ID is:
      1. Stored in request.state.request_id (for route handlers)
      2. Added as X-Request-ID response header (for clients)
      3. Available for logging throughout the request lifecycle

    Usage in route handlers:

        @router.get("/example")
        def example(request: Request):
            rid = request.state.request_id
            logger.info("[%s] doing something", rid)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # -------------------------------------------------
        # Generate a unique request ID
        # -------------------------------------------------
        #
        # Format: "req_" + 8 hex chars = "req_a1b2c3d4"
        #
        # Same format as the old make_request_id() in
        # shared.py, so log output looks the same.
        #
        # WHY NOT USE THE CLIENT'S X-Request-ID?
        #
        #   Some APIs accept an X-Request-ID from the client
        #   and echo it back. We ALWAYS generate our own
        #   because:
        #     1. We can't trust client-supplied IDs
        #     2. We need guaranteed uniqueness
        #     3. Our format is consistent in logs
        #
        #   If the client sends one, it's ignored — they'll
        #   get ours back in the response header.

        request_id = "req_" + uuid.uuid4().hex[:8]

        # Store in request.state for route handlers
        request.state.request_id = request_id

        # Call the actual route handler
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
