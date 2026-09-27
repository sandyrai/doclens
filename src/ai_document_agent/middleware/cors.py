# ---------------------------------------------------------
# middleware/cors.py — CORS configuration (Phase 2)
# ---------------------------------------------------------
#
# WHAT IS CORS?
#
#   Cross-Origin Resource Sharing. When a browser loads a
#   page from http://localhost:8000 and that page tries to
#   fetch from http://localhost:3000 (a different port),
#   the browser BLOCKS the request by default. CORS headers
#   tell the browser "it's okay, this server trusts that
#   origin."
#
# WHY DO WE NEED IT?
#
#   Right now the frontend is served from the SAME origin
#   as the API (both on :8000), so CORS isn't strictly
#   needed. But it's good practice to configure it because:
#
#     1. During development, you might run the frontend on
#        a different port (e.g., Vite on :5173).
#     2. In production, the frontend might be on a CDN
#        (app.example.com) while the API is on a subdomain
#        (api.example.com).
#     3. Mobile apps or third-party integrations might
#        call the API from different origins.
#
# WHY A SEPARATE FILE?
#
#   CORS config is a cross-cutting concern — it affects
#   ALL endpoints. Keeping it in middleware/cors.py:
#     - Makes it easy to find and modify
#     - Keeps main.py slim
#     - Groups it with other middleware (auth, rate limiting)
# ---------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware on the FastAPI app.

    WHY A FUNCTION (not just code at module level)?

        We need the `app` instance to add middleware to it.
        main.py calls this function after creating the app:

            app = FastAPI(...)
            setup_cors(app)

        This keeps the CORS logic here while letting main.py
        control WHEN it's applied.

    CURRENT SETTINGS:

        allow_origins=["*"]  → Accept requests from ANY origin.
                               In production, you'd restrict
                               this to your actual domain:
                               ["https://app.example.com"]

        allow_methods=["*"]  → Allow GET, POST, DELETE, etc.

        allow_headers=["*"]  → Allow any custom headers.

    SECURITY NOTE:

        allow_origins=["*"] is fine for a local/demo app.
        For production with authentication (Phase 3), you'd
        want to restrict origins to prevent CSRF attacks.
    """

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
