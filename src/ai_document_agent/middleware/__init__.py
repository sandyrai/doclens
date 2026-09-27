# ---------------------------------------------------------
# middleware/__init__.py — Middleware package (Phase 2)
# ---------------------------------------------------------
#
# WHY A MIDDLEWARE PACKAGE?
#
#   Middleware runs on EVERY request, before your route
#   handler is called. It's the place for cross-cutting
#   concerns — things that affect all endpoints:
#
#     - CORS headers     (cors.py)     → allow browser requests
#     - Rate limiting    (rate_limit.py)→ re-exported for clarity
#     - Authentication   (auth.py)     → Phase 3 placeholder
#
# CURRENT STATE (Phase 2):
#
#   - cors.py provides the CORS setup function
#   - rate_limit.py re-exports from the existing rate_limiter
#   - auth.py is a placeholder for Phase 3
#
#   In Phase 3, auth.py will add a FastAPI Depends() that
#   checks for a valid session token before allowing access.
# ---------------------------------------------------------
