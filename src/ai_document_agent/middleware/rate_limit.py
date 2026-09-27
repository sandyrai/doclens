# ---------------------------------------------------------
# middleware/rate_limit.py — Rate limiting re-export (Phase 2)
# ---------------------------------------------------------
#
# WHY THIS FILE EXISTS:
#
#   The actual rate limiting logic lives in rate_limiter.py
#   (created in Phase 1). This file re-exports its functions
#   so that imports follow the new package structure:
#
#     OLD (Phase 1):
#       from ai_document_agent.rate_limiter import get_usage
#
#     NEW (Phase 2, optional):
#       from ai_document_agent.middleware.rate_limit import get_usage
#
#   Both import paths work — this is just for organizational
#   clarity. The route files still import from rate_limiter
#   directly (which is fine), but having this file here
#   signals to developers: "rate limiting is middleware,
#   look in middleware/ for it."
#
# WHY NOT MOVE rate_limiter.py INTO middleware/?
#
#   rate_limiter.py was created in Phase 1 and already works.
#   Moving it would break any existing imports and add
#   unnecessary churn. Re-exporting gives us the best of
#   both worlds: the new structure is visible, but nothing
#   breaks.
#
# FUTURE (Phase 3):
#
#   When we add authentication, rate limiting might become
#   a true FastAPI middleware (runs on every request) instead
#   of being checked inside individual route handlers. At
#   that point, this file would contain the middleware class
#   and rate_limiter.py would become its backing store.
# ---------------------------------------------------------

# Re-export everything from the original module.
# This makes both import paths equivalent:
#   from ai_document_agent.rate_limiter import X
#   from ai_document_agent.middleware.rate_limit import X

from ai_document_agent.rate_limiter import (  # noqa: F401
    check_and_increment_question,
    check_and_increment_upload,
    cleanup_old_usage,
    get_usage,
    init_usage_limits,
)
