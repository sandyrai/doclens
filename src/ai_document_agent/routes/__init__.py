# ---------------------------------------------------------
# routes/__init__.py — Route package (Phase 2)
# ---------------------------------------------------------
#
# WHY A PACKAGE (directory with __init__.py)?
#
#   Python needs an __init__.py file to treat a directory
#   as a package that can be imported. Without it, you'd
#   get "ModuleNotFoundError: No module named 'routes'".
#
# WHAT THIS FILE DOES:
#
#   It collects all the APIRouter instances from the sub-
#   modules and exposes them in one place. main.py can then
#   do:
#
#     from ai_document_agent.routes import all_routers
#
#     for router in all_routers:
#         app.include_router(router)
#
#   Instead of importing from each file individually.
#
# WHY APIRouter?
#
#   FastAPI's APIRouter is like a "mini FastAPI app". Each
#   route file creates its own router, defines endpoints on
#   it, and then main.py "mounts" (includes) that router
#   into the real app.
#
#   Benefits:
#     1. Each file is self-contained — you can read chat.py
#        without knowing about upload.py.
#     2. Routes can be tested independently.
#     3. You can add route prefixes (e.g., /api/v1/chat)
#        without changing the route files themselves.
#     4. main.py stays clean — just setup + wiring.
# ---------------------------------------------------------

from ai_document_agent.routes.health import router as health_router
from ai_document_agent.routes.chat import router as chat_router
from ai_document_agent.routes.upload import router as upload_router
from ai_document_agent.routes.documents import router as documents_router
from ai_document_agent.routes.sessions import router as sessions_router
from ai_document_agent.routes.suggestions import router as suggestions_router

# Collect all routers into a list for easy wiring in main.py.
# Order doesn't matter for functionality, but we list them
# in a logical reading order: health → core features → support.
all_routers = [
    health_router,
    chat_router,
    upload_router,
    documents_router,
    sessions_router,
    suggestions_router,
]
