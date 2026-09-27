# ---------------------------------------------------------
# websocket/__init__.py — WebSocket package (Phase 4)
# ---------------------------------------------------------
#
# WHY A SEPARATE PACKAGE (not just one file)?
#
#   The WebSocket system has three distinct concerns:
#
#     1. PROTOCOL (protocol.py):
#        What messages look like — the JSON envelope format,
#        valid action names, error codes. This is the "contract"
#        between DocLens and any wrapper app.
#
#     2. GATEWAY (gateway.py):
#        Connection lifecycle — accept, authenticate, route
#        messages, handle disconnects. Think of it as the
#        "front door" that checks your ID and directs you.
#
#     3. HANDLERS (handlers.py):
#        Business logic for each action — what actually happens
#        when a wrapper sends a "chat" or "document.upload"
#        message. These call into agent.py, pdf_processor.py,
#        and database.py to do the real work.
#
#   Splitting into three files means:
#     - A PHP developer building a wrapper only needs to read
#       protocol.py to understand the message format.
#     - Adding a new action means editing handlers.py, not
#       touching the connection management code.
#     - The gateway can be tested independently of the
#       business logic.
#
# HOW IT CONNECTS TO main.py:
#
#   main.py imports the WebSocket route from gateway.py and
#   registers it on the FastAPI app. The WebSocket endpoint
#   lives alongside the existing HTTP routes — same server,
#   same port, different protocol.
#
#   HTTP routes: GET /documents, POST /chat, etc.
#   WebSocket:   ws://host/ws?api_key=dak_xxx
# ---------------------------------------------------------

from ai_document_agent.websocket.gateway import router as ws_router

__all__ = ["ws_router"]
