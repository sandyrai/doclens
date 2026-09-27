# ---------------------------------------------------------
# routes/sessions.py — Chat session management (Phase 2)
# ---------------------------------------------------------
#
# WHAT'S IN THIS FILE:
#
#   GET    /sessions                      → List all sessions
#   POST   /sessions                      → Create new session
#   GET    /sessions/{id}/messages        → Get chat messages
#   DELETE /sessions/{id}                 → Delete a session
#
# WHY GROUP THESE TOGETHER?
#
#   These are all CRUD operations on chat sessions — the
#   containers that hold conversation messages. They share
#   the same path prefix (/sessions) and the same database
#   functions (from database.py).
#
#   Chat endpoints (POST /chat, /chat/stream) are separate
#   because they involve AI processing, streaming, and rate
#   limiting — much more complex than simple CRUD.
# ---------------------------------------------------------

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ai_document_agent.database import (
    create_session,
    delete_session as db_delete_session,
    get_session,
    get_session_messages as db_get_session_messages,
    list_sessions,
)
from ai_document_agent.shared import (
    MAX_MESSAGES_PER_SESSION,
    make_request_id,
)

router = APIRouter(tags=["sessions"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# GET /sessions — List all chat sessions
# ---------------------------------------------------------

@router.get("/sessions")
def get_sessions():
    """List all chat sessions, newest first.

    Returns the most recent 50 sessions with their
    titles, timestamps, and message counts. Used by
    the frontend sidebar to show the conversation list.
    """

    try:
        sessions = list_sessions(limit=50)
        return {"sessions": sessions}

    except Exception as exc:

        logger.error(
            "Failed to list sessions: %s",
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


# ---------------------------------------------------------
# POST /sessions — Create a new chat session
# ---------------------------------------------------------

@router.post("/sessions")
def create_new_session():
    """Create a new chat session.

    Called when the user clicks "New Chat" in the
    sidebar. Returns the new session's ID so the
    frontend can switch to it.
    """

    request_id = make_request_id()

    try:
        session = create_session()

        logger.info(
            "[%s] Created new session %s",
            request_id,
            session["id"][:8],
        )

        return session

    except Exception as exc:

        logger.error(
            "[%s] Failed to create session: %s",
            request_id,
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "request_id": request_id,
            },
        )


# ---------------------------------------------------------
# GET /sessions/{session_id}/messages — Get chat messages
# ---------------------------------------------------------

@router.get("/sessions/{session_id}/messages")
def get_chat_messages(session_id: str):
    """Get all messages for a specific session.

    Called when the user clicks a conversation in the
    sidebar. Returns messages in chronological order
    so the frontend can display them top-to-bottom.
    """

    try:
        session = get_session(session_id)

        if session is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": (
                        f"Session '{session_id}' "
                        f"not found."
                    ),
                },
            )

        messages = db_get_session_messages(
            session_id,
            limit=MAX_MESSAGES_PER_SESSION,
        )

        return {
            "session": session,
            "messages": messages,
        }

    except Exception as exc:

        logger.error(
            "Failed to get messages for %s: %s",
            session_id[:8],
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


# ---------------------------------------------------------
# DELETE /sessions/{session_id} — Delete a chat session
# ---------------------------------------------------------

@router.delete("/sessions/{session_id}")
def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages.

    Called when the user clicks the delete button on
    a conversation in the sidebar.
    """

    request_id = make_request_id()

    logger.info(
        "[%s] DELETE /sessions/%s",
        request_id,
        session_id[:8],
    )

    success = db_delete_session(session_id)

    if success:
        return {
            "status": "deleted",
            "session_id": session_id,
            "request_id": request_id,
        }

    return JSONResponse(
        status_code=404,
        content={
            "error": (
                f"Session '{session_id}' not found."
            ),
            "request_id": request_id,
        },
    )
