# ---------------------------------------------------------
# routes/chat.py — Chat endpoints (Phase 2)
# ---------------------------------------------------------
#
# WHAT'S IN THIS FILE:
#
#   POST /chat        → Non-streaming chat (returns full answer)
#   POST /chat/stream → Streaming chat (NDJSON, token by token)
#
# WHY TWO CHAT ENDPOINTS?
#
#   /chat/stream is the primary endpoint — the frontend uses
#   it to show the AI "typing" in real time. Each token
#   arrives as a JSON line (NDJSON format) and the UI
#   appends it to the message bubble.
#
#   /chat (non-streaming) exists as a simpler fallback for:
#     - API consumers who don't want to parse streams
#     - Testing/debugging (curl is easier without streaming)
#     - Mobile clients with poor stream support
#
# IMPORTS FROM shared.py:
#
#   ChatRequest, get_client_ip, and MAX_MESSAGES_PER_SESSION
#   all live in shared.py because they're used across multiple
#   route files. See shared.py for circular import prevention.
#
# WHAT CHANGED IN PHASE 5:
#
#   request_id is now provided by the RequestIDMiddleware
#   (middleware/request_id.py) instead of calling make_request_id()
#   in each handler. Access it via: raw_request.state.request_id
# ---------------------------------------------------------

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ai_document_agent.agent import ask_agent, stream_agent
from ai_document_agent.database import (
    create_session,
    get_messages_for_llm,
    get_session,
    save_message,
)
from ai_document_agent.rate_limiter import (
    check_and_increment_question,
)
from ai_document_agent.shared import (
    ChatRequest,
    MAX_MESSAGES_PER_SESSION,
    get_client_ip,
)

router = APIRouter(tags=["chat"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# POST /chat — Non-streaming chat
# ---------------------------------------------------------

@router.post("/chat")
def chat_endpoint(
    request: ChatRequest,
    raw_request: Request,
):
    """Answer a question about uploaded documents.

    WHY TWO 'request' PARAMETERS?

        'request: ChatRequest' is the Pydantic model — it
        parses the JSON body (question, session_id, etc.).

        'raw_request: Request' is FastAPI's HTTP request
        object — it gives us the client's IP address for
        rate limiting. FastAPI is smart enough to know the
        difference: it sees ChatRequest is a Pydantic model
        (-> parse from body) and Request is a FastAPI class
        (-> inject the HTTP request object).
    """

    # -------------------------------------------------
    # Request ID (Phase 5)
    # -------------------------------------------------
    # The RequestIDMiddleware generates this for every
    # HTTP request. No need to call make_request_id()
    # manually anymore.

    request_id = raw_request.state.request_id

    # -------------------------------------------------
    # Rate limit check (Phase 1)
    # -------------------------------------------------
    #
    # Both /chat and /chat/stream count against the same
    # daily question limit — we don't want users bypassing
    # limits by using the non-streaming endpoint.

    client_ip = get_client_ip(raw_request)
    question_check = check_and_increment_question(client_ip)

    if not question_check["allowed"]:
        logger.warning(
            "[%s] Question rate limited for IP %s "
            "(%d/%d today)",
            request_id,
            client_ip,
            question_check["used"],
            question_check["limit"],
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": question_check["message"],
                "usage": {
                    "questions_used": (
                        question_check["used"]
                    ),
                    "questions_limit": (
                        question_check["limit"]
                    ),
                },
                "request_id": request_id,
            },
        )

    logger.info(
        "[%s] POST /chat | session=%s | q=%s | ip=%s",
        request_id,
        request.session_id[:8] + "...",
        request.question[:80],
        client_ip,
    )

    try:

        # -----------------------------------------------
        # 1. Ensure session exists in the database
        # -----------------------------------------------

        session = get_session(request.session_id)

        if session is None:
            create_session(
                session_id=request.session_id,
                title="New Chat",
                source_filter=request.source_filter,
            )

        # -----------------------------------------------
        # 2. Save user message to database
        # -----------------------------------------------

        save_message(
            session_id=request.session_id,
            role="user",
            content=request.question,
        )

        # -----------------------------------------------
        # 3. Load full conversation history for the LLM
        # -----------------------------------------------

        messages = get_messages_for_llm(
            request.session_id,
            limit=MAX_MESSAGES_PER_SESSION,
        )

        # 4. Run the agent with full history
        answer = ask_agent(
            messages,
            request_id=request_id,
            source_filter=request.source_filter,
        )

        # -----------------------------------------------
        # 5. Save assistant reply to database
        # -----------------------------------------------

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=answer,
        )

        return {
            "question": request.question,
            "answer": answer,
            "session_id": request.session_id,
            "request_id": request_id,
        }

    except Exception as exc:

        logger.error(
            "[%s] /chat failed: %s",
            request_id,
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "request_id": request_id,
            },
        )


# ---------------------------------------------------------
# POST /chat/stream — Streaming chat (NDJSON)
# ---------------------------------------------------------

@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    raw_request: Request,
):
    """Stream an AI response to the user's question.

    Returns NDJSON (newline-delimited JSON) — each line is
    a JSON object with a "type" field:

      {"type": "token", "content": "The"}
      {"type": "token", "content": " answer"}
      {"type": "completed", "assistant_content": "The answer..."}
      {"type": "error", "message": "Something broke"}

    The frontend reads these line by line and appends tokens
    to the message bubble in real time, creating the "AI is
    typing" effect.
    """

    # Request ID from middleware (Phase 5)
    request_id = raw_request.state.request_id

    # -------------------------------------------------
    # Rate limit check (Phase 1)
    # -------------------------------------------------

    client_ip = get_client_ip(raw_request)
    question_check = check_and_increment_question(client_ip)

    if not question_check["allowed"]:
        logger.warning(
            "[%s] Question rate limited for IP %s "
            "(%d/%d today)",
            request_id,
            client_ip,
            question_check["used"],
            question_check["limit"],
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": question_check["message"],
                "usage": {
                    "questions_used": (
                        question_check["used"]
                    ),
                    "questions_limit": (
                        question_check["limit"]
                    ),
                },
                "request_id": request_id,
            },
        )

    logger.info(
        "[%s] POST /chat/stream | session=%s | q=%s "
        "| ip=%s (%d/%d)",
        request_id,
        request.session_id[:8] + "...",
        request.question[:80],
        client_ip,
        question_check["used"],
        question_check["limit"],
    )

    # -----------------------------------------------
    # Ensure session exists in the database
    # -----------------------------------------------

    session = get_session(request.session_id)

    if session is None:
        create_session(
            session_id=request.session_id,
            title="New Chat",
            source_filter=request.source_filter,
        )

    # -----------------------------------------------
    # Save user message BEFORE streaming
    # -----------------------------------------------
    # We save upfront so it's in the database even if
    # the stream fails midway.

    save_message(
        session_id=request.session_id,
        role="user",
        content=request.question,
    )

    # -----------------------------------------------
    # Load conversation history for the LLM
    # -----------------------------------------------

    messages = get_messages_for_llm(
        request.session_id,
        limit=MAX_MESSAGES_PER_SESSION,
    )

    def generate():
        """Generator that yields NDJSON lines.

        WHY A NESTED FUNCTION?

            StreamingResponse needs a generator (a function
            that yields values one at a time). By defining
            generate() inside chat_stream(), it has access
            to all the local variables (request, messages,
            request_id) via Python's closure mechanism —
            no need to pass them as arguments.
        """

        had_error = False

        try:

            for event in stream_agent(
                messages,
                request_id=request_id,
                source_filter=request.source_filter,
            ):

                yield json.dumps(event) + "\n"

                # Track if agent reported an error
                if event.get("type") == "error":
                    had_error = True

                # When the stream completes, save the
                # assistant answer to the database.
                if event.get("type") == "completed":

                    assistant_content = event.get(
                        "assistant_content", ""
                    )

                    if assistant_content:

                        save_message(
                            session_id=(
                                request.session_id
                            ),
                            role="assistant",
                            content=(
                                assistant_content
                            ),
                            content_type=event.get(
                                "content_type"
                            ),
                            html_content=event.get(
                                "html_content"
                            ),
                        )

        except Exception as exc:

            logger.error(
                "[%s] Stream failed: %s",
                request_id,
                exc,
                exc_info=True,
            )

            had_error = True

            yield json.dumps(
                {
                    "type": "error",
                    "message": (
                        f"Unexpected error: {exc}"
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "request_id": request_id,
                }
            ) + "\n"

        finally:

            if had_error:
                logger.info(
                    "[%s] Stream ended with error, "
                    "no assistant reply saved",
                    request_id,
                )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )
