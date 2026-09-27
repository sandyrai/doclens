# ---------------------------------------------------------
# websocket/protocol.py — Message format definitions (Phase 4)
# ---------------------------------------------------------
#
# WHAT THIS FILE DEFINES:
#
#   The "contract" between DocLens and wrapper applications.
#   Every WebSocket message (in both directions) follows a
#   standard JSON envelope:
#
#     {
#         "id": "msg_001",        ← unique message ID
#         "action": "chat",       ← what to do
#         "payload": { ... }      ← action-specific data
#     }
#
#   And every response follows:
#
#     {
#         "id": "msg_001",        ← matches the request
#         "status": "success",    ← success | error | streaming
#         "payload": { ... }      ← response data
#     }
#
# WHY A SEPARATE FILE FOR THIS?
#
#   1. DOCUMENTATION — a PHP developer building a wrapper app
#      can read THIS FILE ONLY to understand the protocol.
#
#   2. VALIDATION — Pydantic models catch malformed messages
#      BEFORE they reach the business logic. A missing "id"
#      field returns a clear error, not a Python traceback.
#
#   3. SINGLE SOURCE OF TRUTH — action names and error codes
#      are defined ONCE here. Both gateway.py and handlers.py
#      import from here, so they can never go out of sync.
#
# MESSAGE FLOW EXAMPLE:
#
#   Wrapper sends:
#     {"id": "msg_005", "action": "chat", "payload": {...}}
#
#   DocLens responds (streaming):
#     {"id": "msg_005", "status": "streaming", "payload": {"type": "token", "content": "Hello"}}
#     {"id": "msg_005", "status": "streaming", "payload": {"type": "token", "content": " world"}}
#     {"id": "msg_005", "status": "success", "payload": {"type": "complete", "answer": "Hello world"}}
#
# WHY "id" IN EVERY MESSAGE?
#
#   WebSocket is multiplexed — a wrapper can send multiple
#   requests without waiting for responses. The "id" field
#   lets the wrapper match each response to the request that
#   triggered it. Without IDs, if you send a chat and a
#   search at the same time, you wouldn't know which response
#   belongs to which request.
# ---------------------------------------------------------

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Action names — what a wrapper can ask DocLens to do
# ---------------------------------------------------------
#
# Using an Enum (instead of raw strings) means:
#   1. Typos are caught at parse time ("caht" → validation error)
#   2. IDE autocompletion works
#   3. All valid actions are listed in ONE place
#
# WHY DOTTED NAMES (document.upload, not upload)?
#
#   Namespacing. "upload" is ambiguous — upload what? Where?
#   "document.upload" is clear: it's a document action.
#   This also lets us add "image.upload" later without
#   confusion. Many APIs use this pattern (Stripe uses
#   "customer.created", Slack uses "message.posted").

class Action(str, Enum):
    """Valid WebSocket action types.

    Each action maps to a handler function in handlers.py.
    The wrapper sends one of these as the "action" field
    in its request message.

    WHY str, Enum?

        Inheriting from both str and Enum means each value
        IS a string. So Action.CHAT == "chat" is True, and
        you can use it directly in JSON without .value.
        This is a common Python pattern for string enums.
    """

    # Document operations
    DOCUMENT_UPLOAD = "document.upload"
    DOCUMENT_LIST = "document.list"
    DOCUMENT_DELETE = "document.delete"

    # Search (vector similarity, no LLM)
    SEARCH = "search"

    # Chat (LLM-powered Q&A with streaming)
    CHAT = "chat"
    CHAT_STOP = "chat.stop"

    # Suggestions (starter questions for a document)
    SUGGESTIONS = "suggestions"

    # Keep-alive / connectivity check
    PING = "ping"


# ---------------------------------------------------------
# Error codes — standard error identifiers
# ---------------------------------------------------------
#
# WHY CODES (not just messages)?
#
#   Error messages are for humans: "Document not found."
#   Error codes are for code: "DOCUMENT_NOT_FOUND"
#
#   A PHP wrapper can switch/case on the code to decide
#   what to show the user, regardless of the English
#   message. This also makes it possible to translate
#   error messages on the wrapper side.
#
#   Pattern used by: Stripe, Twilio, GitHub API, AWS

class ErrorCode(str, Enum):
    """Standard error codes for WebSocket responses.

    These appear in the "code" field of error payloads.
    Wrapper apps can switch on these to handle specific
    error conditions programmatically.
    """

    # Protocol errors (wrong message format)
    INVALID_MESSAGE = "INVALID_MESSAGE"
    INVALID_ACTION = "INVALID_ACTION"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"

    # Authentication errors
    AUTH_FAILED = "AUTH_FAILED"

    # Resource errors
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"

    # Processing errors
    PROCESSING_FAILED = "PROCESSING_FAILED"
    LLM_ERROR = "LLM_ERROR"

    # Server errors
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # Rate limiting (future use)
    CONNECTION_LIMIT = "CONNECTION_LIMIT"


# ---------------------------------------------------------
# Request model — what the wrapper sends
# ---------------------------------------------------------
#
# Pydantic validates incoming messages automatically:
#   - Missing "id" → validation error with clear message
#   - Unknown "action" → "value is not a valid Action"
#   - Extra fields → silently ignored (lenient parsing)
#
# WHY Field(default_factory=dict) FOR payload?
#
#   Some actions (like "ping" and "document.list") don't
#   need any payload data. Making it optional with an empty
#   dict default means the wrapper can omit it:
#
#     {"id": "msg_001", "action": "ping"}
#
#   Instead of forcing:
#
#     {"id": "msg_001", "action": "ping", "payload": {}}

class WSRequest(BaseModel):
    """Incoming WebSocket message from a wrapper app.

    Every message must have an 'id' and 'action'. The 'payload'
    is action-specific and defaults to an empty dict for actions
    that don't need parameters (like ping).

    Examples:
        {"id": "msg_001", "action": "ping"}
        {"id": "msg_002", "action": "chat", "payload": {"question": "..."}}
    """

    id: str = Field(
        ...,
        description=(
            "Unique message ID. The wrapper generates this. "
            "All responses to this request will carry the "
            "same ID, so the wrapper can match them."
        ),
    )

    action: Action = Field(
        ...,
        description=(
            "What to do. Must be one of the defined Action "
            "values (e.g., 'chat', 'document.upload')."
        ),
    )

    payload: dict = Field(
        default_factory=dict,
        description=(
            "Action-specific data. Structure depends on the "
            "action — see handlers.py for each action's "
            "expected payload fields."
        ),
    )


# ---------------------------------------------------------
# Response helpers — what DocLens sends back
# ---------------------------------------------------------
#
# These are NOT Pydantic models — they're plain dicts
# wrapped in helper functions. Why?
#
#   1. WebSocket.send_json() expects a dict, not a model.
#   2. Response structure is simpler than request validation.
#   3. Helper functions ensure consistent format without
#      the overhead of model instantiation on every token
#      during streaming (which can send hundreds of messages).
#
# THREE RESPONSE TYPES:
#
#   success  → the request completed successfully
#   error    → something went wrong
#   streaming → partial data (used during LLM token streaming)

def make_success(
    msg_id: str,
    payload: dict,
) -> dict:
    """Build a success response.

    Args:
        msg_id: The request's message ID (echoed back).
        payload: Response data specific to the action.

    Returns:
        {"id": "msg_001", "status": "success", "payload": {...}}
    """

    return {
        "id": msg_id,
        "status": "success",
        "payload": payload,
    }


def make_error(
    msg_id: str,
    code: ErrorCode,
    message: str,
    details: Optional[dict] = None,
) -> dict:
    """Build an error response.

    Args:
        msg_id: The request's message ID (echoed back).
        code: Machine-readable error code (ErrorCode enum).
        message: Human-readable error description.
        details: Optional extra context (e.g., which field
                 failed validation).

    Returns:
        {
            "id": "msg_001",
            "status": "error",
            "payload": {
                "code": "DOCUMENT_NOT_FOUND",
                "message": "Document 'xyz' does not exist",
                "details": {}
            }
        }
    """

    return {
        "id": msg_id,
        "status": "error",
        "payload": {
            "code": code.value if isinstance(code, ErrorCode) else code,
            "message": message,
            "details": details or {},
        },
    }


def make_streaming(
    msg_id: str,
    payload: dict,
) -> dict:
    """Build a streaming response (partial data).

    Used during LLM token streaming — each token is sent
    as a separate streaming message. The wrapper accumulates
    these tokens to build the full answer.

    Args:
        msg_id: The request's message ID (same for all
                tokens in one chat response).
        payload: Streaming data (e.g., {"type": "token",
                 "content": "Hello"}).

    Returns:
        {"id": "msg_001", "status": "streaming", "payload": {...}}

    WHY A SEPARATE STATUS (not just "success")?

        The wrapper needs to distinguish between:
          - "streaming" → more data coming, keep reading
          - "success"   → this is the final message, stop
          - "error"     → something went wrong, stop

        Without "streaming", the wrapper wouldn't know if
        a message is the last one or if more are coming.
    """

    return {
        "id": msg_id,
        "status": "streaming",
        "payload": payload,
    }
