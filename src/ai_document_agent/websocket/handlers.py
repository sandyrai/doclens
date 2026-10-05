# ---------------------------------------------------------
# websocket/handlers.py — Action handlers (Phase 4)
# ---------------------------------------------------------
#
# WHAT THIS FILE DOES:
#
#   Contains the business logic for each WebSocket action.
#   When a wrapper app sends a message like:
#
#     {"id": "msg_001", "action": "chat", "payload": {...}}
#
#   The gateway (gateway.py) parses the message and calls
#   the matching handler function from THIS file:
#
#     action "chat"            → handle_chat()
#     action "document.upload" → handle_document_upload()
#     action "document.list"   → handle_document_list()
#     ...
#
# WHY SEPARATE FROM gateway.py?
#
#   gateway.py handles CONNECTION LIFECYCLE (accept, auth,
#   disconnect, message routing). This file handles BUSINESS
#   LOGIC (what actually happens for each action).
#
#   This separation means:
#     - Adding a new action = add ONE function here + ONE
#       entry in the ACTION_HANDLERS dict at the bottom
#     - You don't touch the gateway code at all
#     - Each handler can be tested independently
#
# HANDLER PATTERN:
#
#   Every handler function has the same signature:
#
#     async def handle_xxx(
#         websocket: WebSocket,
#         msg_id: str,
#         payload: dict,
#         connection_info: dict,
#     ) -> None
#
#   - websocket:       the live WebSocket connection (for
#                       sending responses back to the wrapper)
#   - msg_id:          the request's message ID (echoed in
#                       every response so the wrapper can
#                       match responses to requests)
#   - payload:         the action-specific data from the
#                       wrapper's message
#   - connection_info: metadata about this connection (app
#                       name, connection ID, connected_at)
#
#   Handlers send responses directly via websocket.send_json().
#   They don't return values — everything goes through the
#   WebSocket connection.
# ---------------------------------------------------------

import asyncio
import base64
import hashlib
import logging
import uuid
from pathlib import Path
from time import time as _time

from fastapi import WebSocket

from ai_document_agent.agent import (
    ask_agent,
    generate_document_suggestions,
    stream_agent,
)
from ai_document_agent.database import (
    create_session,
    get_messages_for_llm,
    get_session,
    save_message,
)
from ai_document_agent.pdf_processor import (
    SUPPORTED_EXTENSIONS,
    delete_document,
    list_documents,
    process_document,
    search_documents,   # Returns list[Evidence] dataclass objects
)
from ai_document_agent.query_cache import invalidate_cache
from ai_document_agent.shared import (
    MAX_MESSAGES_PER_SESSION,
    MAX_UPLOAD_BYTES,
    document_suggestions,
    make_request_id,
    sanitize_filename,
    upload_tasks,
)
from ai_document_agent.tenancy import (
    document_hash_salt,
    run_with_context,
    scoped_key,
    visitor_upload_dir,
)
from ai_document_agent.websocket.protocol import (
    ErrorCode,
    make_error,
    make_streaming,
    make_success,
)

logger = logging.getLogger(__name__)


async def _run_in_executor(executor, fn):
    """loop.run_in_executor() that keeps the visitor context.

    Executor threads don't inherit ContextVars, so without
    this the worker would read and write the default
    visitor's documents instead of this connection's.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, run_with_context(fn))


# ---------------------------------------------------------
# In-progress chat tracking
# ---------------------------------------------------------
#
# WHY TRACK ACTIVE CHATS?
#
#   When a wrapper sends "chat.stop", we need to know WHICH
#   chat to stop. A single WebSocket connection could have
#   multiple chats in progress (unlikely but possible).
#
#   This dict maps: connection_id → {target_msg_id → cancel_event}
#
#   When chat.stop arrives, we set the cancel event, which
#   the streaming loop checks between tokens.

active_chats: dict[str, dict[str, asyncio.Event]] = {}


# ---------------------------------------------------------
# PING handler
# ---------------------------------------------------------

async def handle_ping(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """Respond to a keep-alive ping.

    The simplest handler — just sends back "pong" with the
    server's current timestamp. Wrappers can use this to:
      1. Keep the connection alive (prevent idle timeout)
      2. Measure round-trip latency
      3. Verify the connection is still working

    The server also sends pings to clients (see gateway.py
    heartbeat). This handler is for CLIENT-initiated pings.
    """

    from datetime import datetime, timezone

    await websocket.send_json(
        make_success(msg_id, {
            "pong": True,
            "server_time": datetime.now(timezone.utc).isoformat(),
        })
    )


# ---------------------------------------------------------
# DOCUMENT.LIST handler
# ---------------------------------------------------------

async def handle_document_list(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """List all uploaded documents.

    Returns the same data as GET /documents but over WebSocket.
    Each document includes: filename, document_id, pages, chunks.

    WHY DUPLICATE THE HTTP ENDPOINT?

        Wrapper apps communicate ONLY via WebSocket — they
        don't make separate HTTP calls. So every operation
        available via HTTP must also be available via WebSocket.
        The handlers call the SAME underlying functions
        (list_documents, delete_document, etc.), just through
        a different transport.
    """

    try:
        docs = list_documents()

        await websocket.send_json(
            make_success(msg_id, {
                "documents": docs,
                "count": len(docs),
            })
        )

    except Exception as exc:

        logger.error(
            "WS document.list failed: %s",
            exc,
            exc_info=True,
        )

        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INTERNAL_ERROR,
                f"Failed to list documents: {exc}",
            )
        )


# ---------------------------------------------------------
# DOCUMENT.UPLOAD handler
# ---------------------------------------------------------

async def handle_document_upload(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """Upload and process a document via WebSocket.

    Unlike the HTTP upload (which uses multipart form data),
    the WebSocket upload sends the file as base64-encoded
    content in the payload:

        {
            "filename": "syllabus.pdf",
            "content_base64": "JVBERi0xLjQK...",
            "metadata": {"course_id": "CS101"}   ← optional
        }

    WHY BASE64 (not binary frames)?

        WebSocket supports both text and binary frames, but:
        1. JSON is text-only — you can't mix JSON metadata
           with binary file content in one text frame.
        2. Binary frames would need a custom protocol to
           carry the filename and metadata alongside the file.
        3. Base64 adds ~33% overhead, but for documents under
           20MB, the simplicity is worth it.
        4. Every language has base64 encoding built in — PHP's
           base64_encode(), Python's base64.b64encode(), etc.

    PROGRESS UPDATES:

        Unlike HTTP upload (where the browser polls a status
        endpoint), WebSocket uploads send progress updates
        PUSHED to the wrapper automatically. No polling needed!

        This is one of the key advantages of WebSocket over
        HTTP for long-running operations.
    """

    # -------------------------------------------------
    # Validate required fields
    # -------------------------------------------------

    filename = payload.get("filename")
    content_b64 = payload.get("content_base64")

    if not filename or not content_b64:
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                "Missing required fields: 'filename' and "
                "'content_base64'",
                {"required": ["filename", "content_base64"]},
            )
        )
        return

    request_id = make_request_id()
    safe_filename = sanitize_filename(filename)

    logger.info(
        "[%s] WS document.upload | file=%s | app=%s",
        request_id,
        safe_filename,
        connection_info.get("app_name", "unknown"),
    )

    # Validate file extension
    file_ext = Path(safe_filename).suffix.lower()

    if file_ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                f"Unsupported file type: {safe_filename}. "
                f"Supported: {supported}",
            )
        )
        return

    # -------------------------------------------------
    # Decode base64 content
    # -------------------------------------------------

    try:
        file_bytes = base64.b64decode(content_b64)
    except Exception:
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                "Invalid base64 encoding in 'content_base64'",
            )
        )
        return

    # File size check
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        actual_mb = len(file_bytes) / (1024 * 1024)
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                f"File too large: {actual_mb:.1f} MB. "
                f"Maximum: {max_mb} MB.",
            )
        )
        return

    try:
        # Generate document ID from content hash
        document_id = hashlib.sha256(
            document_hash_salt() + file_bytes
        ).hexdigest()[:12]

        # Check for duplicate
        existing_docs = list_documents()
        for doc in existing_docs:
            if doc["document_id"] == document_id:
                await websocket.send_json(
                    make_success(msg_id, {
                        "document_id": document_id,
                        "filename": doc["filename"],
                        "pages": doc["pages"],
                        "chunks": doc["chunks"],
                        "status": "duplicate",
                        "message": (
                            f"This file was already uploaded "
                            f"as '{doc['filename']}'."
                        ),
                    })
                )
                return

        # Save file to disk
        save_path = visitor_upload_dir() / f"{document_id}_{safe_filename}"

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # Send immediate acknowledgment
        await websocket.send_json(
            make_success(msg_id, {
                "document_id": document_id,
                "status": "processing",
                "message": "Processing started",
            })
        )

        # -------------------------------------------------
        # Process the document (with progress updates)
        # -------------------------------------------------
        #
        # Unlike the HTTP upload which runs in a background
        # thread (because HTTP needs to return immediately),
        # WebSocket can send progress updates directly.
        # We still run processing in a thread (because
        # process_document is synchronous/blocking), but
        # we push updates to the wrapper as they happen.

        # Send progress: extracting
        await websocket.send_json(
            make_streaming(msg_id, {
                "stage": "extracting",
                "progress_pct": 20,
                "message": "Extracting text from document...",
            })
        )

        # Run blocking process_document in a thread pool
        # so we don't block the async event loop
        result = await _run_in_executor(
            None,
            lambda: process_document(
                file_path=str(save_path),
                filename=safe_filename,
                document_id=document_id,
            ),
        )

        if result.get("status") == "error":
            await websocket.send_json(
                make_error(
                    msg_id,
                    ErrorCode.PROCESSING_FAILED,
                    result.get("message", "Processing failed"),
                )
            )
            return

        # Invalidate query cache (new document = stale answers)
        invalidate_cache()

        # Send progress: generating suggestions
        await websocket.send_json(
            make_streaming(msg_id, {
                "stage": "suggestions",
                "progress_pct": 80,
                "message": "Generating starter questions...",
            })
        )

        # Generate suggestions (non-critical)
        try:
            doc_text = result.get("text", "")
            if doc_text:
                suggestions = generate_document_suggestions(
                    document_text=doc_text,
                    filename=safe_filename,
                )
                document_suggestions[scoped_key(safe_filename)] = suggestions
        except Exception as e:
            logger.warning(
                "[%s] WS suggestion generation failed: %s",
                request_id,
                e,
            )

        # Send final success with full document info
        await websocket.send_json(
            make_success(msg_id, {
                "document_id": document_id,
                "filename": safe_filename,
                "status": "ready",
                "pages": result.get("pages", 0),
                "chunks": result.get("chunks", 0),
                "message": "Document processed successfully",
            })
        )

        logger.info(
            "[%s] WS upload complete: %d pages, %d chunks",
            request_id,
            result.get("pages", 0),
            result.get("chunks", 0),
        )

    except Exception as exc:

        logger.error(
            "[%s] WS document.upload failed: %s",
            request_id,
            exc,
            exc_info=True,
        )

        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INTERNAL_ERROR,
                f"Upload failed: {exc}",
            )
        )


# ---------------------------------------------------------
# DOCUMENT.DELETE handler
# ---------------------------------------------------------

async def handle_document_delete(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """Delete a document from DocLens.

    Expected payload:
        {"document_id": "abc123def456"}
    """

    document_id = payload.get("document_id")

    if not document_id:
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                "Missing required field: 'document_id'",
            )
        )
        return

    logger.info(
        "WS document.delete | doc=%s | app=%s",
        document_id,
        connection_info.get("app_name", "unknown"),
    )

    success = delete_document(document_id)

    if success:
        invalidate_cache()

        await websocket.send_json(
            make_success(msg_id, {
                "document_id": document_id,
                "message": "Document deleted",
            })
        )
    else:
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.DOCUMENT_NOT_FOUND,
                f"Document '{document_id}' does not exist",
            )
        )


# ---------------------------------------------------------
# SEARCH handler
# ---------------------------------------------------------

async def handle_search(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """Search documents using vector similarity (no LLM).

    Expected payload:
        {
            "query": "what is photosynthesis",
            "source": "biology.pdf",    ← optional filter
            "n_results": 10             ← optional, default 10
        }

    WHY A SEPARATE SEARCH ACTION (not just chat)?

        Search returns RAW chunks from ChromaDB — the most
        relevant text fragments ranked by semantic similarity.
        No LLM is involved, so it's:
          1. FAST — milliseconds, not seconds
          2. FREE — no API credits consumed
          3. TRANSPARENT — shows exactly what the system found

        Chat uses search internally, then sends the results
        to the LLM for a human-friendly answer. But sometimes
        a wrapper just wants the raw data (e.g., to build its
        own UI for search results).
    """

    query = payload.get("query")

    if not query:
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                "Missing required field: 'query'",
            )
        )
        return

    source = payload.get("source")
    n_results = payload.get("n_results", 10)

    try:
        # search_documents returns a list of Evidence dataclass
        # objects. We convert them to plain dicts so they can
        # be serialized to JSON for the WebSocket response.
        #
        # Evidence has: text, source, page, score
        evidence_list = await _run_in_executor(
            None,
            lambda: search_documents(
                query=query,
                source_filter=source,
                n_results=n_results,
            ),
        )

        # Convert Evidence dataclasses → JSON-safe dicts
        results = [
            {
                "text": ev.text,
                "source": ev.source,
                "page": ev.page,
                "score": round(ev.score, 4),
            }
            for ev in evidence_list
        ]

        await websocket.send_json(
            make_success(msg_id, {
                "results": results,
                "count": len(results),
                "query": query,
            })
        )

    except Exception as exc:

        logger.error(
            "WS search failed: %s",
            exc,
            exc_info=True,
        )

        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INTERNAL_ERROR,
                f"Search failed: {exc}",
            )
        )


# ---------------------------------------------------------
# CHAT handler (with streaming)
# ---------------------------------------------------------

async def handle_chat(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """Ask a question with LLM-generated answer.

    Expected payload:
        {
            "question": "Explain photosynthesis",
            "source": "biology.pdf",       ← optional filter
            "session_id": "sess_edu_123",  ← optional (for context)
            "stream": true                 ← true=stream, false=wait
        }

    STREAMING vs NON-STREAMING:

        stream: true  → sends tokens one by one as they arrive
                        from the LLM. The wrapper sees the AI
                        "typing" in real time.

        stream: false → waits for the full answer, then sends
                        it all at once. Simpler for the wrapper
                        to handle, but no real-time feedback.

    SESSION CONTEXT:

        If session_id is provided, the conversation history
        from that session is included in the LLM prompt. This
        lets the AI reference previous questions and answers
        in the same session — essential for follow-up questions
        like "tell me more about that" or "what about X?"
    """

    question = payload.get("question")

    if not question:
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                "Missing required field: 'question'",
            )
        )
        return

    request_id = make_request_id()
    source_filter = payload.get("source")
    session_id = payload.get("session_id", str(uuid.uuid4()))
    stream = payload.get("stream", True)
    conn_id = connection_info.get("connection_id", "")

    logger.info(
        "[%s] WS chat | q=%s | session=%s | stream=%s "
        "| app=%s",
        request_id,
        question[:80],
        session_id[:8] + "...",
        stream,
        connection_info.get("app_name", "unknown"),
    )

    # Ensure session exists
    session = get_session(session_id)
    if session is None:
        create_session(
            session_id=session_id,
            title="WebSocket Chat",
            source_filter=source_filter,
        )

    # Save user message
    save_message(
        session_id=session_id,
        role="user",
        content=question,
    )

    # Load conversation history
    messages = get_messages_for_llm(
        session_id,
        limit=MAX_MESSAGES_PER_SESSION,
    )

    # -------------------------------------------------
    # Set up cancellation for chat.stop support
    # -------------------------------------------------

    cancel_event = asyncio.Event()

    if conn_id not in active_chats:
        active_chats[conn_id] = {}
    active_chats[conn_id][msg_id] = cancel_event

    try:

        if stream:
            # -------------------------------------------
            # STREAMING MODE
            # -------------------------------------------
            #
            # stream_agent() is a generator that yields
            # events (dicts) one at a time. Each event has
            # a "type" field: "token", "completed", "error".
            #
            # We convert each event into a WebSocket message
            # and send it immediately. The wrapper sees tokens
            # arriving in real time.

            full_answer = ""
            completed_event = None

            # Run the blocking generator in a thread
            def _run_stream():
                return list(stream_agent(
                    messages,
                    request_id=request_id,
                    source_filter=source_filter,
                ))

            events = await _run_in_executor(
                None, _run_stream,
            )

            for event in events:

                # Check if chat.stop was requested
                if cancel_event.is_set():
                    logger.info(
                        "[%s] Chat stopped by client",
                        request_id,
                    )
                    break

                event_type = event.get("type")

                if event_type == "token":
                    content = event.get("content", "")
                    full_answer += content

                    await websocket.send_json(
                        make_streaming(msg_id, {
                            "type": "token",
                            "content": content,
                        })
                    )

                elif event_type == "completed":
                    completed_event = event

                elif event_type == "error":
                    await websocket.send_json(
                        make_error(
                            msg_id,
                            ErrorCode.LLM_ERROR,
                            event.get(
                                "message",
                                "LLM generation failed",
                            ),
                        )
                    )
                    return

            # Send the final "complete" message
            assistant_content = ""
            if completed_event:
                assistant_content = completed_event.get(
                    "assistant_content", full_answer,
                )
            else:
                assistant_content = full_answer

            # Save assistant reply to database
            if assistant_content:
                save_message(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_content,
                )

            # Get follow-up suggestions
            suggestions_list = []
            if completed_event:
                suggestions_list = completed_event.get(
                    "suggestions", []
                )

            await websocket.send_json(
                make_success(msg_id, {
                    "type": "complete",
                    "answer": assistant_content,
                    "session_id": session_id,
                    "suggestions": suggestions_list,
                    "request_id": request_id,
                })
            )

        else:
            # -------------------------------------------
            # NON-STREAMING MODE
            # -------------------------------------------
            #
            # Wait for the full answer, then send it all
            # at once. Simpler for wrappers that don't want
            # to handle token-by-token streaming.

            answer = await _run_in_executor(
                None,
                lambda: ask_agent(
                    messages,
                    request_id=request_id,
                    source_filter=source_filter,
                ),
            )

            # Save assistant reply
            if answer:
                save_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                )

            await websocket.send_json(
                make_success(msg_id, {
                    "type": "complete",
                    "answer": answer,
                    "session_id": session_id,
                    "request_id": request_id,
                })
            )

    except Exception as exc:

        logger.error(
            "[%s] WS chat failed: %s",
            request_id,
            exc,
            exc_info=True,
        )

        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.LLM_ERROR,
                f"Chat failed: {exc}",
            )
        )

    finally:
        # Clean up the cancel event
        if conn_id in active_chats:
            active_chats[conn_id].pop(msg_id, None)
            if not active_chats[conn_id]:
                del active_chats[conn_id]


# ---------------------------------------------------------
# CHAT.STOP handler
# ---------------------------------------------------------

async def handle_chat_stop(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """Stop an in-progress LLM generation.

    Expected payload:
        {"target_id": "msg_005"}   ← the chat message to stop

    WHY THIS IS BETTER THAN HTTP:

        With the current HTTP streaming (SSE), stopping
        generation requires a SEPARATE HTTP request to a
        different endpoint. The browser has to:
          1. Remember which request to stop
          2. Send a POST to /chat/stop
          3. Hope it arrives before the stream finishes

        With WebSocket, the stop command goes through the
        SAME connection as the chat. It's simpler, faster,
        and more reliable.
    """

    target_id = payload.get("target_id")

    if not target_id:
        await websocket.send_json(
            make_error(
                msg_id,
                ErrorCode.INVALID_PAYLOAD,
                "Missing required field: 'target_id'",
            )
        )
        return

    conn_id = connection_info.get("connection_id", "")

    # Look up the active chat and set its cancel event
    if conn_id in active_chats and target_id in active_chats[conn_id]:
        active_chats[conn_id][target_id].set()

        await websocket.send_json(
            make_success(msg_id, {
                "message": "Generation stopped",
                "target_id": target_id,
            })
        )

        logger.info(
            "WS chat.stop | target=%s | app=%s",
            target_id,
            connection_info.get("app_name", "unknown"),
        )
    else:
        # Not found — either already finished or invalid ID
        await websocket.send_json(
            make_success(msg_id, {
                "message": (
                    "No active generation found for "
                    f"'{target_id}' (may have already finished)"
                ),
                "target_id": target_id,
            })
        )


# ---------------------------------------------------------
# SUGGESTIONS handler
# ---------------------------------------------------------

async def handle_suggestions(
    websocket: WebSocket,
    msg_id: str,
    payload: dict,
    connection_info: dict,
) -> None:
    """Get pre-generated starter questions for a document.

    Expected payload:
        {"source": "biology.pdf"}

    Returns the same data as GET /suggestions. These are
    AI-generated questions that help users get started
    when they don't know what to ask about a document.
    """

    source = payload.get("source", "")

    if not source:
        await websocket.send_json(
            make_success(msg_id, {
                "suggestions": [],
            })
        )
        return

    suggestions_list = document_suggestions.get(scoped_key(source), [])

    await websocket.send_json(
        make_success(msg_id, {
            "suggestions": suggestions_list,
            "source": source,
        })
    )


# ---------------------------------------------------------
# ACTION HANDLER REGISTRY
# ---------------------------------------------------------
#
# This dict maps action names to handler functions. The
# gateway imports this and uses it to route messages:
#
#   handler = ACTION_HANDLERS.get(message.action)
#   if handler:
#       await handler(websocket, msg_id, payload, info)
#
# WHY A DICT (not if/elif)?
#
#   1. EXTENSIBLE — adding a new action is ONE line here
#   2. READABLE — you can see ALL actions at a glance
#   3. NO ELSE — if the action isn't in the dict, gateway.py
#      sends an INVALID_ACTION error. No forgotten else clause.
#   4. TESTABLE — you can iterate over the dict in tests to
#      ensure every action has a handler

ACTION_HANDLERS = {
    "ping":            handle_ping,
    "document.list":   handle_document_list,
    "document.upload": handle_document_upload,
    "document.delete": handle_document_delete,
    "search":          handle_search,
    "chat":            handle_chat,
    "chat.stop":       handle_chat_stop,
    "suggestions":     handle_suggestions,
}
