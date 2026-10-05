# ---------------------------------------------------------
# routes/upload.py — File upload & processing (Phase 5)
# ---------------------------------------------------------
#
# WHAT'S IN THIS FILE:
#
#   POST /upload              → Accept a file upload
#   GET  /upload/status/{id}  → Poll processing progress
#   _process_in_background()  → Background processing logic
#
# WHAT CHANGED IN PHASE 5:
#
#   Upload tasks moved from in-memory dict (shared.py) to
#   SQLite database (database.py). This means:
#     1. Tasks survive server restarts
#     2. If the server crashes during processing, the task
#        is marked as "failed" on the next startup
#     3. No more stale-task cleanup in Python — SQL handles it
#
#   The in-memory upload_tasks dict in shared.py is STILL
#   updated for backward compatibility with WebSocket handlers
#   (which read from it). Both stores are kept in sync.
# ---------------------------------------------------------

import hashlib
import logging
import uuid
from pathlib import Path
from time import time as _time

from fastapi import APIRouter, BackgroundTasks, Request, UploadFile
from fastapi.responses import JSONResponse

from ai_document_agent.agent import generate_document_suggestions
from ai_document_agent.database import (
    create_upload_task,
    get_upload_task,
    update_upload_task,
)
from ai_document_agent.pdf_processor import (
    SUPPORTED_EXTENSIONS,
    list_documents,
    process_document,
)
from ai_document_agent.query_cache import invalidate_cache
from ai_document_agent.rate_limiter import (
    check_and_increment_upload,
)
from ai_document_agent.shared import (
    MAX_UPLOAD_BYTES,
    document_suggestions,
    get_client_ip,
    sanitize_filename,
    upload_tasks,
)
from ai_document_agent.tenancy import (
    VISITOR_ISOLATION,
    current_visitor,
    document_hash_salt,
    purge_expired_visitor_files,
    scoped_key,
    visitor_upload_dir,
)
from ai_document_agent.vector_store import purge_expired_chunks

router = APIRouter(tags=["upload"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Background document processing function
# ---------------------------------------------------------
#
# This runs in a background thread (via FastAPI's
# BackgroundTasks) AFTER the upload endpoint has already
# returned a response to the browser.
#
# It updates upload_tasks[task_id] at each stage so the
# browser can poll GET /upload/status/{task_id} and show
# real-time progress to the user.
#
# Stages (in order):
#   1. extracting — reading text from the PDF/DOCX/etc.
#   2. processing — chunking + embedding + storing
#   3. done       — processing complete, document is ready
#
# If any stage fails, status becomes "error" and the
# error message is stored for the browser to display.

def _process_in_background(
    task_id: str,
    save_path: str,
    safe_filename: str,
    document_id: str,
    request_id: str,
) -> None:
    """Process a document in the background.

    This function is called by FastAPI's BackgroundTasks.
    It updates the upload_tasks dict at each stage so
    the browser can show progress via polling.

    Args:
        task_id: Unique task ID for progress tracking.
        save_path: Path where the file was saved on disk.
        safe_filename: Sanitized original filename.
        document_id: SHA-256-based document ID.
        request_id: Request ID for log correlation.
    """

    try:
        # Stage 1: Extracting text
        # Update BOTH in-memory dict AND database
        upload_tasks[task_id]["stage"] = "extracting"
        upload_tasks[task_id]["progress_pct"] = 20
        update_upload_task(
            task_id, stage="extracting", progress_pct=20,
        )

        logger.info(
            "[%s] Background: extracting text from %s",
            request_id,
            safe_filename,
        )

        upload_tasks[task_id]["stage"] = "processing"
        upload_tasks[task_id]["progress_pct"] = 40
        update_upload_task(
            task_id, stage="processing", progress_pct=40,
        )

        result = process_document(
            file_path=save_path,
            filename=safe_filename,
            document_id=document_id,
        )

        # Check if processing succeeded
        if result.get("status") == "error":
            err_msg = result.get(
                "message", "Processing failed"
            )

            upload_tasks[task_id]["status"] = "error"
            upload_tasks[task_id]["stage"] = "failed"
            upload_tasks[task_id]["progress_pct"] = 0
            upload_tasks[task_id]["error"] = err_msg
            update_upload_task(
                task_id, status="error", stage="failed",
                progress_pct=0, error=err_msg,
            )

            logger.error(
                "[%s] Background processing failed: %s",
                request_id,
                err_msg,
            )
            return

        # -------------------------------------------------
        # Invalidate the semantic query cache
        # -------------------------------------------------
        #
        # A new document was added — cached answers based
        # on the OLD document set are now stale.

        cleared = invalidate_cache()
        if cleared > 0:
            logger.info(
                "[%s] Cleared %d cached answers after "
                "new document upload",
                request_id,
                cleared,
            )

        # Expire other visitors' old documents. Cheap (an
        # indexed DELETE + a folder scan) and keeps retention
        # working without a separate scheduler.
        if VISITOR_ISOLATION:
            try:
                purge_expired_chunks()
                purge_expired_visitor_files()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Visitor purge failed: %s",
                    request_id, exc,
                )

        # Stage complete: document is ready!
        result_data = {
            "document_id": document_id,
            "filename": safe_filename,
            "pages": result.get("pages", 0),
            "chunks": result.get("chunks", 0),
            "status": "success",
            "request_id": request_id,
            "summary": (
                f"Document indexed successfully: "
                f"{result.get('pages', 0)} pages, "
                f"{result.get('chunks', 0)} chunks. "
                f"Ask me anything about this document!"
            ),
            "insights": [],
        }

        upload_tasks[task_id]["status"] = "ready"
        upload_tasks[task_id]["stage"] = "done"
        upload_tasks[task_id]["progress_pct"] = 100
        upload_tasks[task_id]["result"] = result_data
        update_upload_task(
            task_id, status="ready", stage="done",
            progress_pct=100, result=result_data,
        )

        logger.info(
            "[%s] Background complete: %d pages, %d chunks",
            request_id,
            result.get("pages", 0),
            result.get("chunks", 0),
        )

        # -------------------------------------------------
        # Generate smart suggestions (Phase 4)
        # -------------------------------------------------

        try:
            doc_text = result.get("text", "")

            if doc_text:
                suggestions = generate_document_suggestions(
                    document_text=doc_text,
                    filename=safe_filename,
                )

                # Store in shared dict — suggestions.py
                # reads from this same dict
                document_suggestions[scoped_key(safe_filename)] = suggestions

                logger.info(
                    "[%s] Generated %d suggestions for '%s'",
                    request_id,
                    len(suggestions),
                    safe_filename,
                )
            else:
                logger.warning(
                    "[%s] No text extracted — skipping "
                    "suggestion generation for '%s'",
                    request_id,
                    safe_filename,
                )
        except Exception as e:
            # Suggestions are nice-to-have, not critical
            logger.warning(
                "[%s] Suggestion generation failed for "
                "'%s': %s",
                request_id,
                safe_filename,
                e,
            )

    except Exception as exc:

        upload_tasks[task_id]["status"] = "error"
        upload_tasks[task_id]["stage"] = "failed"
        upload_tasks[task_id]["progress_pct"] = 0
        upload_tasks[task_id]["error"] = str(exc)
        update_upload_task(
            task_id, status="error", stage="failed",
            progress_pct=0, error=str(exc),
        )

        logger.error(
            "[%s] Background processing exception: %s",
            request_id,
            exc,
            exc_info=True,
        )


# ---------------------------------------------------------
# POST /upload — Accept a file upload
# ---------------------------------------------------------

@router.post("/upload")
def upload_pdf(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Upload a document for AI analysis.

    Accepts PDF, DOCX, TXT, CSV, and image files.
    Returns immediately with a task_id — the browser
    polls /upload/status/{task_id} for progress.

    WHY 'request: Request' HERE?

        Unlike the chat endpoints, /upload uses UploadFile
        (not a Pydantic model) for the file body. So there's
        no naming conflict — 'request' can be the FastAPI
        Request object directly. We need it for the client
        IP (rate limiting).
    """

    # Request ID from middleware (Phase 5)
    request_id = request.state.request_id

    # -------------------------------------------------
    # Rate limit check (Phase 1)
    # -------------------------------------------------

    client_ip = get_client_ip(request)
    upload_check = check_and_increment_upload(client_ip)

    if not upload_check["allowed"]:
        logger.warning(
            "[%s] Upload rate limited for IP %s "
            "(%d/%d today)",
            request_id,
            client_ip,
            upload_check["used"],
            upload_check["limit"],
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": upload_check["message"],
                "usage": {
                    "uploads_used": upload_check["used"],
                    "uploads_limit": upload_check["limit"],
                },
                "request_id": request_id,
            },
        )

    # Sanitize the filename to prevent path traversal
    safe_filename = sanitize_filename(
        file.filename or "unnamed",
    )

    logger.info(
        "[%s] POST /upload | file=%s | size=%s | ip=%s",
        request_id,
        safe_filename,
        file.size,
        client_ip,
    )

    # Validate file type
    file_ext = Path(safe_filename).suffix.lower()

    if file_ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Unsupported file type: "
                    f"{safe_filename}. "
                    f"Supported: {supported}"
                ),
                "request_id": request_id,
            },
        )

    try:

        # File size check
        file_bytes = file.file.read()

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            actual_mb = len(file_bytes) / (1024 * 1024)
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"File too large: "
                        f"{actual_mb:.1f} MB. "
                        f"Maximum: {max_mb} MB."
                    ),
                    "request_id": request_id,
                },
            )

        # SHA-256 content-based document ID
        # Salted per visitor, so two visitors uploading the
        # same file get separate documents (and separate
        # image folders / table CSVs).
        document_id = hashlib.sha256(
            document_hash_salt() + file_bytes
        ).hexdigest()[:12]

        # Check for duplicate
        existing_docs = list_documents()
        for doc in existing_docs:
            if doc["document_id"] == document_id:
                logger.info(
                    "[%s] Duplicate detected: '%s' "
                    "matches existing '%s' (id=%s)",
                    request_id,
                    safe_filename,
                    doc["filename"],
                    document_id,
                )

                return {
                    "document_id": document_id,
                    "filename": doc["filename"],
                    "pages": doc["pages"],
                    "chunks": doc["chunks"],
                    "status": "duplicate",
                    "request_id": request_id,
                    "summary": (
                        f"This file was already uploaded "
                        f"as '{doc['filename']}'. "
                        f"No duplicate created."
                    ),
                    "insights": [],
                }

        # Save file to disk
        save_path = visitor_upload_dir() / f"{document_id}_{safe_filename}"

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        logger.info(
            "[%s] Saved to: %s (hash=%s)",
            request_id,
            save_path,
            document_id,
        )

        # Create task and start background processing
        # Full 128-bit ID: the status endpoint has no other
        # access check, so task IDs must be unguessable.
        task_id = "task_" + uuid.uuid4().hex

        # Phase 5: Save to database (survives restarts)
        create_upload_task(
            task_id=task_id,
            filename=safe_filename,
            document_id=document_id,
            request_id=request_id,
        )

        # Also keep in-memory dict for fast polling
        # and backward compatibility with WebSocket handlers
        upload_tasks[task_id] = {
            "task_id": task_id,
            "visitor": current_visitor(),
            "status": "processing",
            "stage": "saving",
            "filename": safe_filename,
            "document_id": document_id,
            "request_id": request_id,
            "created_at": _time(),
            "progress_pct": 10,
            "result": None,
            "error": None,
        }

        background_tasks.add_task(
            _process_in_background,
            task_id=task_id,
            save_path=str(save_path),
            safe_filename=safe_filename,
            document_id=document_id,
            request_id=request_id,
        )

        logger.info(
            "[%s] Upload accepted, processing in background "
            "(task=%s)",
            request_id,
            task_id,
        )

        # Return IMMEDIATELY — browser polls for progress
        return {
            "task_id": task_id,
            "document_id": document_id,
            "filename": safe_filename,
            "status": "processing",
            "request_id": request_id,
        }

    except Exception as exc:

        logger.error(
            "[%s] Upload failed: %s",
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
# GET /upload/status/{task_id} — Poll processing progress
# ---------------------------------------------------------

@router.get("/upload/status/{task_id}")
def get_upload_status(task_id: str):
    """Poll the status of a background upload task.

    Called by the browser every 2 seconds after file upload
    to show real-time progress to the user.

    WHAT CHANGED IN PHASE 5:

        Now checks TWO places for task status:

        1. In-memory dict (fast, updated in real-time during
           processing). This is the primary source while
           the server is running.

        2. Database (persistent, survives restarts). This is
           the fallback if the dict doesn't have the task —
           which happens after a server restart when a task
           was created before the restart.

        This two-tier lookup gives us the best of both:
        real-time updates during processing AND durability
        across restarts.
    """

    # -------------------------------------------------
    # Tier 1: Check in-memory dict (fast path)
    # -------------------------------------------------

    task = upload_tasks.get(task_id)
    foreign = (
        task is not None
        and task.get("visitor", current_visitor()) != current_visitor()
    )

    # -------------------------------------------------
    # Tier 2: Fall back to database (after restart)
    # -------------------------------------------------

    if task is None:
        task = get_upload_task(task_id)

    if task is None or foreign:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Unknown task: {task_id}",
                "status": "not_found",
            },
        )

    # Build response based on current status
    response = {
        "task_id": task.get("task_id", task_id),
        "status": task["status"],
        "stage": task["stage"],
        "filename": task["filename"],
        "progress_pct": task.get("progress_pct", 0),
    }

    if task["status"] == "ready" and task.get("result"):
        response["result"] = task["result"]

    if task["status"] == "error" and task.get("error"):
        response["error"] = task["error"]

    return response
