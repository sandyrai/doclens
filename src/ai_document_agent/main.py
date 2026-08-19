import hashlib
import json
import logging
import re
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ai_document_agent.agent import (
    ask_agent,
    stream_agent,
)
from ai_document_agent.pdf_processor import (
    SUPPORTED_EXTENSIONS,
    delete_document,
    list_documents,
    process_document,
)


# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
#
# Why use logging instead of print()?
#
# 1. TIMESTAMPS — every log line shows when it happened.
# 2. LOG LEVELS — INFO for normal flow, ERROR for
#    failures, WARNING for suspicious but not broken.
# 3. SOURCE — shows which module the log came from.
# 4. FILTERING — you can turn off noisy modules without
#    changing code.
# 5. PRODUCTION-READY — in production you'd send logs
#    to a file, Datadog, CloudWatch, etc. print() can't
#    do that.
#
# Format example:
#   2026-08-18 14:23:05 INFO  main  [req_abc123] ...

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)-5s %(name)s  "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# App setup
# ---------------------------------------------------------

app = FastAPI(
    title="AI Document Agent",
    description="Local AI agent using Ollama",
    version="0.1.0",
)


BASE_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------
# Upload directory
# ---------------------------------------------------------
#
# Where uploaded PDFs are temporarily stored before
# processing. After extraction and embedding, the PDF
# stays on disk so you could re-process it later.

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount(
    "/static",
    StaticFiles(
        directory=BASE_DIR / "static"
    ),
    name="static",
)


# ---------------------------------------------------------
# Session store
# ---------------------------------------------------------

sessions: dict[str, list[dict]] = {}

MAX_MESSAGES_PER_SESSION = 50


def get_session_messages(
    session_id: str,
) -> list[dict]:
    """Get or create a session's message history."""

    if session_id not in sessions:
        sessions[session_id] = []

    return sessions[session_id]


def trim_session(messages: list[dict]) -> None:
    """Keep only the most recent messages."""

    while len(messages) > MAX_MESSAGES_PER_SESSION:
        messages.pop(0)


# ---------------------------------------------------------
# Request ID generation
# ---------------------------------------------------------
#
# Every API call gets a unique request_id like "req_a1b2".
# This ID appears in:
#   - every log line for this request
#   - every streaming event sent to the browser
#   - error responses
#
# Why?
#   When something goes wrong, you search your terminal
#   for the request_id and see the complete trace —
#   which session, what the user asked, which tool was
#   called, where it failed. Without this, debugging
#   concurrent requests is nearly impossible.

def make_request_id() -> str:
    """Generate a short, unique request ID."""

    return "req_" + uuid.uuid4().hex[:8]


# ---------------------------------------------------------
# Request / response models
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Browser session ID. If not provided, "
            "a new session is created."
        ),
    )
    source_filter: str | None = Field(
        default=None,
        description=(
            "Optional filename to scope retrieval "
            "to a single document. If None, searches "
            "all uploaded documents."
        ),
    )


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(
        BASE_DIR / "static" / "index.html"
    )


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat_endpoint(request: ChatRequest):

    request_id = make_request_id()

    logger.info(
        "[%s] POST /chat | session=%s | q=%s",
        request_id,
        request.session_id[:8] + "...",
        request.question[:80],
    )

    try:

        # 1. Get session history
        messages = get_session_messages(
            request.session_id
        )

        # 2. Add the new user message
        messages.append(
            {
                "role": "user",
                "content": request.question,
            }
        )

        # 3. Run the agent with full history
        answer = ask_agent(
            messages,
            request_id=request_id,
            source_filter=request.source_filter,
        )

        # 4. Add the assistant's reply to history
        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # 5. Trim if too long
        trim_session(messages)

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

        # Remove the user message we just added,
        # since the request failed. We don't want
        # a broken exchange in the history.

        if (
            messages
            and messages[-1].get("role") == "user"
        ):
            messages.pop()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "request_id": request_id,
            },
        )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    request_id = make_request_id()

    logger.info(
        "[%s] POST /chat/stream | session=%s | q=%s",
        request_id,
        request.session_id[:8] + "...",
        request.question[:80],
    )

    # Get session history and add user message
    messages = get_session_messages(
        request.session_id
    )

    messages.append(
        {
            "role": "user",
            "content": request.question,
        }
    )

    def generate():

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

                # When the stream completes, save
                # the assistant answer to history.
                if event.get("type") == "completed":

                    assistant_content = event.get(
                        "assistant_content", ""
                    )

                    if assistant_content:

                        messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    assistant_content
                                ),
                            }
                        )

                        trim_session(messages)

        except Exception as exc:

            # This catches unexpected errors that
            # weren't handled inside stream_agent.
            # For example, a network drop mid-stream
            # or an Ollama crash.

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

            # If there was an error and no assistant
            # reply was saved, remove the dangling
            # user message so the history stays clean.

            if had_error:

                if (
                    messages
                    and messages[-1].get("role")
                    == "user"
                ):
                    messages.pop()

                    logger.info(
                        "[%s] Removed dangling user "
                        "message after error",
                        request_id,
                    )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


# ---------------------------------------------------------
# PDF upload endpoint
# ---------------------------------------------------------
#
# How file uploads work in FastAPI:
#
# 1. The browser sends a multipart/form-data request
#    (not JSON). This is the standard way to send files
#    over HTTP.
#
# 2. FastAPI's UploadFile gives us a file-like object
#    with .filename, .read(), .file, etc.
#
# 3. We save the file to disk first, then process it.
#    Why not process from memory? Because PyMuPDF (fitz)
#    works best with file paths, and saving first means
#    we have a backup copy.
#
# 4. Processing = extract text → chunk → embed → store
#    in ChromaDB. This can take 10-60 seconds depending
#    on PDF size and CPU speed.

# ---------------------------------------------------------
# Upload security constants
# ---------------------------------------------------------

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _sanitize_filename(raw_name: str) -> str:
    """Sanitize an uploaded filename to prevent
    path traversal attacks.

    Strips directory components, replaces dangerous
    characters, and ensures a safe basename remains.
    """

    # Take only the final path component — kills
    # "../../etc/passwd" and "C:\\Windows\\system32\\x"
    name = Path(raw_name).name

    # Remove any remaining path separators
    name = name.replace("/", "_").replace("\\", "_")

    # Remove null bytes and other control characters
    name = re.sub(r'[\x00-\x1f]', '', name)

    # Collapse whitespace
    name = name.strip()

    if not name:
        name = "unnamed_upload"

    return name


@app.post("/upload")
def upload_pdf(file: UploadFile):

    request_id = make_request_id()

    # Sanitize the filename to prevent path traversal
    safe_filename = _sanitize_filename(
        file.filename or "unnamed",
    )

    logger.info(
        "[%s] POST /upload | file=%s | size=%s",
        request_id,
        safe_filename,
        file.size,
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

        # -------------------------------------------------
        # File size check (Phase 8.1)
        # -------------------------------------------------
        #
        # Prevent huge uploads from consuming all memory
        # and disk. We read the file bytes below anyway,
        # so checking length is free.

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

        # -------------------------------------------------
        # SHA-256 content-based document ID
        # -------------------------------------------------
        #
        # Why SHA-256 instead of random UUID?
        #
        #   With UUID: uploading report.pdf twice creates
        #   two separate document IDs → duplicate chunks in
        #   ChromaDB → wasted storage + search returns the
        #   same text twice from two "different" documents.
        #
        #   With SHA-256: the hash of file bytes is always
        #   the same for the same file content. Re-uploading
        #   the same PDF produces the same document_id, and
        #   ChromaDB's upsert overwrites the old chunks
        #   instead of duplicating them.
        #
        # We use the first 12 hex chars (48 bits) of the
        # SHA-256 hash. Collision probability is negligible
        # for a personal document tool (you'd need ~16
        # million PDFs for a 50% chance of one collision).

        document_id = hashlib.sha256(
            file_bytes
        ).hexdigest()[:12]

        # Check if this exact file was already uploaded
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

        # Save file to disk (using sanitized filename)
        save_path = UPLOAD_DIR / f"{document_id}_{safe_filename}"

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        logger.info(
            "[%s] Saved to: %s (hash=%s)",
            request_id,
            save_path,
            document_id,
        )

        # Process: extract → chunk → embed → store
        result = process_document(
            file_path=str(save_path),
            filename=safe_filename,
            document_id=document_id,
        )

        result["request_id"] = request_id

        logger.info(
            "[%s] Upload complete: %d pages, %d chunks",
            request_id,
            result.get("pages", 0),
            result.get("chunks", 0),
        )

        # -------------------------------------------------
        # No auto-summary — return immediately
        # -------------------------------------------------
        #
        # Previously we called generate_summary() here,
        # which added 30-60s of LLM processing on CPU
        # and also re-extracted the PDF (double work).
        #
        # Now we return right after indexing. The user
        # can ask for a summary in chat — the forced
        # retrieval will find the right chunks and the
        # LLM will summarize them.
        #
        # This cuts upload time from 60-90s to 5-15s.

        result["summary"] = (
            f"Document indexed successfully: "
            f"{result.get('pages', 0)} pages, "
            f"{result.get('chunks', 0)} chunks. "
            f"Ask me anything about this document!"
        )
        result["insights"] = []

        return result

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
# Document management endpoints
# ---------------------------------------------------------

@app.get("/documents")
def get_documents():
    """List all uploaded documents."""

    try:
        docs = list_documents()
        return {"documents": docs}

    except Exception as exc:

        logger.error(
            "Failed to list documents: %s",
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    """Delete a document and its chunks."""

    request_id = make_request_id()

    logger.info(
        "[%s] DELETE /documents/%s",
        request_id,
        document_id,
    )

    success = delete_document(document_id)

    if success:
        return {
            "status": "deleted",
            "document_id": document_id,
            "request_id": request_id,
        }

    return JSONResponse(
        status_code=404,
        content={
            "error": f"Document '{document_id}' not found.",
            "request_id": request_id,
        },
    )
