# ---------------------------------------------------------
# routes/documents.py — Document management (Phase 2)
# ---------------------------------------------------------
#
# WHAT'S IN THIS FILE:
#
#   GET    /documents                         → List all docs
#   DELETE /documents/{id}                    → Delete a doc
#   GET    /documents/{id}/images             → List images
#   GET    /documents/{id}/images/{filename}  → Serve image
#
# WHY GROUP THESE TOGETHER?
#
#   These all deal with the document lifecycle AFTER upload:
#   listing, deleting, and accessing extracted images. They
#   share the same path prefix (/documents) and the same
#   underlying data (ChromaDB collection + disk files).
#
#   Upload is separate because it has its own complex flow
#   (background processing, task polling).
# ---------------------------------------------------------

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from ai_document_agent.pdf_processor import (
    delete_document,
    list_document_images,
    list_documents,
)
from ai_document_agent.query_cache import invalidate_cache
from ai_document_agent.shared import make_request_id
from ai_document_agent.tenancy import visitor_images_dir

router = APIRouter(tags=["documents"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# GET /documents — List all uploaded documents
# ---------------------------------------------------------

@router.get("/documents")
def get_documents():
    """List all uploaded documents.

    Returns document metadata from ChromaDB: filename,
    document_id, page count, chunk count. The frontend
    uses this to populate the sidebar document list
    and the document switcher chips.
    """

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


# ---------------------------------------------------------
# DELETE /documents/{document_id} — Remove a document
# ---------------------------------------------------------

@router.delete("/documents/{document_id}")
def remove_document(document_id: str):
    """Delete a document and its chunks from ChromaDB.

    Also invalidates the semantic query cache, because
    cached answers may reference content from the deleted
    document.
    """

    request_id = make_request_id()

    logger.info(
        "[%s] DELETE /documents/%s",
        request_id,
        document_id,
    )

    success = delete_document(document_id)

    if success:

        # Invalidate cache — answers referencing the
        # deleted document are now stale
        cleared = invalidate_cache()
        if cleared > 0:
            logger.info(
                "[%s] Cleared %d cached answers after "
                "document deletion",
                request_id,
                cleared,
            )

        return {
            "status": "deleted",
            "document_id": document_id,
            "request_id": request_id,
        }

    return JSONResponse(
        status_code=404,
        content={
            "error": (
                f"Document '{document_id}' not found."
            ),
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------
# Image endpoints (Phase 8)
# ---------------------------------------------------------
#
# These endpoints serve images extracted from uploaded PDFs.
# Images are saved to disk during document processing and
# served directly as files for browser caching + lazy loading.

@router.get("/documents/{document_id}/images")
def get_document_images(document_id: str):
    """List all extracted images for a document.

    Returns metadata about each image: filename, page
    number, size, and format. The frontend uses this to
    build the image gallery thumbnails.
    """

    try:
        images = list_document_images(document_id)

        return {
            "document_id": document_id,
            "images": images,
            "count": len(images),
        }

    except Exception as exc:

        logger.error(
            "Failed to list images for %s: %s",
            document_id,
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


@router.get("/documents/{document_id}/images/{filename}")
def get_document_image(
    document_id: str,
    filename: str,
):
    """Serve a single extracted image file.

    Used as the `src` attribute in <img> tags. Security:
    filename is validated to prevent path traversal.
    """

    # Sanitize filename to prevent path traversal attacks
    safe_name = Path(filename).name

    # document_id is a hex content hash. Rejecting anything
    # else also blocks "..", which would otherwise step out
    # of the images folder into the raw uploads.
    if (
        safe_name != filename
        or ".." in filename
        or not document_id.isalnum()
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid filename."},
        )

    img_path = visitor_images_dir() / document_id / safe_name

    if not img_path.exists() or not img_path.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    f"Image '{filename}' not found "
                    f"for document '{document_id}'."
                ),
            },
        )

    # Determine MIME type from file extension
    ext = img_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/png")

    return FileResponse(
        path=str(img_path),
        media_type=media_type,
        filename=safe_name,
    )
