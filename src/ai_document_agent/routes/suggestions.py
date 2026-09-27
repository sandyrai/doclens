# ---------------------------------------------------------
# routes/suggestions.py — Smart suggestions (Phase 2)
# ---------------------------------------------------------
#
# WHAT'S IN THIS FILE:
#
#   GET /suggestions?source=report.pdf → Starter questions
#
# WHY ITS OWN FILE?
#
#   Even though it's just one endpoint, suggestions are a
#   distinct feature with their own data source (the
#   document_suggestions dict in shared.py). Keeping it
#   separate means:
#     1. Easy to find when debugging suggestion issues
#     2. Easy to extend (e.g., add POST /suggestions/refresh)
#     3. Clean separation — upload.py WRITES suggestions,
#        this file READS them
#
# DATA FLOW:
#
#   1. upload.py: _process_in_background() generates
#      suggestions and stores them in shared.document_suggestions
#   2. This file: GET /suggestions reads from that same dict
#   3. The dict lives in shared.py so both files can access
#      it without importing each other (no circular deps)
# ---------------------------------------------------------

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ai_document_agent.shared import document_suggestions

router = APIRouter(tags=["suggestions"])

logger = logging.getLogger(__name__)


@router.get("/suggestions")
def get_suggestions(source: str = ""):
    """Return pre-generated starter questions for a document.

    Query params:
        source — the filename to get suggestions for
                 (e.g., "report.pdf")

    Returns:
        {"suggestions": ["question1", "question2", ...]}

    If no suggestions exist for the given filename (either
    the document hasn't been processed yet, or suggestion
    generation failed), returns an empty list. This is
    intentional — the frontend simply doesn't show chips
    when there are no suggestions, no error needed.
    """

    if not source:
        return JSONResponse(
            content={"suggestions": []},
        )

    suggestions = document_suggestions.get(source, [])

    logger.info(
        "Suggestions request for '%s': %d found",
        source,
        len(suggestions),
    )

    return JSONResponse(
        content={"suggestions": suggestions},
    )
