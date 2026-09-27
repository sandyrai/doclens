# ---------------------------------------------------------
# middleware/auth.py — App Authentication (Phase 3)
# ---------------------------------------------------------
#
# WHAT THIS MODULE DOES:
#
#   Provides authentication for WRAPPER APPLICATIONS that
#   connect to DocAgent. It does NOT authenticate end-users
#   (students, teachers, etc.) — each wrapper handles its
#   own user authentication.
#
#   Think of it like a building's key card system:
#     - DocAgent checks: "Is this a registered company?"
#     - The company (wrapper) checks: "Is this employee
#       allowed in?"
#
# TWO AUTHENTICATION PATHS:
#
#   1. WEB UI (DocAgent's own frontend):
#      No authentication needed. Anonymous users are
#      throttled by IP-based rate limits (Phase 1).
#      This keeps the web UI simple — no login required.
#
#   2. WRAPPER APPS (via WebSocket, Phase 4):
#      Each wrapper sends an API key during the WebSocket
#      handshake. The key is validated against stored
#      hashes in the database.
#
#      ws://docagent.example.com/ws?api_key=dak_7f3a9b2c...
#
# WHY API KEYS (not OAuth / JWT)?
#
#   API keys are the simplest auth mechanism for server-to-
#   server communication:
#     - No token refresh flow needed
#     - No authorization code exchange
#     - No third-party auth provider dependency
#     - One key per app — easy to create and revoke
#
#   OAuth/JWT is designed for USER authentication (login
#   flows, token refresh, scopes). We're authenticating
#   APPLICATIONS, not users — a much simpler problem.
#
# SECURITY LAYERS:
#
#   1. Key is transmitted over TLS (wss://) — encrypted
#   2. Key is stored as SHA-256 hash — database leak is safe
#   3. Keys can be revoked instantly — one CLI command
#   4. Each connection validates the key — no session reuse
#   5. last_seen tracking — detect unused/orphaned keys
# ---------------------------------------------------------

import hashlib
import logging
from typing import Optional

from ai_document_agent.database import (
    get_api_client_by_key_hash,
    update_api_client_last_seen,
)

logger = logging.getLogger(__name__)


def validate_api_key(api_key: str) -> Optional[dict]:
    """Validate an API key and return the client info.

    This is the core authentication function. It will be
    called during the WebSocket handshake (Phase 4) to
    verify the connecting wrapper app.

    HOW IT WORKS:

        1. Hash the provided key with SHA-256
        2. Look up the hash in the api_clients table
        3. If found AND active → return client info
        4. If found but revoked → return None (log warning)
        5. If not found → return None (log warning)

    WHY HASH BEFORE LOOKUP?

        We never store raw API keys. The database only has
        SHA-256 hashes. So we hash the incoming key and
        search for the matching hash. This is the same
        pattern used by GitHub, Stripe, and most API key
        systems.

    TIMING ATTACK PREVENTION:

        A true production system would use hmac.compare_digest()
        for constant-time comparison. For our use case (keys
        are 64+ hex chars with 256 bits of entropy), timing
        attacks aren't practical — but it's worth knowing
        about for future reference.

    Args:
        api_key: The raw API key (e.g., "dak_7f3a9b2c...")
                 provided by the connecting wrapper app.

    Returns:
        Client dict (id, name, is_active, etc.) if valid,
        None if invalid or revoked.
    """

    if not api_key:
        logger.warning("Empty API key provided")
        return None

    # -------------------------------------------------
    # Step 1: Validate the key format
    # -------------------------------------------------
    #
    # Keys must start with "dak_" (DocAgent Key prefix).
    # This quick check rejects obviously wrong values
    # before hitting the database.

    if not api_key.startswith("dak_"):
        logger.warning(
            "Invalid API key format (missing dak_ prefix)"
        )
        return None

    # -------------------------------------------------
    # Step 2: Hash the key
    # -------------------------------------------------
    #
    # Same hash function used when creating the key
    # (in manage_keys.py). SHA-256 always produces the
    # same hash for the same input, so:
    #   hash("dak_abc123") at creation time
    #   == hash("dak_abc123") at validation time

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # -------------------------------------------------
    # Step 3: Look up the hash in the database
    # -------------------------------------------------

    client = get_api_client_by_key_hash(key_hash)

    if client is None:
        logger.warning(
            "API key not found (hash=%s...)",
            key_hash[:12],
        )
        return None

    # -------------------------------------------------
    # Step 4: Check if the key is still active
    # -------------------------------------------------
    #
    # A revoked key still exists in the database (for
    # audit), but is_active = 0. We reject it.

    if not client["is_active"]:
        logger.warning(
            "Revoked API key used by '%s' (id=%s)",
            client["name"],
            client["id"],
        )
        return None

    # -------------------------------------------------
    # Step 5: Update last_seen and return
    # -------------------------------------------------
    #
    # Track when each app last connected. This helps
    # identify unused keys that should be cleaned up.

    update_api_client_last_seen(client["id"])

    logger.info(
        "API key validated for '%s' (id=%s)",
        client["name"],
        client["id"],
    )

    return client


# ---------------------------------------------------------
# FastAPI dependency (for future HTTP route protection)
# ---------------------------------------------------------
#
# WHY DEFINE THIS NOW?
#
#   Even though the WebSocket gateway (Phase 4) will use
#   validate_api_key() directly, having a FastAPI Depends()
#   ready means we can easily protect HTTP routes too:
#
#     @router.post("/admin/something")
#     def admin_action(app = Depends(require_api_key)):
#         # 'app' is the authenticated client dict
#         ...
#
#   This isn't used yet, but it's ready for Phase 4/5 when
#   we might want to protect certain HTTP endpoints for
#   wrapper apps (e.g., a bulk upload API).

# from fastapi import Depends, HTTPException, Query
#
# async def require_api_key(
#     api_key: str = Query(..., alias="api_key"),
# ) -> dict:
#     """FastAPI dependency that requires a valid API key.
#
#     Usage:
#         @router.get("/protected")
#         def endpoint(app = Depends(require_api_key)):
#             print(f"Request from {app['name']}")
#
#     The API key is expected as a query parameter:
#         GET /protected?api_key=dak_7f3a9b2c...
#     """
#
#     client = validate_api_key(api_key)
#
#     if client is None:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid or revoked API key",
#         )
#
#     return client
