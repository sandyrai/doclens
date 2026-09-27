# ---------------------------------------------------------
# websocket/gateway.py — WebSocket Gateway (Phase 5)
# ---------------------------------------------------------
#
# WHAT THIS FILE DOES:
#
#   Manages the lifecycle of WebSocket connections from
#   wrapper applications. Think of it as the "front door"
#   to DocLens's real-time API:
#
#     1. CONNECT — wrapper opens ws://host/ws?api_key=dak_xxx
#     2. AUTHENTICATE — validate the API key (Phase 3)
#     3. MESSAGE LOOP — receive messages, route to handlers
#     4. HEARTBEAT — server pings every 30s (Phase 5)
#     5. DISCONNECT — clean up when the connection closes
#
# HOW WEBSOCKET DIFFERS FROM HTTP:
#
#   HTTP is stateless — each request is independent. The
#   server doesn't "remember" previous requests.
#
#   WebSocket is stateful — the connection stays open, and
#   the server tracks each connected client. This is why
#   we need a ConnectionManager to keep track of who's
#   connected.
#
#   Analogy:
#     HTTP  = sending letters (each one is independent)
#     WS    = phone call (ongoing conversation)
#
# WHY FastAPI's NATIVE WebSocket SUPPORT?
#
#   FastAPI (via Starlette) has built-in WebSocket support.
#   No extra libraries needed — no Socket.IO, no channels.
#   It uses Python's async/await for concurrent connections.
#
#   Each WebSocket connection gets its own coroutine (async
#   function), so multiple wrappers can be connected at the
#   same time without blocking each other. This is handled
#   by Python's asyncio event loop.
#
# WHAT CHANGED IN PHASE 5:
#
#   1. HEARTBEAT (ping/pong every 30s)
#      The server periodically sends WebSocket protocol-level
#      pings. If a client doesn't respond with a pong within
#      10 seconds, the connection is considered dead and is
#      closed. This detects "zombie" connections — cases where
#      the TCP socket is open but the client is gone (laptop
#      closed, process killed, network cable unplugged).
#
#   2. GRACEFUL SHUTDOWN
#      When the server shuts down (SIGTERM, Ctrl+C), it sends
#      a close frame with code 1001 ("going away") to all
#      connected wrappers. This tells them "I'm shutting down
#      intentionally" vs "I crashed". Smart wrappers can then
#      wait a few seconds and reconnect, rather than retrying
#      immediately against a server that isn't there yet.
# ---------------------------------------------------------

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ai_document_agent.middleware.auth import validate_api_key
from ai_document_agent.websocket.handlers import (
    ACTION_HANDLERS,
    active_chats,
)
from ai_document_agent.websocket.protocol import (
    ErrorCode,
    WSRequest,
    make_error,
    make_success,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


# ---------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------
#
# WHY A CLASS (not module-level dicts)?
#
#   Grouping connection state and operations in a class
#   makes the code self-documenting:
#     - manager.connect()    → add a new connection
#     - manager.disconnect() → remove a connection
#     - manager.get_info()   → who is this connection?
#
#   It also prevents accidentally modifying the dicts
#   from outside — all mutations go through methods with
#   proper logging and cleanup.
#
# IS THIS THREAD-SAFE?
#
#   Yes, for our use case. FastAPI's async handlers run on
#   a single thread (the asyncio event loop). Multiple
#   connections are CONCURRENT (interleaved), not PARALLEL
#   (simultaneous). Dict operations in Python's asyncio
#   are safe because only one coroutine runs at a time
#   between await points.
#
#   If we ever needed true parallelism (multiple event
#   loops), we'd switch to asyncio.Lock. But for a single-
#   server setup, this is perfectly fine.

class ConnectionManager:
    """Manages active WebSocket connections.

    Tracks which wrapper apps are connected and provides
    methods to:
      - Register new connections (after auth)
      - Send messages to specific connections
      - Clean up when connections close
      - List active connections (for admin/health checks)
      - Gracefully close all connections on shutdown (Phase 5)
    """

    def __init__(self):
        # -------------------------------------------------
        # Map: connection_id → WebSocket instance
        # -------------------------------------------------
        # The WebSocket object represents the live connection.
        # We need it to send messages back to the wrapper.

        self.active_connections: dict[str, WebSocket] = {}

        # -------------------------------------------------
        # Map: connection_id → connection metadata
        # -------------------------------------------------
        # Stores info about each connection:
        #   - app_name: which wrapper app (from api_clients)
        #   - app_id: the api_client ID
        #   - connected_at: when the connection was established
        #   - last_message_at: when the last message was received
        #
        # This is useful for:
        #   - Logging (which app sent what)
        #   - Health checks (how many connections per app)
        #   - Idle timeout detection

        self.connection_info: dict[str, dict] = {}

        # -------------------------------------------------
        # Shutdown flag (Phase 5)
        # -------------------------------------------------
        #
        # When True, the message loop stops accepting new
        # messages and the heartbeat task exits. Set by
        # shutdown_all() during server shutdown.

        self._shutting_down = False

    def register(
        self,
        connection_id: str,
        websocket: WebSocket,
        app_name: str,
        app_id: str,
    ) -> None:
        """Register a new authenticated connection.

        Called AFTER the API key has been validated.
        """

        self.active_connections[connection_id] = websocket

        self.connection_info[connection_id] = {
            "connection_id": connection_id,
            "app_name": app_name,
            "app_id": app_id,
            "connected_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "last_message_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        logger.info(
            "WebSocket connected: '%s' (id=%s, conn=%s) "
            "— %d total connections",
            app_name,
            app_id,
            connection_id,
            len(self.active_connections),
        )

    def disconnect(self, connection_id: str) -> None:
        """Remove a connection and clean up its state.

        Called when a connection closes (cleanly or due to
        error/network interruption).
        """

        info = self.connection_info.pop(connection_id, {})
        self.active_connections.pop(connection_id, None)

        # Clean up any active chats for this connection
        active_chats.pop(connection_id, None)

        logger.info(
            "WebSocket disconnected: '%s' (conn=%s) "
            "— %d remaining connections",
            info.get("app_name", "unknown"),
            connection_id,
            len(self.active_connections),
        )

    def update_last_message(
        self, connection_id: str,
    ) -> None:
        """Update the last_message_at timestamp.

        Called every time a message is received. Used for
        idle timeout detection.
        """

        if connection_id in self.connection_info:
            self.connection_info[connection_id][
                "last_message_at"
            ] = datetime.now(timezone.utc).isoformat()

    def get_info(self, connection_id: str) -> dict:
        """Get metadata for a connection."""

        return self.connection_info.get(
            connection_id, {}
        )

    def get_stats(self) -> dict:
        """Get summary statistics for health checks.

        Returns:
            {
                "total_connections": 3,
                "apps": {
                    "Education AI": 2,
                    "Business AI": 1
                }
            }
        """

        apps: dict[str, int] = {}

        for info in self.connection_info.values():
            name = info.get("app_name", "unknown")
            apps[name] = apps.get(name, 0) + 1

        return {
            "total_connections": len(
                self.active_connections
            ),
            "apps": apps,
        }

    # -------------------------------------------------
    # Graceful shutdown (Phase 5)
    # -------------------------------------------------
    #
    # WHY GRACEFUL SHUTDOWN?
    #
    #   When the server stops (deploy, restart, Ctrl+C),
    #   connected wrappers see a sudden TCP disconnect.
    #   They can't tell if:
    #     (a) the server crashed unexpectedly, or
    #     (b) it's shutting down intentionally
    #
    #   With graceful shutdown, we send a WebSocket close
    #   frame with code 1001 ("Going Away") — the standard
    #   code for "the server is shutting down." Smart
    #   wrappers can read this code and wait before
    #   reconnecting, instead of hammering a server that
    #   isn't ready yet.
    #
    # CLOSE CODE 1001 ("Going Away"):
    #
    #   This is defined in RFC 6455 §7.4.1:
    #     "An endpoint is 'going away', such as a server
    #      going down or a browser having navigated away
    #      from a page."
    #
    #   It's the correct code for planned shutdowns.
    #   Other codes you might see:
    #     1000 = normal closure (clean disconnect by client)
    #     1001 = going away (server/client shutting down)
    #     1011 = unexpected condition (server error)

    async def shutdown_all(self) -> None:
        """Close all connections gracefully.

        Called during server shutdown. Sends a close frame
        with code 1001 to each connected wrapper before
        terminating the connection.
        """

        self._shutting_down = True

        if not self.active_connections:
            logger.info(
                "Shutdown: no WebSocket connections to close"
            )
            return

        count = len(self.active_connections)
        logger.info(
            "Shutdown: closing %d WebSocket connection(s)...",
            count,
        )

        # Close each connection with a "going away" frame.
        # We iterate over a copy of the dict because
        # disconnect() modifies it during iteration.

        for conn_id, ws in list(
            self.active_connections.items()
        ):
            try:
                await ws.close(
                    code=1001,
                    reason="Server is shutting down",
                )
            except Exception as exc:
                # Connection might already be dead — that's fine
                logger.debug(
                    "Shutdown close failed for conn=%s: %s",
                    conn_id,
                    exc,
                )

            # Clean up the connection state
            self.disconnect(conn_id)

        logger.info(
            "Shutdown: all %d WebSocket connection(s) closed",
            count,
        )


# ---------------------------------------------------------
# Global connection manager instance
# ---------------------------------------------------------
#
# Like shared.py's dicts, this is a module-level singleton.
# All WebSocket connections share the same manager because
# Python modules are imported once and cached.

manager = ConnectionManager()


# ---------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------
#
# HOW THE URL WORKS:
#
#   ws://localhost:8000/ws?api_key=dak_7f3a9b2c...
#
#   The wrapper connects to /ws with the API key as a
#   query parameter. FastAPI's @router.websocket decorator
#   handles the HTTP→WebSocket upgrade automatically.
#
# WHY api_key IN THE URL (not a header)?
#
#   The WebSocket protocol in browsers (JavaScript's
#   new WebSocket(url)) does NOT support custom headers
#   during the handshake. The only way to send auth data
#   is via the URL or cookies.
#
#   Using a query parameter is the standard approach:
#     - Stripe uses ?api_key=...
#     - Slack uses ?token=...
#     - Firebase uses ?auth=...
#
#   The connection should be over TLS (wss://), so the
#   key is encrypted in transit just like a header.
#
# CLOSE CODES:
#
#   WebSocket close codes tell the client WHY the
#   connection was closed:
#
#     1000 = Normal closure (clean disconnect)
#     1008 = Policy violation (auth failed)
#     4001 = Custom: authentication failed
#     4002 = Custom: invalid message format
#     4003 = Custom: idle timeout
#
#   Custom codes (4000-4999) are application-defined.
#   We use 4001+ for our own error conditions so the
#   wrapper can distinguish between "server crashed"
#   (1011) and "your key is wrong" (4001).

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    api_key: str = Query(
        ...,
        description="API key for authentication",
    ),
):
    """WebSocket gateway for wrapper applications.

    This is the main entry point for real-time communication
    with DocLens. Wrapper apps connect here and send JSON
    messages to perform document operations, search, and chat.

    Connection flow:
        1. Wrapper connects with API key in URL
        2. Server validates the key (Phase 3 auth)
        3. If valid: accept, enter message loop
        4. If invalid: reject with 4001 close code

    Message loop:
        - Receive JSON message from wrapper
        - Validate the message format (protocol.py)
        - Route to the correct handler (handlers.py)
        - Handler sends response(s) back to wrapper
        - Repeat until disconnect
    """

    # -------------------------------------------------
    # Step 1: Authenticate the API key
    # -------------------------------------------------
    #
    # We validate BEFORE accepting the connection. This
    # way, an invalid key gets rejected immediately — the
    # wrapper sees a WebSocket close frame with code 4001,
    # not a successful connection followed by an error.

    client = validate_api_key(api_key)

    if client is None:
        # Reject the connection — don't accept() it
        await websocket.close(
            code=4001,
            reason="Authentication failed: invalid or "
                   "revoked API key",
        )

        logger.warning(
            "WebSocket auth failed: key=%s...",
            api_key[:12] if api_key else "(empty)",
        )
        return

    # -------------------------------------------------
    # Step 2: Accept the connection
    # -------------------------------------------------

    await websocket.accept()

    # Generate a unique connection ID
    connection_id = "conn_" + uuid.uuid4().hex[:12]

    # Register in the connection manager
    manager.register(
        connection_id=connection_id,
        websocket=websocket,
        app_name=client["name"],
        app_id=client["id"],
    )

    # Send a welcome message with connection info
    await websocket.send_json(
        make_success("_connect", {
            "message": "Connected to DocLens",
            "connection_id": connection_id,
            "app_name": client["name"],
        })
    )

    # -------------------------------------------------
    # Step 3: Start heartbeat task (Phase 5)
    # -------------------------------------------------
    #
    # WHAT IS A HEARTBEAT?
    #
    #   A heartbeat is a periodic "are you still there?"
    #   check. The server sends a WebSocket ping frame
    #   every 30 seconds. The client's WebSocket library
    #   automatically responds with a pong frame.
    #
    #   If the pong doesn't arrive within 10 seconds, the
    #   client is considered dead and the connection is
    #   closed. This catches "zombie" connections.
    #
    # WHAT ARE ZOMBIE CONNECTIONS?
    #
    #   When a client disappears abruptly (laptop closes,
    #   process killed, network cable unplugged), the TCP
    #   connection can remain "open" on the server side
    #   for hours. The server doesn't know the client is
    #   gone — it's just sitting there waiting for a
    #   message that will never come.
    #
    #   These zombie connections waste memory (the WebSocket
    #   object, connection metadata, any chat state) and
    #   make get_stats() report inflated numbers.
    #
    #   The heartbeat detects and cleans them up within
    #   30-40 seconds of the client disappearing.
    #
    # HOW PING/PONG WORKS AT THE PROTOCOL LEVEL:
    #
    #   WebSocket has two control frame types for heartbeats:
    #     - Ping (opcode 0x9): "are you there?"
    #     - Pong (opcode 0xA): "yes I'm here"
    #
    #   These are PROTOCOL-LEVEL frames, not application
    #   messages. The client's WebSocket library handles
    #   pong replies automatically — the wrapper developer
    #   doesn't need to write any code for this.
    #
    #   Starlette's WebSocket.send() with {"type":
    #   "websocket.ping"} sends a protocol-level ping.
    #
    # WHY asyncio.create_task()?
    #
    #   The heartbeat needs to run IN PARALLEL with the
    #   message loop. While the message loop is waiting
    #   for the next message (await websocket.receive_json),
    #   the heartbeat task is sleeping for 30 seconds and
    #   then sending a ping.
    #
    #   asyncio.create_task() schedules the heartbeat
    #   coroutine to run concurrently on the same event
    #   loop. Think of it as "start this function running
    #   in the background."
    #
    # HEARTBEAT SETTINGS:
    #
    #   PING_INTERVAL = 30 seconds between pings
    #   PONG_TIMEOUT  = 10 seconds to wait for pong
    #
    #   These are conservative values. Most production
    #   WebSocket servers use 15-60 second intervals.
    #   30s is a good balance: frequent enough to detect
    #   dead connections quickly, but not so frequent
    #   that it wastes bandwidth on cellular connections.

    PING_INTERVAL = 30  # seconds between pings
    PONG_TIMEOUT = 10   # seconds to wait for pong reply

    async def _heartbeat(
        ws: WebSocket,
        conn_id: str,
    ) -> None:
        """Send periodic pings to detect dead connections.

        Runs as a background task alongside the message loop.
        If a ping goes unanswered for PONG_TIMEOUT seconds,
        the connection is closed.

        This coroutine exits when:
          - The connection is closed (send raises an exception)
          - The server is shutting down (manager._shutting_down)
          - A ping times out (pong not received)
        """

        try:
            while not manager._shutting_down:

                # Wait between pings
                await asyncio.sleep(PING_INTERVAL)

                # Don't ping if we're shutting down
                # (checked again after sleep)
                if manager._shutting_down:
                    break

                try:
                    # -----------------------------------------
                    # Send a protocol-level ping
                    # -----------------------------------------
                    #
                    # Starlette's WebSocket.send() accepts a
                    # dict with a "type" key. The type
                    # "websocket.ping" sends a WebSocket
                    # protocol-level ping frame.
                    #
                    # The client's WebSocket library will
                    # automatically respond with a pong —
                    # no application code needed on the
                    # client side.
                    #
                    # We use wait_for() with a timeout to
                    # detect unresponsive clients. If the
                    # pong doesn't arrive within PONG_TIMEOUT
                    # seconds, asyncio raises TimeoutError.

                    pong_waiter = ws.send(
                        {"type": "websocket.ping"}
                    )
                    await asyncio.wait_for(
                        pong_waiter,
                        timeout=PONG_TIMEOUT,
                    )

                except asyncio.TimeoutError:
                    # Pong didn't arrive in time — client
                    # is probably dead
                    logger.warning(
                        "Heartbeat timeout for conn=%s "
                        "— closing dead connection",
                        conn_id,
                    )
                    await ws.close(
                        code=4003,
                        reason="Heartbeat timeout: no pong "
                               "received",
                    )
                    return

        except Exception:
            # Connection already closed or errored out.
            # This is normal — the message loop will handle
            # the actual cleanup via the finally block.
            pass

    # Launch the heartbeat as a background task
    heartbeat_task = asyncio.create_task(
        _heartbeat(websocket, connection_id)
    )

    # -------------------------------------------------
    # Step 4: Message loop
    # -------------------------------------------------
    #
    # This loop runs for the ENTIRE lifetime of the
    # connection. It:
    #   1. Waits for a message from the wrapper
    #   2. Parses and validates the message
    #   3. Routes it to the correct handler
    #   4. Repeats
    #
    # The loop ends when:
    #   - The wrapper disconnects (WebSocketDisconnect)
    #   - A network error occurs
    #   - The server shuts down (Phase 5)
    #   - The heartbeat detects a dead connection

    try:
        while not manager._shutting_down:

            # Receive raw JSON from the wrapper
            try:
                raw_data = await websocket.receive_json()
            except ValueError:
                # Not valid JSON
                await websocket.send_json(
                    make_error(
                        "_parse_error",
                        ErrorCode.INVALID_MESSAGE,
                        "Message must be valid JSON",
                    )
                )
                continue

            # Update activity timestamp
            manager.update_last_message(connection_id)

            # -----------------------------------------
            # Validate message format
            # -----------------------------------------
            #
            # WSRequest (from protocol.py) checks:
            #   - "id" field exists (string)
            #   - "action" is a valid Action enum value
            #   - "payload" is a dict (defaults to {})
            #
            # If validation fails, Pydantic gives us a
            # detailed error like "action: value is not
            # a valid Action". We send this back so the
            # wrapper developer can fix their code.

            try:
                message = WSRequest(**raw_data)
            except ValidationError as ve:
                # Extract the message ID if it was provided
                # (even if other fields are invalid)
                msg_id = raw_data.get("id", "_validation_error")

                await websocket.send_json(
                    make_error(
                        msg_id,
                        ErrorCode.INVALID_MESSAGE,
                        f"Invalid message format: {ve}",
                        {"errors": ve.errors()},
                    )
                )
                continue

            # -----------------------------------------
            # Route to the correct handler
            # -----------------------------------------
            #
            # ACTION_HANDLERS is a dict mapping action
            # names to handler functions. If the action
            # isn't in the dict, it's an unknown action
            # (shouldn't happen because Pydantic already
            # validated the action enum, but we check
            # anyway for safety).

            handler = ACTION_HANDLERS.get(message.action.value)

            if handler is None:
                await websocket.send_json(
                    make_error(
                        message.id,
                        ErrorCode.INVALID_ACTION,
                        f"Unknown action: '{message.action}'",
                    )
                )
                continue

            # Call the handler
            try:
                await handler(
                    websocket=websocket,
                    msg_id=message.id,
                    payload=message.payload,
                    connection_info=manager.get_info(
                        connection_id
                    ),
                )
            except Exception as exc:
                # Handler crashed — send error, keep connection
                logger.error(
                    "Handler '%s' crashed: %s",
                    message.action,
                    exc,
                    exc_info=True,
                )

                await websocket.send_json(
                    make_error(
                        message.id,
                        ErrorCode.INTERNAL_ERROR,
                        f"Internal error processing "
                        f"'{message.action}': {exc}",
                    )
                )

    except WebSocketDisconnect:
        # -------------------------------------------------
        # Clean disconnect
        # -------------------------------------------------
        # The wrapper closed the connection normally. This
        # is expected — e.g., the PHP app shut down or
        # the page was closed.

        logger.info(
            "WebSocket cleanly disconnected: conn=%s",
            connection_id,
        )

    except Exception as exc:
        # -------------------------------------------------
        # Unexpected error
        # -------------------------------------------------
        # Network interruption, server error, etc. Log it
        # and clean up. The wrapper should automatically
        # reconnect (it's the wrapper's responsibility to
        # implement reconnection with exponential backoff).

        logger.error(
            "WebSocket error for conn=%s: %s",
            connection_id,
            exc,
            exc_info=True,
        )

    finally:
        # -------------------------------------------------
        # Cleanup (always runs)
        # -------------------------------------------------
        # 1. Cancel the heartbeat task so it doesn't keep
        #    pinging a closed connection.
        # 2. Remove the connection from the manager.
        #
        # This is in a finally block so it runs whether the
        # disconnection was clean, dirty, or caused by an
        # error.

        heartbeat_task.cancel()

        # Suppress the CancelledError — it's expected when
        # we cancel a sleeping task.
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        manager.disconnect(connection_id)
