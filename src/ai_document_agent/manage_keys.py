# ---------------------------------------------------------
# manage_keys.py — API Key Management CLI (Phase 3)
# ---------------------------------------------------------
#
# WHAT THIS FILE DOES:
#
#   A command-line tool for managing API keys that wrapper
#   applications use to authenticate with DocAgent. Run it
#   from the project root:
#
#     python -m ai_document_agent.manage_keys create "Education AI"
#     python -m ai_document_agent.manage_keys list
#     python -m ai_document_agent.manage_keys revoke "Education AI"
#
# WHY A CLI TOOL (not a web endpoint)?
#
#   API key management is an ADMIN action — only the server
#   operator should create or revoke keys. Putting this behind
#   a web endpoint would require admin authentication (chicken-
#   and-egg problem: you need auth to set up auth).
#
#   A CLI tool runs on the server itself, so only someone with
#   SSH/terminal access can use it. This is the standard pattern
#   used by Django (manage.py), Rails (rake), and most web
#   frameworks for admin operations.
#
# HOW API KEYS WORK:
#
#   1. Admin runs: manage_keys create "Education AI"
#   2. The tool generates a random key: dak_7f3a9b2c4d5e...
#   3. The key is SHA-256 hashed and stored in the database
#   4. The UNHASHED key is displayed ONCE — admin copies it
#      and gives it to the wrapper app developer
#   5. The wrapper app sends this key in its WebSocket
#      connection URL: ws://host/ws?api_key=dak_7f3a9b2c...
#   6. DocAgent hashes the incoming key and compares it to
#      the stored hash — if they match, connection is allowed
#
#   The actual key is NEVER stored. If lost, generate a new one.
#
# KEY FORMAT:
#
#   dak_<32 hex characters>
#
#   "dak" = DocAgent Key (a clear prefix so you can identify
#   what this key is for, like how GitHub uses "ghp_" and
#   Stripe uses "sk_").
#
# WHY __main__ GUARD?
#
#   The `if __name__ == "__main__":` guard at the bottom
#   ensures this code only runs when executed directly
#   (python -m ...), not when imported. This lets other
#   modules import functions from here without triggering
#   the CLI.
#
# USAGE:
#
#   python -m ai_document_agent.manage_keys create "App Name"
#     → Creates a new API key, prints it ONCE
#
#   python -m ai_document_agent.manage_keys list
#     → Shows all registered apps and their status
#
#   python -m ai_document_agent.manage_keys revoke "App Name"
#     → Deactivates an app's key (it can no longer connect)
# ---------------------------------------------------------

import hashlib
import secrets
import sqlite3
import sys

# ---------------------------------------------------------
# Load .env FIRST — same pattern as main.py
# ---------------------------------------------------------
#
# WHY HERE TOO?
#
#   This script runs INDEPENDENTLY from the web server.
#   When you run `python -m ai_document_agent.manage_keys`,
#   main.py is NOT executed, so its load_dotenv() doesn't
#   run. We need our own load_dotenv() to read any .env
#   config (though this module doesn't currently use env
#   vars, it's good practice for consistency).

from dotenv import load_dotenv
load_dotenv()

from ai_document_agent.database import (
    create_api_client,
    init_api_clients,
    init_db,
    list_api_clients,
    revoke_api_client,
)


# ---------------------------------------------------------
# Key generation
# ---------------------------------------------------------

def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its SHA-256 hash.

    HOW IT WORKS:

        1. secrets.token_hex(32) generates 32 random bytes
           as a 64-character hex string. This uses the OS's
           cryptographically secure random number generator
           (/dev/urandom on Linux, CryptGenRandom on Windows).

        2. We prepend "dak_" (DocAgent Key) as a prefix so
           the key is immediately identifiable.

        3. We hash the full key with SHA-256 — this hash is
           what gets stored in the database.

    WHY secrets (not random)?

        Python's `random` module is NOT cryptographically
        secure — its output can be predicted if you know the
        seed. `secrets` uses the OS's CSPRNG, which is
        designed for generating keys, tokens, and passwords.

    WHY SHA-256 (not bcrypt)?

        API keys are high-entropy random strings (64 hex chars
        = 256 bits of entropy). Unlike passwords (which are
        low-entropy and benefit from bcrypt's slow hashing),
        API keys can't be brute-forced — there are 2^256
        possible keys. SHA-256 is fast and sufficient here.

        bcrypt would work too, but it's unnecessarily slow
        for high-entropy inputs and would add a dependency.

    Returns:
        Tuple of (raw_key, key_hash):
          - raw_key: "dak_7f3a9b2c..." — give this to the app
          - key_hash: SHA-256 hex digest — stored in database
    """

    # Generate 32 random bytes as hex (64 chars)
    random_part = secrets.token_hex(32)

    # Full key with prefix
    raw_key = f"dak_{random_part}"

    # Hash for storage (never store the raw key!)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    return raw_key, key_hash


def hash_api_key(raw_key: str) -> str:
    """Hash an API key with SHA-256.

    Used when validating incoming keys: hash the provided
    key and compare to the stored hash.

    Args:
        raw_key: The full API key (e.g., "dak_7f3a9b2c...")

    Returns:
        SHA-256 hex digest of the key.
    """

    return hashlib.sha256(raw_key.encode()).hexdigest()


# ---------------------------------------------------------
# CLI commands
# ---------------------------------------------------------

def cmd_create(name: str) -> None:
    """Create a new API key for a wrapper application.

    The key is displayed ONCE and never stored in plain text.
    If lost, revoke the old key and create a new one.

    Args:
        name: Human-readable app name (e.g., "Education AI").
    """

    # Generate the key
    raw_key, key_hash = generate_api_key()

    try:
        # Store the hash in the database
        client = create_api_client(
            name=name,
            key_hash=key_hash,
        )

        # Display the key — this is the ONLY time it's shown
        print()
        print("=" * 60)
        print(f"  API Key created for: {name}")
        print("=" * 60)
        print()
        print(f"  Client ID:  {client['id']}")
        print(f"  API Key:    {raw_key}")
        print()
        print("  IMPORTANT: Save this key now!")
        print("  It will NEVER be shown again.")
        print("  If lost, revoke and create a new one.")
        print()
        print("  Use it in WebSocket connections:")
        print(f"  ws://your-server/ws?api_key={raw_key}")
        print("=" * 60)
        print()

    except sqlite3.IntegrityError as e:
        error_msg = str(e).lower()

        if "name" in error_msg:
            print(
                f"\nError: An app named '{name}' already "
                f"exists. Use a different name, or revoke "
                f"the existing one first.\n"
            )
        elif "key_hash" in error_msg:
            # Astronomically unlikely — two keys with the
            # same hash. Just try again.
            print(
                "\nError: Key collision (extremely rare). "
                "Please try again.\n"
            )
        else:
            print(f"\nError: {e}\n")

        sys.exit(1)


def cmd_list() -> None:
    """List all registered API clients."""

    clients = list_api_clients()

    if not clients:
        print("\nNo API clients registered yet.")
        print("Create one with: python -m ai_document_agent"
              ".manage_keys create \"App Name\"\n")
        return

    print()
    print(f"{'Name':<25} {'Status':<10} {'ID':<20} "
          f"{'Created':<22} {'Last Seen'}")
    print("-" * 100)

    for client in clients:
        status = (
            "Active" if client["is_active"]
            else "Revoked"
        )
        last_seen = client["last_seen"] or "Never"

        # Truncate timestamps for display
        created = (
            client["created_at"][:19]
            if client["created_at"] else "?"
        )
        if last_seen != "Never":
            last_seen = last_seen[:19]

        print(
            f"{client['name']:<25} {status:<10} "
            f"{client['id']:<20} {created:<22} {last_seen}"
        )

    print()
    print(f"Total: {len(clients)} client(s)")
    print()


def cmd_revoke(name: str) -> None:
    """Revoke an API client's access.

    The client record stays in the database (for audit)
    but the key can no longer be used to connect.

    Args:
        name: The app name to revoke.
    """

    success = revoke_api_client(name)

    if success:
        print(f"\nRevoked API access for '{name}'.")
        print("The app can no longer connect to DocAgent.\n")
    else:
        print(f"\nError: No app named '{name}' found.")
        print("Use 'list' to see registered apps.\n")
        sys.exit(1)


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
#
# HOW PYTHON -m WORKS:
#
#   When you run `python -m ai_document_agent.manage_keys`,
#   Python finds this file and executes it as a script.
#   The __name__ variable is set to "__main__" only when
#   the file is run directly (not when imported).
#
#   sys.argv contains the command-line arguments:
#     sys.argv[0] = the module path (ignored)
#     sys.argv[1] = the command ("create", "list", "revoke")
#     sys.argv[2] = the argument (app name, if needed)

if __name__ == "__main__":

    # Ensure database tables exist before any operations
    init_db()
    init_api_clients()

    if len(sys.argv) < 2:
        print()
        print("DocAgent API Key Manager")
        print()
        print("Usage:")
        print("  python -m ai_document_agent.manage_keys "
              "create \"App Name\"")
        print("  python -m ai_document_agent.manage_keys "
              "list")
        print("  python -m ai_document_agent.manage_keys "
              "revoke \"App Name\"")
        print()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "create":
        if len(sys.argv) < 3:
            print(
                "\nError: Please provide an app name."
            )
            print(
                'Usage: python -m ai_document_agent'
                '.manage_keys create "Education AI"\n'
            )
            sys.exit(1)

        cmd_create(sys.argv[2])

    elif command == "list":
        cmd_list()

    elif command == "revoke":
        if len(sys.argv) < 3:
            print(
                "\nError: Please provide the app name "
                "to revoke."
            )
            print(
                'Usage: python -m ai_document_agent'
                '.manage_keys revoke "Education AI"\n'
            )
            sys.exit(1)

        cmd_revoke(sys.argv[2])

    else:
        print(f"\nUnknown command: '{command}'")
        print("Available commands: create, list, revoke\n")
        sys.exit(1)
