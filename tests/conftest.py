"""Shared pytest setup.

Tests must never touch your real chat history database
(data/chat_history.db). Before any app module is imported,
we point every module's DB path at a throwaway temp folder
and raise the anonymous usage limits so tests aren't
rate limited.
"""

import os
import tempfile
from pathlib import Path

os.environ["ANON_MAX_QUESTIONS"] = "100000"
os.environ["ANON_MAX_UPLOADS"] = "100000"
os.environ.setdefault("TRUST_PROXY_HEADERS", "false")

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="doclens-tests-"))
_TEST_DB = _TEST_DATA_DIR / "test.db"

import ai_document_agent.database as _database  # noqa: E402
import ai_document_agent.rate_limiter as _rate_limiter  # noqa: E402
import ai_document_agent.query_cache as _query_cache  # noqa: E402

for _module in (_database, _rate_limiter, _query_cache):
    if hasattr(_module, "DATA_DIR"):
        _module.DATA_DIR = _TEST_DATA_DIR
    if hasattr(_module, "DB_PATH"):
        _module.DB_PATH = _TEST_DB
