# AI Document Agent — Progress Log

This file tracks what has been built, when, and what was learned at each step.

---

## Phase 1 — Agent Fundamentals (Completed)

### 1.1 Python Environment
- **Status:** Done
- **What:** Python 3.12.14, `uv` for package management, VS Code, Git
- **Lesson:** `uv` is fast and handles virtual environments + dependencies in one tool.

### 1.2 Ollama + Qwen3 8B
- **Status:** Done
- **What:** Installed Ollama, pulled `qwen3:8b` model, tested local inference.
- **Lesson:** 8B parameter model runs on CPU but is slow (~26s per inference call on Ryzen 5 5500U). Good enough for learning, not for production.

### 1.3 Basic LLM Interaction
- **Status:** Done
- **What:** Python successfully communicates with Ollama using the `ollama` Python package.
- **Lesson:** The Ollama Python SDK makes it simple — `from ollama import chat` and pass messages.

### 1.4 Calculator Tool
- **Status:** Done
- **What:** Created a `calculator()` function supporting add, subtract, multiply, divide.
- **Lesson:** Tools are just regular Python functions. The LLM decides when to call them based on the tool's description and parameter schema.

### 1.5 Agent Tool-Calling Loop
- **Status:** Done
- **What:** Built a non-streaming agent loop: User → LLM decision → Tool execution → LLM final answer.
- **Lesson:** Agent loops require multiple LLM calls, which multiplies latency. Tool-calling is powerful but expensive on CPU.

### 1.6 Performance Measurement
- **Status:** Done
- **What:** Added timing to measure LLM decision time, tool execution time, and total response time.
- **Typical results:** LLM decision ~27s, tool ~0s, final LLM ~9s, total ~37s.
- **Lesson:** Always measure. Without timing, you can't tell where the bottleneck is.

### 1.7 Streaming
- **Status:** Done
- **What:** Implemented streaming from Ollama so tokens appear as they're generated.
- **Lesson:** Streaming dramatically improves perceived performance. First token at ~14.7s feels much better than waiting 25s for a complete blank-then-answer.

### 1.8 FastAPI Backend
- **Status:** Done
- **What:** FastAPI with Uvicorn. Endpoints: `GET /`, `GET /health`, `POST /chat`, `POST /chat/stream`.
- **Lesson:** FastAPI auto-generates Swagger docs at `/docs`. Pydantic handles request validation. `StreamingResponse` with NDJSON enables real-time streaming to the browser.

### 1.9 Browser Chat UI
- **Status:** Done
- **What:** HTML/CSS/JS chat interface served by FastAPI. Shows user messages, AI responses, agent status events, timing, and errors.
- **Lesson:** Vanilla JavaScript + Fetch API + NDJSON streaming is enough for a real-time chat UI. No framework needed yet.

### 1.10 Streaming Agent Events
- **Status:** Done
- **What:** Structured event protocol: `thinking`, `decision_completed`, `tool_call`, `tool_result`, `generating`, `first_token`, `token`, `completed`, `error`.
- **Lesson:** Expose safe application status, not private chain-of-thought. Users should see "Using calculator..." not the model's internal reasoning.

---

## Phase 2 — Robust Agent (Completed)

### 2.1 Calculator Input Normalization
- **Status:** Done (2026-08-18)
- **What:** LLM sent `operation: "+"` but calculator only accepted words like "add". Added an `OPERATION_MAP` that normalizes symbols (`+`, `-`, `*`, `/`) and word variants (`addition`, `subtraction`, etc.) to canonical operations.
- **Lesson:** LLM-generated tool arguments are unpredictable. Always validate and normalize inputs. A lookup dictionary is a clean pattern for this.

### 2.2 Conversation State
- **Status:** Done (2026-08-18)
- **What:** Added in-memory session store so the agent remembers previous messages within a conversation.
  - `main.py`: `sessions` dict keyed by `session_id`, with `trim_session()` capping at 50 messages.
  - `agent.py`: `ask_agent()` and `stream_agent()` now accept full `messages` list instead of a single question.
  - `index.html`: Generates `SESSION_ID` via `crypto.randomUUID()` per browser tab.
- **Verified:** "What is 25 times 48?" → 1200, then "Now divide that by 5" → correctly answered 240.
- **Lesson:** Separation of concerns — session management belongs in the API layer (`main.py`), not the agent (`agent.py`). The agent should be unaware of how sessions are stored. This makes it easy to swap the session store later (dict → Redis → PostgreSQL).

### 2.3 Strict Tool Schema Validation
- **Status:** Done (2026-08-18)
- **What:** Replaced auto-generated tool schemas with explicit JSON schemas including `enum` constraints. Added a `TOOL_REGISTRY` pattern.
  - `CALCULATOR_SCHEMA`: Explicit JSON schema with `enum: ["add", "subtract", "multiply", "divide"]` for the operation parameter.
  - `TOOLS` list: All tool schemas in one place — easy to extend.
  - `TOOL_REGISTRY` dict: Maps tool names to Python functions — adding a new tool is one line, no if/elif changes in the agent loop.
- **Lesson:** Two layers of defense: (1) the schema tells the LLM *what to send* (prevention), (2) the `OPERATION_MAP` normalizes *what it actually sends* (safety net). This is defense in depth. The JSON schema format (`type`, `properties`, `enum`) is the same across Ollama, Claude, and GPT — so this schema is portable when you switch providers.

### 2.4 Error Handling
- **Status:** Done (2026-08-18)
- **What:** Added structured error handling across the entire stack so failures produce useful information instead of raw stack traces.
  - `agent.py`: `try/except` around every Ollama `chat()` call. On failure, `stream_agent()` yields a structured `{"type": "error"}` event. `ask_agent()` lets exceptions propagate to the API layer.
  - `main.py /chat`: `try/except` returns `JSONResponse(status_code=500)` with `error`, `error_type`, and `request_id`. On failure, removes the dangling user message from session history so conversations stay clean.
  - `main.py /chat/stream`: `try/except` inside the generator yields an error event. `finally` block cleans up dangling user messages when `had_error` is True.
- **Lesson:** Error handling in streaming is trickier than in regular endpoints — you can't return a 500 status after you've already started sending 200 chunks. Instead, you send an error *event* inside the stream and clean up session state in a `finally` block. Always clean up side effects (like appended messages) when an operation fails partway through.

### 2.5 Request IDs + Logging
- **Status:** Done (2026-08-18)
- **What:** Replaced all `print()` with Python's `logging` module and added unique request IDs for tracing.
  - `main.py`: `logging.basicConfig()` with timestamp format. `make_request_id()` generates `"req_" + uuid4().hex[:8]` for each API call. Every log line includes `[request_id]`.
  - `agent.py`: `logger = logging.getLogger(__name__)`. All log lines prefixed with `[request_id]`. Logs LLM decision time, tool calls, tool results, and total response time.
  - Every streaming event and error response includes `request_id` so the browser can correlate with server logs.
- **Lesson:** `logging` beats `print()` because it gives you timestamps, log levels (INFO/ERROR/WARNING), source module names, and the ability to route output (to files, Datadog, CloudWatch) without code changes. Request IDs are essential — when debugging concurrent requests, you search for the ID and see the complete trace.

### 2.6 Tests
- **Status:** Done (2026-08-18)
- **What:** Added 44 automated tests across two test files, all passing.
  - `tests/test_calculator.py` (31 tests): Basic operations (9), synonym handling (18), case insensitivity (3), error handling (3). Tests every supported operation, every synonym (`+`, `sum`, `plus`, `addition`, etc.), and edge cases (negatives, zeros, floats, large numbers).
  - `tests/test_api.py` (13 tests): Health endpoint, root endpoint, chat endpoint (answer, session save, error 500, error cleanup, validation), stream endpoint (NDJSON format, session save, error cleanup), session management (auto-generate ID, continuity, isolation).
  - Ollama is fully mocked (`unittest.mock.patch`) so tests run in <1 second without a running model.
  - Added `pytest` and `httpx` as dev dependencies in `pyproject.toml`.
- **Run with:** `uv run pytest tests/ -v`
- **Lesson:** Mock external dependencies (like Ollama) so tests are fast, deterministic, and runnable anywhere. FastAPI's `TestClient` simulates HTTP without a real server. Test the contract (request → response shape) not the implementation details.

---

## Phase 3 — PDF Upload & Vector Search (Completed)

### 3.1 PDF Text Extraction
- **Status:** Done (2026-08-18)
- **What:** Created `pdf_processor.py` using PyMuPDF (fitz) to extract text from PDFs page by page. Returns structured data with page numbers.
- **Lesson:** The Python import for PyMuPDF is `fitz`, not `pymupdf` — historical naming from the MuPDF C library. Always handle PDFs with no extractable text (scanned images need OCR, which is a future enhancement).

### 3.2 Text Chunking with Overlap
- **Status:** Done (2026-08-18)
- **What:** Split extracted text into 500-character chunks with 100-character overlap. Each chunk carries metadata (source filename, page number, chunk index).
- **Lesson:** Overlap prevents sentences from being cut in half at chunk boundaries. 20% overlap (100/500) is a good starting point. Metadata is essential — without it, you find the answer but can't tell the user where it came from.

### 3.3 Embeddings via Ollama + ChromaDB
- **Status:** Done (2026-08-18)
- **What:** Generate vector embeddings using Ollama's `nomic-embed-text` model (768 dimensions). Store chunks + embeddings in ChromaDB (persistent, survives server restarts). Cosine similarity for search.
  - `generate_embeddings()`: Sends text to Ollama's embed endpoint, returns vectors.
  - `store_chunks()`: Upserts into ChromaDB with deterministic IDs (re-upload won't duplicate).
  - `search_documents()`: Converts query to vector, finds N nearest chunks.
- **Lesson:** Embeddings capture *meaning*, not keywords. "revenue" and "income" are close in vector space. ChromaDB's `PersistentClient` saves to disk automatically — no separate database server needed. `upsert` (insert-or-update) prevents duplicates when re-uploading the same PDF.

### 3.4 PDF Upload API + Document Management
- **Status:** Done (2026-08-18)
- **What:** Three new endpoints:
  - `POST /upload`: Accept PDF via multipart/form-data, process end-to-end (extract → chunk → embed → store). Validates file type, saves to `uploads/` directory.
  - `GET /documents`: List all uploaded documents with chunk/page counts.
  - `DELETE /documents/{id}`: Remove a document and all its chunks from ChromaDB.
- **Lesson:** File uploads use `multipart/form-data` (not JSON). FastAPI's `UploadFile` handles this. `python-multipart` package is required for file upload support. Always validate file type server-side — don't trust the client.

### 3.5 Browser Upload UI
- **Status:** Done (2026-08-18)
- **What:** Added drag-and-drop + click-to-browse PDF upload to `index.html`. Shows upload progress, success/error states, and a document list with delete buttons.
- **Lesson:** `preventDefault()` on drag events is critical — without it, the browser navigates to the dropped file. `FormData` is the standard JavaScript API for sending files. Visual feedback (uploading → success → reset) makes the UX feel responsive.

### 3.6 Search Documents Agent Tool
- **Status:** Done (2026-08-18)
- **What:** Added `search_documents` as a new tool in the agent's `TOOLS` list and `TOOL_REGISTRY`. When the LLM receives a question about uploaded documents, it calls this tool to retrieve relevant chunks, then uses them to formulate an answer with source citations.
  - `SEARCH_DOCUMENTS_SCHEMA`: JSON schema with query (required) and n_results (optional) parameters.
  - `_search_tool()`: Wrapper that formats search results as readable text for the LLM, including source, page, and relevance score.
- **Lesson:** This is the "R" in RAG — Retrieval-Augmented Generation. The LLM decides *when* to search (tool calling), the vector DB finds *what's relevant* (semantic search), and the LLM writes the answer *using those sources* (generation). The tool registry pattern paid off — adding a second tool was just one schema + one registry entry.

### 3.7 New Dependencies
- **Status:** Done (2026-08-18)
- **What:** Added to `pyproject.toml`: `chromadb` (vector database), `pymupdf` (PDF extraction), `python-multipart` (file upload support). Pull `nomic-embed-text` model via Ollama.
- **Setup:** `ollama pull nomic-embed-text` + `uv sync`

### 3.8 UI Redesign + Auto-Summary
- **Status:** Done (2026-08-18)
- **What:** Complete UI overhaul and auto-summary feature.
  - **Text formatting:** Replaced raw markdown (`**bold**`) with proper HTML `<strong>` rendering. Added `formatText()` function that converts `**text**` → bold, `*text*` → italic, and `\n` → `<br>`. Also escapes HTML to prevent XSS injection.
  - **Sticky document panel:** Redesigned from single-page layout to a two-column layout. Left sidebar (340px) holds the upload area and document list — always visible, scrolls independently. Right side holds the chat.
  - **Multiple PDF upload:** Added `multiple` attribute to the file input. `handleFiles()` processes an array of files sequentially, showing per-file progress (`Processing invoice.pdf (2/3)...`).
  - **Chat blocked until upload:** Chat input and send button are `disabled` until at least one PDF exists. A placeholder message ("Upload a document to start") explains the restriction. `lockChat()` and `unlockChat()` toggle the state. This prevents users from using the agent as a general-purpose chatbot.
  - **Auto-summary on upload:** After `process_pdf()` succeeds, `generate_summary()` sends the first ~2000 chars of extracted text to the LLM with a prompt asking for a 300-word summary + 3-5 key insights. The summary is parsed from the LLM's response and returned in the upload API response. The browser renders it as a styled "summary card" in the chat area.
  - **New helper in `main.py`:** `generate_summary(pdf_path, request_id)` — calls Ollama directly (not through the agent loop) with a structured prompt. Parses `SUMMARY:` and `KEY INSIGHTS:` sections from the response.
- **Lesson:** Blocking chat until upload is a product decision, not a technical one — it forces the tool to be used as intended (document analysis) rather than as a general AI chatbot. Auto-summary provides immediate value after upload, so the user doesn't stare at "3 pages, 12 chunks indexed" wondering what to ask. Limiting LLM input to ~2000 chars keeps summary generation fast (~10-30s) even on CPU.

## Phase 3.9 — Performance Optimization (Completed)

### 3.9.1 True Streaming
- **Status:** Done (2026-08-18)
- **What:** The main chat path (document questions without calculator) was NOT streaming — the entire LLM response was generated silently, then sent as one blob. Users saw a blank screen for 2-8 minutes. Rewrote `stream_agent()` to use `chat(..., stream=True)` so tokens appear in the browser as they're generated. First visible token now appears within ~15-25 seconds.
- **Lesson:** Streaming doesn't make generation faster — it makes it *feel* faster. The user sees progress instead of a blank screen. With tool calls + streaming, you need to handle two paths: content tokens (direct answer) vs tool_calls (calculator), detected by checking what the stream produces.

### 3.9.2 Context Limiting
- **Status:** Done (2026-08-18)
- **What:** The full conversation history (up to 50 messages) was sent to the LLM on every call. On CPU, more input tokens = proportionally slower generation. Added `LLM_CONTEXT_MESSAGES = 6` — only the system prompt + last 6 messages (3 exchanges) are sent. Full history stays in the session store.
- **Lesson:** Context length is the #1 performance lever on CPU. Halving the input can cut generation time by 30-50%. 6 messages is enough for follow-up questions ("And in January?" after "What was spending?").

### 3.9.3 Removed Auto-Summary from Upload
- **Status:** Done (2026-08-18)
- **What:** The `/upload` endpoint called `generate_summary()` synchronously, adding 30-60s of LLM processing. It also re-extracted the PDF text (double work — `process_pdf()` already extracted it). Removed the LLM summary call. Upload now returns immediately after indexing with a static message. Users can ask for a summary in chat.
- **Lesson:** Don't block I/O operations on LLM calls. Upload should be fast (save + index). Analysis should be on-demand. Also: never extract data twice — pass it through.

### 3.9.4 Conditional Calculator Tool
- **Status:** Done (2026-08-18)
- **What:** The calculator tool schema was included in every LLM call, forcing the model to consider tool use even for "What is the topper's name?" Added `_needs_calculator()` — a regex check for math-related keywords. Tool schema is only included when the question looks like it needs arithmetic.
- **Lesson:** Every token in the prompt costs processing time on CPU. Remove what isn't needed. A simple regex filter avoids unnecessary context expansion for 95% of questions.

### 3.9.5 Disabled Chain-of-Thought (think=False)
- **Status:** Done (2026-08-18)
- **What:** `think=True` was set on all LLM calls, making Qwen3 8B generate hidden reasoning tokens before every visible answer. This roughly doubled response time on CPU. Changed to `think=False` — with forced retrieval, the model doesn't need to reason about what to do.
- **Lesson:** `think=True` is valuable when the model needs to make complex decisions (like choosing tools). With forced retrieval, the model's job is simple: "read context, answer question." No thinking needed.

---

## Phase 3.10 — Retrieval Accuracy: Active Document Selection (Completed)

### 3.10.1 Problem
- With 2+ PDFs uploaded, semantic search returned chunks from the WRONG document.
- Example: asking "what is the topper name" with Student Results + a Bill uploaded → retrieval returned Bill chunks because cosine similarity doesn't distinguish documents.

### 3.10.2 Fix: Source Filter
- **Discovery:** `search_documents()` in `pdf_processor.py` ALREADY accepted a `source_filter` parameter and built a ChromaDB `where` clause — it was just never called with it.
- **Backend wiring (agent.py):** Threaded `source_filter` through `build_context_prompt()`, `ask_agent()`, `stream_agent()`. When set, appends `"You are answering questions about: {filename}"` to the system prompt.
- **Backend wiring (main.py):** Added `source_filter` field to `ChatRequest` Pydantic model. Both `/chat` and `/chat/stream` pass it to the agent functions.
- **Frontend (index.html):** Click a document in the sidebar to make it "active" (highlighted with blue border). Chat requests include `source_filter: activeDocumentSource`. Indicator chip above input shows "Searching: filename" or "Searching: All documents". Auto-selects when only 1 document exists.
- **Lesson:** Semantic search alone can't distinguish between documents. Metadata filtering (ChromaDB `where` clause) is the correct solution — simple and fast.

### Files Modified
- `src/ai_document_agent/agent.py` — source_filter parameter threading
- `src/ai_document_agent/main.py` — ChatRequest model, source_filter passthrough
- `static/index.html` — document selection UI, active styling, source filter chip

---

## Future Phases

## Phase 4.1 — Hybrid Retrieval: BM25 + Semantic Search (Completed)

### 4.1.1 Problem
- Semantic search (ChromaDB cosine similarity) finds meaning-similar chunks but misses exact keyword matches. Searching for a specific name, number, or term could rank a vague paraphrase higher than the chunk with the exact match.

### 4.1.2 Fix: BM25 + Reciprocal Rank Fusion
- **Added `rank-bm25` dependency** — pure Python BM25 keyword search library.
- **`BM25Index` class** — in-memory keyword index that rebuilds from ChromaDB on startup and after any add/delete. Simple tokenizer (lowercase + split on non-word chars).
- **Hybrid `search_documents()`** — runs BOTH semantic search (ChromaDB) and BM25 keyword search, then merges results using Reciprocal Rank Fusion (RRF, k=60).
- **RRF formula:** `score = sum(1 / (k + rank))` for each retriever. Rank-based, not score-based, so no need to normalize cosine vs BM25 scores.
- **Source filter support** — BM25 respects `source_filter` by building a temporary index over just that document's chunks.
- **Lesson:** Hybrid retrieval is the industry standard for RAG. Semantic search alone has blind spots for exact matches. BM25 alone can't understand meaning. Together via RRF, you get the best of both with zero tuning.

### Files Modified
- `src/ai_document_agent/pdf_processor.py` — BM25Index class, _semantic_search(), _reciprocal_rank_fusion(), hybrid search_documents()
- `src/ai_document_agent/agent.py` — updated relevance display to show RRF score
- `pyproject.toml` — added rank-bm25 dependency

---

## Phase 4.2 — Conversation-Aware Retrieval (Completed)

### 4.2.1 Problem
- Each question was searched independently. Follow-up questions like "what about Hindi?" or "tell me more about that" contained almost no searchable keywords, so BM25 and semantic search returned irrelevant chunks or nothing.

### 4.2.2 Fix: Heuristic Query Enrichment
- **No extra LLM call** — on CPU (Ryzen 5 5500U), each LLM call takes 70-80s. A query-rewrite LLM call would double response time. Instead, used lightweight heuristics.
- **`_enrich_query()` function** — detects follow-up questions using three checks:
  1. Contains referential words: "that", "it", "this", "those", "these", "them", "the same", "above", "previous", "more"
  2. Starts with connecting words: "and", "but", "also", "what about", "how about"
  3. Very short (< 6 words) and doesn't look like a standalone question (no "what is", "list", "show", etc.)
- **Enrichment:** Prepends the previous user question: `"{prev_question} — {follow_up}"`. This gives BM25 real keywords and semantic search a richer meaning vector.
- **Example:** Q1: "list the subject wise topper name" → Q2: "what about Hindi?" → Enriched: "list the subject wise topper name — Hindi" → BM25 matches "topper" + "Hindi".
- **Wired into both `ask_agent()` and `stream_agent()`** — only the search query is enriched; the original question is still used for the LLM conversation and logging.
- **Lesson:** Not every problem needs an LLM. A 20-line heuristic with zero latency cost solved the follow-up problem. Save the LLM for what only an LLM can do.

### Files Modified
- `src/ai_document_agent/agent.py` — added `_enrich_query()`, wired into `ask_agent()` and `stream_agent()`

---

## Phase 4.3 — SHA-256 Content-Based Document IDs (Completed)

### 4.3.1 Problem
- Document IDs were random UUIDs (`uuid.uuid4().hex[:12]`). Uploading the same PDF twice created duplicate chunks in ChromaDB — wasted storage and search returned the same text twice from two "different" documents.

### 4.3.2 Fix: SHA-256 of File Content
- **Content-based IDs:** Read the uploaded file bytes, compute `hashlib.sha256(bytes).hexdigest()[:12]`. Same file content = same document_id, regardless of filename.
- **Duplicate detection:** Before processing, check `list_documents()` for an existing document with the same hash. If found, return immediately with `status: "duplicate"` and a friendly message — no re-processing.
- **Frontend handling:** `app.js` checks `result.status === "duplicate"` and shows "already uploaded" instead of "indexed".
- **Upsert safety:** Even if duplicate detection is bypassed, ChromaDB's `upsert` overwrites existing chunks with the same deterministic IDs (hash of `document_id + chunk_index`).
- **Lesson:** Content-based IDs are an old idea (Git uses SHA-1 for commits, Docker uses SHA-256 for layers). The pattern: hash the content, use the hash as the identity. If two things have the same hash, they're the same thing — don't store it twice.

### Files Modified
- `src/ai_document_agent/main.py` — SHA-256 hashing, duplicate detection
- `static/app.js` — duplicate status handling in upload flow

---

## Phase 4.4 — Evidence Abstraction (Completed)

### 4.4.1 Problem
- Search results were raw Python dicts with inconsistent keys (`rrf_score`, `bm25_score`, `distance`). No autocomplete, no typo protection, unclear what fields a result actually has.

### 4.4.2 Fix: Evidence Dataclass
- **`Evidence` dataclass** in `pdf_processor.py` with four fields: `text` (chunk content), `source` (filename), `page` (page number), `score` (RRF relevance score).
- **`search_documents()` returns `list[Evidence]`** instead of `list[dict]`. Internal helper functions (`_semantic_search`, `BM25Index.search`, `_reciprocal_rank_fusion`) still use dicts internally — only the public API boundary converts to Evidence.
- **`agent.py` uses attribute access** — `result.score` instead of `result.get("rrf_score", 0)`, `result.source` instead of `result['source']`.
- **Lesson:** Dataclasses are Python's lightweight way to define structured data. They give you autocomplete, typo protection, and self-documenting code. Use them at API boundaries — the point where one module talks to another. Internal helpers can use whatever is convenient.

### Files Modified
- `src/ai_document_agent/pdf_processor.py` — Evidence dataclass, search_documents() return type
- `src/ai_document_agent/agent.py` — Evidence import, attribute access in build_context_prompt()

---

## Phase 4 Complete

All Phase 4 improvements:
- 4.1: Hybrid Retrieval (BM25 + Semantic + RRF)
- 4.2: Conversation-Aware Retrieval (heuristic query enrichment)
- 4.3: SHA-256 Content-Based Document IDs (duplicate prevention)
- 4.4: Evidence Abstraction (structured search results)

---

## Phase 5 — Document Understanding (Completed)

### 5.1 Table Extraction
- **Status:** Done
- **What:** Added `_extract_tables_from_page(page)` using PyMuPDF's `find_tables()` method. Regular `get_text()` garbles tables (columns collapse into messy lines). `find_tables()` detects table structures by analyzing cell boundaries and grid lines, returning proper rows and columns.
- **Format:** Tables are wrapped in `[TABLE]...[/TABLE]` markers in the extracted text. The LLM sees structured data like `Name | Marks | Grade` instead of garbled text.
- **Lesson:** PyMuPDF has table detection built in since v1.23 — no need for external libraries like tabula or camelot. The key is using the right API for the right task: `get_text()` for paragraphs, `find_tables()` for tables.

### 5.2 Heading/Section Detection
- **Status:** Done
- **What:** Added `_extract_text_with_headings(page)` which uses font size analysis via `get_text("dict")`. Finds the body font size (most common), then marks any text 15%+ larger as a heading with `[SECTION: heading text]` markers.
- **How it works:** `get_text("dict")` returns every text span with its font size, font name, and position. We find the mode (most common) font size = body text. Anything significantly larger = heading. Short text (<200 chars) with large font = section title.
- **For DOCX:** Even simpler — `python-docx` exposes paragraph styles (`Heading 1`, `Heading 2`, etc.), so no font size guessing needed.
- **Lesson:** Document structure (headings, sections) helps the LLM give better answers. Instead of "the document says X", it can say "in the Income Details section on page 3, the document says X." The markers become part of the chunk text naturally.

### 5.3 Multi-Format Support
- **Status:** Done
- **What:** Added support for DOCX, TXT, and CSV alongside PDF.
  - `extract_text_from_txt()` — reads plain text files (page=1)
  - `extract_text_from_csv()` — reads CSV as tabular data wrapped in `[TABLE]` markers
  - `extract_text_from_docx()` — reads Word documents with heading detection from paragraph styles + table extraction
  - `process_document()` — new dispatcher that picks the right extractor based on file extension, then runs the chunk → embed → store pipeline
  - `SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx"}`
- **Frontend:** Upload zone now shows "PDF, DOCX, TXT, CSV". File input accepts all four types.
- **Dependency:** Added `python-docx>=1.0` to `pyproject.toml`. Run `uv sync` to install.
- **Lesson:** The extract → chunk → embed → store pipeline is format-agnostic — only the extraction step changes per format. Good separation of concerns means adding a new format is just one new function + one line in the dispatcher.

### Files Modified
- `src/ai_document_agent/pdf_processor.py` — table extraction, heading detection, multi-format extractors, process_document()
- `src/ai_document_agent/main.py` — multi-format upload validation, process_document() import
- `static/index.html` — file input accepts all formats, updated text
- `static/app.js` — multi-format file validation in upload handler
- `pyproject.toml` — added python-docx dependency

---

## Phase 5.5 — GUI Improvements (Completed)

### 5.5.1 Typing Animation
- **Status:** Done
- **What:** Added bouncing dots animation (three pulsing circles) that shows in the AI message bubble while waiting for the first token (~29 seconds on CPU). Dots disappear automatically when the first token arrives.
- **How:** CSS `@keyframes typingBounce` with staggered `animation-delay` on three `<span>` elements. Inserted into `answerEl` in `sendMessage()`, removed on first `token` event in `handleStreamEvent()`.
- **Lesson:** Perceived performance matters as much as real performance. The same 29-second wait feels much shorter when the user sees animated activity vs a static "Thinking..." label.

### 5.5.2 Markdown Table Rendering
- **Status:** Done
- **What:** Enhanced `formatText()` in app.js to detect markdown table syntax (`| col | col |`) and convert it to styled HTML `<table>` elements. Also added inline `code` backtick support.
- **How:** `formatText()` now splits text into lines, detects consecutive table lines (starting and ending with `|`), and delegates to `buildMarkdownTable()` which parses headers, separators, and data rows into `<table>/<thead>/<tbody>` HTML. `formatLine()` handles bold, italic, and code inline formatting.
- **CSS:** `.md-table` with alternating row colors, hover highlights, rounded borders, sticky headers. `.table-wrapper` for horizontal overflow scrolling on mobile.
- **Lesson:** Never try to handle markdown rendering with a single regex replace chain. Tables need line-by-line parsing with state tracking (am I inside a table? is this a separator row?). Split the problem: `formatText()` detects blocks, `buildMarkdownTable()` handles table structure, `formatLine()` handles inline formatting.

### Files Modified
- `static/app.js` — typing indicator, markdown table rendering, inline code support
- `static/styles.css` — typing animation CSS, table styles, code styles

---

## Phase 6 — Data Analysis Engine (Completed)

### 6.1 Problem
When users ask analytical questions about CSV data ("how many Hard questions?", "average score?", "count by category"), the normal RAG flow retrieves 5 random text chunks from the table. The LLM tries to count or compute from these fragments — and gets it wrong. It's like trying to calculate a spreadsheet total by reading 5 random cells.

### 6.2 Solution: Python-Computed Analysis
Instead of making the LLM guess from chunks, we load the FULL CSV data in Python and compute real answers:
- **Column type detection** — samples first 10 values to classify each column as numeric or categorical
- **Numeric stats** — min, max, average, sum, count for every numeric column
- **Categorical breakdowns** — unique value counts (e.g., "Hard (30), Medium (45), Easy (25)")
- **Raw data inclusion** — all rows if ≤40, first 15 if larger

### 6.3 How It Works
1. User asks a question about a CSV document
2. `build_context_prompt()` detects the source is a CSV file
3. `data_analyzer.py` loads the CSV using Python's `csv.DictReader`
4. Column statistics and raw data are computed in <1ms
5. Results are injected into the LLM context WITH the label "DATA ANALYSIS (computed by Python, 100% accurate)"
6. The LLM uses the pre-computed numbers to format its answer — no guessing

### 6.4 Key Design Decisions
- **No extra LLM call** — analysis runs in pure Python, zero latency cost. On CPU where each LLM call takes 70-80s, this matters enormously.
- **Alongside chunks, not replacing them** — analysis context is appended after chunk excerpts. Chunks help with "what does this mean?" questions, analysis helps with "how many?" questions.
- **No pandas dependency** — stdlib `csv` module is enough. Keeps the install lightweight and avoids the 50MB pandas dependency.
- **System prompt updated** — Rule 5 tells the LLM: "When DATA ANALYSIS results are provided, use those computed values (they are 100% accurate). Do not try to recount or recalculate from the raw text."
- **File path lookup** — `find_upload_path()` scans the uploads directory for `{hash}_{filename}` pattern to locate the original CSV file.

### 6.5 New Module
- `src/ai_document_agent/data_analyzer.py` — self-contained analysis engine
  - `load_csv(file_path)` → (headers, rows) tuple
  - `build_analysis_context(file_path)` → formatted analysis text
  - `find_upload_path(filename, upload_dir)` → file path on disk
  - `is_tabular_file(filename)` → True for CSV files
  - `_detect_column_type(rows, column)` → "numeric" or "categorical"

### Files Modified
- `src/ai_document_agent/data_analyzer.py` — NEW: analysis engine
- `src/ai_document_agent/agent.py` — imports data_analyzer, injects analysis in build_context_prompt(), updated system prompt
- `PROGRESS.md` — documented Phase 6

### Lesson
The best AI systems know when NOT to use AI. Counting rows in a CSV is not an AI problem — it's a Python one-liner. The pattern: use Python for computation, use the LLM for language. Each does what it's best at.

---

## Phase 7 — Real Agent: Autonomous Tool Selection (Completed)

### 7.1 Problem
The system was a smart chatbot, not an agent. Python heuristics decided which tools to offer (e.g., `_needs_calculator()` only offered the calculator when math keywords appeared). The LLM had no autonomy — it could only answer from chunks or use the calculator. Users wanted a REAL agent where the LLM decides which tools to call on document data.

### 7.2 Solution: LLM-Driven Tool Selection
Removed all Python heuristic gating. ALL tool schemas are included in every LLM call. The LLM sees the tools and autonomously decides:
- Answer directly from retrieved chunks (most common, fastest)
- Call `filter_rows` to filter tabular data by column/value
- Call `aggregate_data` to compute stats with optional group-by
- Call `search_more` to search documents with a refined query
- Call `get_page_content` to retrieve full page text
- Call `calculator` for arithmetic on document numbers
- Call multiple tools in sequence (multi-step reasoning)

### 7.3 New Document Tools

**filter_rows(column, value)**
Filter CSV rows by column value. Supports exact match ("Hard") and numeric comparisons (">90", "<=50"). Returns filtered rows as formatted text.

**aggregate_data(column, operation, group_by?)**
Compute count/sum/average/min/max on a CSV column, optionally grouped by another column. Examples: "count by Difficulty", "average Score per Subject".

**search_more(query, n_results?)**
Search uploaded documents with a custom query. The LLM uses this to refine search when the initial forced retrieval didn't find the right chunks. Respects the active source_filter.

**get_page_content(page_number)**
Get ALL chunks from a specific page in the active document. Queries ChromaDB by page metadata. Useful for "what's on page 3?" or when the LLM needs more context around a retrieved excerpt.

### 7.4 Multi-Step Tool Loop
The LLM can now call tools in sequence — up to MAX_TOOL_ROUNDS=3:
1. LLM sees chunks + tool schemas → calls filter_rows
2. LLM sees filtered data → calls aggregate_data
3. LLM sees aggregated stats → writes final answer

Each round is one LLM call (~70-80s on CPU). Most questions need 0 rounds (direct answer). Complex analytical questions might use 1-2 rounds.

### 7.5 Architecture Changes

**What changed:**
- `SYSTEM_PROMPT` updated — tells the LLM it's an "agent" with tools, instructs it to decide: answer directly or call a tool
- `TOOLS` list — expanded from 1 tool (calculator) to 5 tools (calculator + 4 document tools)
- `TOOL_REGISTRY` — maps all 5 tool names to Python implementations
- `_needs_calculator()` — REMOVED. No more Python heuristic gating.
- `_active_source_filter` — module-level variable set per-request so tool functions know which document is active
- `ask_agent()` — rewritten with multi-step loop (MAX_TOOL_ROUNDS iterations)
- `stream_agent()` — rewritten with multi-step streaming. First round streams; subsequent rounds use non-streaming for faster tool decisions.
- `_execute_tool_streaming()` — new helper that executes one tool and yields status events

**What stayed the same:**
- Forced retrieval — search ALWAYS happens first (unchanged)
- Context limiting — still sends only last 6 messages
- CSV data analysis — still injects Python-computed stats for CSV files
- Calculator — same implementation, now just one of 5 tools
- Streaming — tokens still appear in real-time for direct answers

**New imports:**
- `csv`, `json` — for tool implementations
- `load_csv` from `data_analyzer` — used by filter_rows and aggregate_data
- `collection` from `pdf_processor` — used by get_page_content to query ChromaDB

### 7.6 Key Design Decisions
- **All tools always available** — the LLM is smart enough to ignore irrelevant tools. The schemas are small (~500 tokens total). This is what makes it a REAL agent vs a routed chatbot.
- **Forced retrieval kept** — search still happens before the LLM sees anything. This is reliable and fast. The LLM can additionally call `search_more` if it wants different results.
- **Tool errors are non-fatal** — if a tool fails, the error message is returned to the LLM as a tool result. The LLM can try a different approach or explain the error.
- **Multi-step capped at 3** — prevents runaway tool loops. On CPU, 3 rounds = ~4 minutes worst case. Most questions resolve in 1 round.
- **Tools operate on real data** — filter_rows loads the actual CSV file, aggregate_data computes real numbers, get_page_content queries ChromaDB. No fake data, no approximations.

### Files Modified
- `src/ai_document_agent/agent.py` — complete rewrite of tool system: 5 tool schemas, 5 implementations, multi-step loop, updated system prompt

### Lesson
The difference between a chatbot and an agent is WHO makes the decisions. A chatbot has hardcoded routing (Python if/else decides). An agent has LLM-driven routing (the LLM sees tool schemas and decides). The agent is more flexible — it handles questions you didn't anticipate — but costs more LLM calls. The design balances this: forced retrieval (cheap, always works) + optional tools (expensive, only when needed).

---

## Phase 8 — Production Hardening (Completed)

### 8.1 Security Fixes
- **Status:** Done (2026-08-19)
- **What:** Three security improvements to the upload endpoint:
  1. **Filename sanitization** — `_sanitize_filename()` strips directory traversal (`../../etc/passwd`), Windows paths (`C:\Windows\...`), null bytes, and control characters. Only the final basename survives. Prevents an attacker from writing files outside the uploads directory.
  2. **File size limit** — `MAX_UPLOAD_BYTES = 20 MB`. File bytes are checked immediately after reading; oversized uploads get a 400 response before any processing. Prevents memory exhaustion and disk filling.
  3. **Safe filename threading** — all references to `file.filename` in the upload handler replaced with `safe_filename` so the sanitized name propagates through the entire flow.
- **Lesson:** Never trust user input, especially filenames. `Path("../../etc/passwd").name` → `"passwd"` is your first line of defense. File size limits should be checked as early as possible in the request lifecycle.

### 8.2 Concurrency Fix — _active_source_filter
- **Status:** Done (2026-08-19)
- **What:** Replaced the `_active_source_filter` module-level global variable with a per-request tool registry pattern.
  - **Before:** `_active_source_filter` was a module-level variable set at the start of each request. In concurrent requests (FastAPI is async), Request B could overwrite the filter before Request A's tools finished executing — causing tools to query the wrong document.
  - **After:** `_build_tool_registry(source_filter)` returns a dict of tool functions with `source_filter` captured in closures. Each request gets its own registry. No shared mutable state.
  - Updated `filter_rows()`, `aggregate_data()`, `search_more_tool()`, `get_page_content()` to accept `source_filter` as a parameter instead of reading a global.
  - Updated `_execute_tool_streaming()` to accept and forward the registry.
  - Fixed `callable` → `Callable` (proper type from `collections.abc`).
- **Lesson:** Module-level mutable state is a concurrency bug waiting to happen. In web applications, each request should carry its own state. Closures are a clean way to bind per-request data to tool functions without threading parameters through every call.

### 8.3 Dead Code Cleanup
- **Status:** Done (2026-08-19)
- **What:** Removed unused code that accumulated during development:
  1. **`generate_summary()`** in `main.py` — 140 lines of dead code. Was disabled in Phase 3.9.3 but the function body remained. Also removed unused imports: `ollama_chat`, `MODEL`, `extract_text_from_pdf`.
  2. **`process_pdf()`** wrapper in `pdf_processor.py` — one-line wrapper around `process_document()` marked as "backward-compatible". Nothing called it.
  3. **`import csv as csv_mod`** in `pdf_processor.py` — local re-import inside `extract_tables_to_csv()` that shadowed the top-level `import csv`. Replaced with direct use of the already-imported `csv` module.
- **Lesson:** Dead code is technical debt with interest. It confuses new readers ("is this called somewhere?"), creates false IDE references, and can mask real bugs in grep results. Delete it — git remembers.

### 8.4 Tests for Agent Module
- **Status:** Done (2026-08-19)
- **What:** Added `tests/test_agent_tools.py` with tests for pure functions in agent.py:
  - `TestEnrichQuery` (10 tests) — standalone vs follow-up detection, referential words, connecting words, short questions, empty/None handling, enrichment format
  - `TestSanitizeFilename` (8 tests) — path traversal, Windows paths, null bytes, empty strings, whitespace
  - `TestCalculatorEdgeCases` (4 tests) — small floats, negative division, both-negative, float return type
- **Run with:** `uv run pytest tests/ -v`
- **Lesson:** Test pure functions first — they're fast, deterministic, and catch the most bugs per line of test code. The functions that need mocking (Ollama, ChromaDB) are harder to test but less likely to have logic bugs.

### Files Modified
- `src/ai_document_agent/agent.py` — `_enrich_query()`, `_build_tool_registry()`, `Callable` import, removed `_active_source_filter` global
- `src/ai_document_agent/main.py` — `_sanitize_filename()`, `MAX_UPLOAD_BYTES`, removed `generate_summary()` and unused imports
- `src/ai_document_agent/pdf_processor.py` — removed `process_pdf()` wrapper, fixed `import csv` shadowing
- `tests/test_agent_tools.py` — NEW: 22 tests for agent tools and security
- `PROGRESS.md` — documented Phase 8

---

## Phase 9 — CPU Performance Optimization (Completed)

### Problem
LLM inference on CPU (Ryzen 5 5500U, no GPU) took ~166s per question with qwen3:8b. Unacceptable for interactive use.

### 9.1 Model Switch
- **Changed:** `MODEL = "qwen3:8b"` → `MODEL = "qwen3:4b"`
- **Why:** 4B parameter model runs ~2x faster on CPU. Quality is sufficient for structured data questions.
- **User action required:** Run `ollama pull qwen3:4b` once.

### 9.2 Skip Chunk Retrieval for CSV Files
- **Changed:** `build_context_prompt()` skips ChromaDB search when source is a CSV file.
- **Why:** For CSVs, chunks are random table fragments — useless. The data analysis context (column stats, sample rows, value counts) is what the LLM needs. Skipping retrieval saves ~2-3s per query and removes noise from the prompt.

### 9.3 Reduce Retrieval Chunks (5 → 3)
- **Changed:** `n_results=5` → `n_results=3` in `build_context_prompt()`.
- **Why:** Fewer chunks = fewer tokens = faster inference. 3 chunks still provide enough context for most questions. The LLM can always call `search_more` if it needs more.

### 9.4 Compress System Prompt and Tool Schemas
- **Changed:** SYSTEM_PROMPT reduced from ~350 tokens to ~60 tokens. All 5 tool schemas compressed — removed verbose descriptions, kept only essential info.
- **Why:** Every token in the system prompt is processed on every call. Smaller prompt = faster first-token time. The LLM understands tool schemas fine with minimal descriptions.

### 9.5 Set Ollama Threading
- **Added:** `LLM_OPTIONS = {"num_thread": 6}` passed to all `chat()` calls.
- **Why:** Ryzen 5 5500U has 6 physical cores. Ollama defaults to auto-detect but explicit setting ensures all cores are used.

### Expected Impact
- Model switch: ~2x faster (166s → ~80s)
- Context reduction (9.2 + 9.3 + 9.4): ~20-30% faster on top
- Threading (9.5): ensures full CPU utilization
- Combined: ~60-70s per question (down from 166s)

### Files Modified
- `src/ai_document_agent/agent.py` — all changes above

---

## Future Phases

- **Phase 10:** Advanced features — OCR for scanned PDFs, multi-user auth, PostgreSQL sessions, background processing, deployment