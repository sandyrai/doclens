# AI Document Agent — Local Setup Guide

Step-by-step instructions to set up and run this project on your machine.

---

## Prerequisites

You need these installed before starting:

### 1. Python 3.12

Download from https://www.python.org/downloads/ or use a version manager.

Verify:
```
python --version
```
Expected: `Python 3.12.x`

### 2. uv (Python package manager)

Install uv (replaces pip + venv in one fast tool):

```
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:
```
uv --version
```

### 3. Git

Download from https://git-scm.com/downloads

Verify:
```
git --version
```

### 4. Ollama

Download from https://ollama.ai

After installing, pull the required models:
```
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

**What are these models?**
- `qwen3:8b` — The main chat/reasoning model. It reads your questions and generates answers.
- `nomic-embed-text` — The embedding model (~270MB). It converts text into vectors (lists of numbers) for semantic search. Used when you upload PDFs.

Verify Ollama is running:
```
ollama list
```
You should see both `qwen3:8b` and `nomic-embed-text` in the list.

---

## Choosing your LLM (local or cloud)

The app works with local Ollama by default. To use another LLM (LM Studio,
OpenRouter, Gemini, Claude, OpenAI, Groq, or any OpenAI-compatible server):

1. Copy the template: `copy .env.example .env` (macOS/Linux: `cp .env.example .env`)
2. Set `LLM_PROVIDER`, `LLM_MODELS` and, for cloud providers, your API key.
3. Test it: `uv run python -m ai_document_agent.llm_provider`

See the README's "Choosing an LLM" section for examples. Embeddings for
document search always use Ollama (`nomic-embed-text`), so keep Ollama
installed even when a cloud model answers the questions.

---

## Project Setup

### Step 1: Clone the repository

```
git clone <your-repo-url>
cd ai-document-agent
```

Or if you already have the project:
```
cd E:\ai-document-agent
```

### Step 2: Create virtual environment and install dependencies

```
uv sync
```

This single command does everything:
- Creates a `.venv` virtual environment (if it doesn't exist)
- Installs all dependencies from `pyproject.toml`
- Locks versions in `uv.lock`

**What is `uv sync`?**
Think of it like `npm install` for Python. It reads `pyproject.toml` (like `package.json`), creates an isolated environment, and installs exactly the right packages.

### Step 3: Verify the virtual environment

```
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

After activation, your terminal prompt should show `(.venv)` at the beginning.

**Note:** You don't need to manually activate the venv if you use `uv run` (see below). `uv run` automatically uses the project's virtual environment.

---

## Running the Application

### Option 1: Using `uv run` (Recommended)

```
uv run uvicorn ai_document_agent.main:app --reload
```

**What this does:**
- `uv run` — runs the command inside the project's virtual environment (no manual activation needed)
- `uvicorn` — the ASGI server that runs FastAPI
- `ai_document_agent.main:app` — tells uvicorn to find the `app` object in `src/ai_document_agent/main.py`
- `--reload` — auto-restarts when you edit code (great for development, don't use in production)

### Option 2: Manual activation + run

```
# Activate venv first
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Then run
uvicorn ai_document_agent.main:app --reload
```

### What you should see

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx]
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## Using the Application

### Browser UI (primary)

Open: http://127.0.0.1:8000/

This is the chat interface. Type a question and click Send.

### Swagger API docs

Open: http://127.0.0.1:8000/docs

Interactive API documentation. You can test endpoints directly from here.

### Health check

Open: http://127.0.0.1:8000/health

Should return: `{"status": "healthy"}`

---

## Testing It Works

Try these questions in order to verify everything:

1. **Simple question (no tool):**
   "What is Python?"
   → Should get a direct text answer.

2. **Calculator question (tool calling):**
   "What is 25 times 48?"
   → Should show tool status events, then answer 1200.

3. **Follow-up (conversation state):**
   "Now divide that by 5"
   → Should answer 240, proving it remembers the previous answer.

4. **Symbol operation (input normalization):**
   "20 + 25 = ?"
   → Should answer 45.

5. **PDF upload:**
   Drag a PDF file onto the upload area (or click to browse).
   → Should show "X pages, Y chunks indexed" when done.

6. **Document question (RAG):**
   After uploading a PDF, ask a question about its content.
   → Should search the document, find relevant passages, and answer with context.

---

## Stopping the Server

Press `Ctrl+C` in the terminal where uvicorn is running.

---

## Adding New Dependencies

```
uv add <package-name>
```

Example:
```
uv add chromadb
```

This updates `pyproject.toml` and `uv.lock` automatically.

---

## Project Structure

```
ai-document-agent/
│
├── src/
│   └── ai_document_agent/
│       ├── __init__.py        # Package marker
│       ├── agent.py           # Agent logic, LLM interaction, tool execution
│       ├── main.py            # FastAPI app, routes, session management
│       └── pdf_processor.py   # PDF extraction, chunking, embeddings, ChromaDB
│
├── static/
│   └── index.html             # Browser chat UI (with PDF upload)
│
├── tests/
│   ├── __init__.py            # Test package marker
│   ├── test_api.py            # API endpoint tests
│   └── test_calculator.py     # Calculator unit tests
│
├── uploads/                   # Uploaded PDF files (auto-created)
├── chroma_db/                 # ChromaDB vector database (auto-created)
│
├── .env                       # Environment variables (empty for now)
├── .gitignore                 # Git ignore rules
├── .python-version            # Python version for uv
├── pyproject.toml             # Project config + dependencies
├── uv.lock                   # Locked dependency versions
├── PROGRESS.md                # What's been built and learned
├── SETUP.md                   # This file
└── README.md                  # Project readme
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ai_document_agent'"
Run `uv sync` to install the project package, or use `uv run` instead of calling `uvicorn` directly.

### "Connection refused" or "Cannot connect to Ollama"
Make sure Ollama is running. Open a separate terminal and run:
```
ollama serve
```

### "Model not found: qwen3:8b"
Pull the model:
```
ollama pull qwen3:8b
```

### Server won't start on port 8000
Another process might be using that port. Either stop it, or run on a different port:
```
uv run uvicorn ai_document_agent.main:app --reload --port 8001
```

### Changes not reflected in browser
Hard refresh the browser with `Ctrl+F5` to clear cached HTML/CSS/JS.
