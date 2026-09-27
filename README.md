# AI Document Agent

Upload PDFs, Word files or spreadsheets and ask questions about them. The agent
retrieves the relevant passages (hybrid BM25 + vector search), can call data
tools on extracted tables, and streams the answer back to a web UI.

**Bring your own LLM.** You can run it fully locally (Ollama, LM Studio) or with any cloud
model, free or paid: OpenRouter, Google Gemini, Anthropic Claude, OpenAI,
Groq, or any other OpenAI-compatible endpoint. You switch providers by editing `.env`;
no code changes are needed.

## Quick start

```bash
git clone https://github.com/<you>/ai-document-agent.git
cd ai-document-agent
cp .env.example .env          # Windows: copy .env.example .env
uv sync
```

Install [Ollama](https://ollama.com) and pull the embedding model (used for
document search regardless of which LLM answers questions):

```bash
ollama pull nomic-embed-text
```

Then choose an LLM (next section), test it, and start the server:

```bash
uv run python -m ai_document_agent.llm_provider   # sends one test prompt
uv run uvicorn ai_document_agent.main:app --reload
```

Open http://127.0.0.1:8000. See [SETUP.md](SETUP.md) for a detailed walkthrough
(Tesseract OCR, troubleshooting).

## Choosing an LLM

All LLM settings live in `.env`. Three variables matter:

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `ollama`, `lmstudio`, `openrouter`, `gemini`, `anthropic`, `openai`, `groq`, or `custom` |
| `LLM_MODELS` | One or more model names, comma-separated, **tried in order** |
| `LLM_API_KEY` | Your key (cloud only). Or use the provider's own variable, e.g. `GEMINI_API_KEY` |

**Local, free, private (default):**
```env
LLM_PROVIDER=ollama
LLM_MODELS=qwen3:8b
```
Run `ollama pull qwen3:8b` first. Ollama runs on CPU without a GPU, but slowly.

**LM Studio:** start its local server, then
```env
LLM_PROVIDER=lmstudio
LLM_MODELS=<model id shown in LM Studio>
```

**Cloud examples:**
```env
LLM_PROVIDER=gemini
LLM_MODELS=gemini-2.5-flash
GEMINI_API_KEY=your-key
```
```env
LLM_PROVIDER=anthropic
LLM_MODELS=claude-haiku-4-5
ANTHROPIC_API_KEY=your-key
```
```env
LLM_PROVIDER=openrouter
LLM_MODELS=first-choice-model:free,backup-model:free,paid-model
OPENROUTER_API_KEY=your-key
```

**Any other OpenAI-compatible server** (vLLM, llama.cpp, Together, DeepSeek, …):
```env
LLM_PROVIDER=custom
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODELS=my-model
LLM_API_KEY=optional
```

Model names change often, especially free ones. Check your provider's current
model list. The app needs a model that supports **tool/function calling** for
the table-analysis features.

### What happens when a model fails

`llm_provider.py` handles failures on its own:

- **429 rate limit**: waits and retries the same model, then puts it in a
  5-minute cooldown so later requests skip it.
- **404 / model removed / server error**: moves to the next model in `LLM_MODELS`.
- **401/403 bad key** or **server not reachable**: stops immediately with a
  clear message, since trying other models wouldn't help.
- **Stream breaks mid-answer**: reports an error instead of switching models,
  so the answer doesn't appear twice.

Error messages shown in the browser never include keys or raw provider
responses. Full details go to the server log. All retry settings can be tuned
in `.env` (see `.env.example`).

## Security

- Keys belong only in `.env`, which is git-ignored. `.env.example` holds
  placeholders only.
- Uploaded files, the vector DB (`chroma_db/`), chat history (`data/`) and the
  OCR cache are git-ignored as well.

## Tests

```bash
uv run pytest
```

`tests/test_llm_provider.py` covers provider configuration and the
retry/cooldown/fallback logic with fake models. It needs no network.
