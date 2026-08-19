# AI Document Agent — Project Context & Claude Cowork Instructions

## 1. Project summary

**Project name:** AI Document Agent

**Current stage:** Local learning/prototype stage

**Developer profile:** Beginner in AI agents/RAG, but comfortable with Python and Java.

**Operating system:** Windows

**Primary backend:** Python + FastAPI

**Local LLM runtime:** Ollama

**Current local model:** `qwen3:8b`

**Hardware:** HONOR laptop, AMD Ryzen 5 5500U, 16 GB RAM, integrated AMD Radeon Graphics. Ollama currently runs Qwen3 8B on CPU.

The purpose of this project is to learn how to build a real AI agent from the ground up rather than immediately depending on a high-level agent framework.

---

# 2. Long-term goal

Build a general-purpose **AI Document Agent** that can accept uploaded PDF documents and answer questions about them using:

- document understanding
- RAG (Retrieval-Augmented Generation)
- tool calling
- structured data extraction
- calculations
- document/page/source references
- conversational interaction
- agentic decision making

The initial use case is **bank/account statements**, but the architecture must NOT be hard-coded around bank statements.

The long-term scope is:

> Upload any supported PDF → understand/index it → retrieve relevant information → use tools when necessary → reason over the evidence → produce a useful answer.

Examples of future questions:

- "What was my total spending last month?"
- "How much did I spend on restaurants?"
- "Show transactions above ₹50,000."
- "Compare expenses between January and February."
- "Find all recurring payments."
- "Summarize this agreement."
- "What are the important clauses in this contract?"
- "What are the key risks mentioned in this report?"
- "Find all dates and amounts related to a particular topic."

The architecture should therefore be document-agnostic.

---

# 3. Learning objective

The immediate goal is **not monetization** and not production deployment.

The priority is to understand and build:

1. LLM interaction
2. Tool calling
3. Agent loops
4. Streaming
5. FastAPI integration
6. Frontend integration
7. PDF ingestion
8. Text extraction
9. Chunking
10. Embeddings
11. Vector search
12. RAG
13. Agent + RAG
14. Structured data analysis
15. Evaluation and reliability
16. Production architecture
17. Optional MCP integration if justified

Do not introduce unnecessary frameworks before the underlying concepts are understood.

---

# 4. Current technology stack

## Development environment

- Windows
- VS Code
- Git
- Python 3.12.14
- `uv` for Python environment/package management

## Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

## AI runtime

- Ollama
- Qwen3 8B (`qwen3:8b`)
- Local inference
- CPU execution on current laptop

## Frontend

Currently intentionally simple:

- HTML
- CSS
- Vanilla JavaScript
- Fetch API
- NDJSON streaming

A modern frontend framework can be introduced later if justified.

## Current project structure

```text
ai-document-agent/
│
├── src/
│   └── ai_document_agent/
│       ├── __init__.py
│       ├── agent.py
│       └── main.py
│
├── static/
│   └── index.html
│
├── .env
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

This structure is intentionally small at the current learning stage.

---

# 5. Architecture currently built

```text
                    Browser
                       │
                       │ HTTP
                       ▼
                  FastAPI
                       │
              ┌────────┴────────┐
              │                 │
           /chat          /chat/stream
              │                 │
              ▼                 ▼
         ask_agent()      stream_agent()
              │                 │
              └────────┬────────┘
                       ▼
                    Ollama
                       │
                       ▼
                   Qwen3 8B
                       │
                       ▼
                  Tool Calling
                       │
                       ▼
                  Calculator
```

The frontend is served by FastAPI.

Swagger remains available for API development/testing, but the browser UI is now the primary user-facing interface.

---

# 6. What has already been built

## 6.1 Python environment

Working:

```text
Python 3.12.14
uv
Git
VS Code
```

The project uses a `.venv` managed through `uv`.

## 6.2 Ollama

Ollama is installed and working.

Current model:

```text
qwen3:8b
```

The model has tool-calling capability.

## 6.3 Basic LLM interaction

Python successfully communicates with Ollama using the Ollama Python package.

## 6.4 Calculator tool

A Python calculator tool has been created:

```python
def calculator(operation: str, a: float, b: float) -> float:
    ...
```

Supported concepts:

- add
- subtract
- multiply
- divide

The agent can ask the model whether a calculator tool is required.

## 6.5 Agent tool-calling loop

The current non-streaming agent follows:

```text
User question
      ↓
Qwen
      ↓
Tool decision
      ↓
Calculator
      ↓
Tool result
      ↓
Qwen
      ↓
Final answer
```

This was successfully tested.

## 6.6 Performance measurement

Timing has been added to understand local inference performance.

Typical results on the current laptop have been approximately:

```text
LLM decision:       ~27 seconds
Tool execution:      ~0 seconds
Final LLM:           ~9 seconds
Total:              ~37 seconds
```

A simple non-tool question has also been observed at around 3–4 seconds.

The important observation is:

> Tool-calling agent workflows can be substantially slower because they may require multiple LLM inference calls.

Qwen3 8B is currently CPU-bound on this laptop.

## 6.7 Streaming experiment

Streaming from Ollama was successfully tested.

Observed example:

```text
First token: ~14.7 seconds
Total:       ~25.5 seconds
```

The application can therefore stream output instead of keeping the browser completely blank until generation finishes.

## 6.8 FastAPI

FastAPI is working.

Installed versions at the current stage:

```text
FastAPI 0.141.1
Uvicorn 0.52.3
```

Working endpoints include:

```text
GET /
GET /health
POST /chat
POST /chat/stream
```

Swagger is available at:

```text
http://127.0.0.1:8000/docs
```

## 6.9 Browser UI

A simple browser chat UI has been built using:

- HTML
- CSS
- JavaScript
- Fetch API
- streaming NDJSON events

The UI displays:

- user messages
- AI messages
- agent status
- first-token timing
- total response time
- errors

The frontend is served at:

```text
http://127.0.0.1:8000/
```

## 6.10 Streaming agent events

The streaming backend uses events such as:

```text
thinking
decision_completed
tool_call
tool_result
generating
first_token
token
completed
error
```

The intent is to expose **application/agent status**, not private chain-of-thought.

---

# 7. Current important issue

The streaming agent has recently been changed to use tool calling.

A test exposed this error:

```text
ValueError: Unknown operation: sum
```

The model selected:

```text
operation = "sum"
```

while the original calculator only accepted:

```text
add
subtract
multiply
divide
```

This demonstrates an important production principle:

> LLM-generated tool arguments must be validated.

A defensive improvement was proposed to accept reasonable synonyms such as:

```text
sum / plus / add
minus / subtract
times / multiply
div
```

A further improvement should be made later using stricter tool schemas/enums.

The frontend also needs to receive structured error events instead of exposing a generic browser "network error" when the generator fails.

**Do not assume this issue is fully resolved unless the user confirms the fix has been tested.**

---

# 8. Current backend design principle

FastAPI should NOT contain AI orchestration logic.

Good separation:

```text
main.py
  → HTTP/API/frontend serving

agent.py
  → agent orchestration
  → LLM interaction
  → tool execution

future tools/
  → calculator
  → PDF search
  → database
  → analysis

future rag/
  → ingestion
  → chunking
  → embeddings
  → retrieval
```

The API layer should not directly call Ollama.

This allows the model provider/runtime to change later without rewriting the API.

---

# 9. Target architecture

The intended future architecture is:

```text
                         User
                          │
                          ▼
                     Web Frontend
                          │
                          ▼
                       FastAPI
                          │
                          ▼
                    Agent Orchestrator
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
           LLM           RAG          Tools
            │             │             │
            │             │             ├── Calculator
            │             │             ├── SQL
            │             │             └── Analysis
            │             │
            │             ├── Vector Store
            │             ├── Embeddings
            │             └── Document Store
            │
            ▼
       Local / Cloud LLM
```

Document pipeline:

```text
PDF Upload
    │
    ▼
Document Validation
    │
    ▼
PDF Extraction
    │
    ▼
Text + Metadata
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector Database
    │
    ▼
Retriever
    │
    ▼
Agent
    │
    ▼
Answer + Sources
```

---

# 10. RAG goal

RAG is a core requirement.

The future system should not simply place an entire PDF into an LLM prompt.

Instead:

```text
PDF
 ↓
Extract text
 ↓
Split into chunks
 ↓
Create embeddings
 ↓
Store vectors
 ↓
User asks question
 ↓
Retrieve relevant chunks
 ↓
Agent uses evidence
 ↓
LLM generates answer
```

The answer should ideally provide document/page/source references.

Example:

```text
Your restaurant spending was ₹42,350.

Sources:
- Statement page 4
- Statement page 7
- Statement page 12
```

This is especially important for financial documents.

---

# 11. Statement-specific future capabilities

The first serious domain implementation will be bank/account statements.

Potential pipeline:

```text
PDF
 ↓
Extract text/tables
 ↓
Identify transactions
 ↓
Normalize transaction records
 ↓
Store structured data
 ↓
Create searchable document representation
 ↓
RAG + SQL/Python analysis
 ↓
Agent
```

Potential structured transaction fields:

```text
date
description
merchant
debit
credit
balance
account/reference information
page number
source document
```

Then questions can be answered using the right mechanism.

Example:

```text
"What is my total debit?"
       ↓
SQL/Python aggregation

"Which merchants did I spend the most on?"
       ↓
SQL/Python aggregation

"Why did my spending increase?"
       ↓
Structured analysis + RAG + LLM

"Find the transaction related to XYZ."
       ↓
RAG / structured search
```

This distinction is important:

> RAG should not be used for everything.

Structured calculations should use deterministic code/SQL where possible.

---

# 12. General-purpose PDF goal

The product should eventually support more than statements.

Possible documents:

- bank statements
- invoices
- receipts
- contracts
- reports
- insurance documents
- tax documents
- business reports
- policies
- manuals
- research papers

The system should identify the document type and choose an appropriate processing/analysis strategy.

---

# 13. Agent philosophy

The project should be a **real agent**, not just:

```text
PDF → embedding → search → LLM
```

The agent should be able to decide:

```text
Do I need retrieval?
Do I need a calculator?
Do I need structured data?
Do I need SQL?
Do I need multiple searches?
Do I need to verify evidence?
How should I formulate the answer?
```

The agent should use tools where they provide deterministic or external capabilities.

---

# 14. Model strategy

Local development:

```text
Ollama
└── Qwen3 8B
```

No paid API is required for the learning phase.

Later, the architecture should support interchangeable model providers, for example:

```text
Local Ollama
Cloud LLM
Enterprise LLM
GPU-hosted model
```

Do not tightly couple business logic to a specific model provider.

---

# 15. MCP decision

MCP is **not currently required**.

The initial agent can use direct Python tools.

MCP should be introduced only when there is a clear need to expose/reuse tools across different AI clients or agent systems.

Possible future MCP tools:

```text
search_documents
query_transactions
calculate
get_document_metadata
```

But introducing MCP too early would add complexity while the core agent concepts are still being learned.

---

# 16. LangChain / LangGraph decision

Do not introduce them immediately.

The current goal is to understand the underlying concepts first:

```text
LLM
Tool
Tool calling
Agent loop
Streaming
RAG
Retrieval
State
```

After these concepts work manually, a framework can be evaluated.

LangGraph may eventually be useful for:

- multi-step workflows
- stateful agents
- retries
- branching
- human approval
- complex orchestration

But it should be introduced when complexity justifies it.

---

# 17. Production scalability goal

The local prototype is intentionally simple.

A future production architecture may look like:

```text
                    Load Balancer
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           FastAPI    FastAPI    FastAPI
              │          │          │
              └──────────┼──────────┘
                         ▼
                 Agent Service
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         LLM API      Vector DB    PostgreSQL
             │           │           │
             ▼           ▼           ▼
          Claude/       RAG       Structured
          other LLM               data
```

PDF processing can eventually become asynchronous:

```text
Upload
 ↓
Object Storage
 ↓
Queue
 ↓
Document Worker
 ↓
Extraction
 ↓
Embedding
 ↓
Vector DB
```

This allows large documents and many simultaneous users without blocking the API server.

---

# 18. Cost philosophy

Development target:

```text
₹0 / $0
```

Use:

- local Python
- local Ollama
- local Qwen
- local vector database during development
- local storage

Production cost will depend heavily on:

- number of users
- PDF volume
- document size
- embedding model
- LLM provider
- GPU/CPU infrastructure
- vector database
- object storage
- database
- observability

Do not optimize production cloud cost yet. First make the product technically correct.

---

# 19. Frontend philosophy

The current frontend is intentionally simple.

It should eventually support:

```text
Chat
Upload PDF
Document list
Document status
Source/page citations
Agent status
Streaming answer
Tables
Charts where useful
Download/export
Conversation history
```

The application should remain primarily an **AI analyst/document agent**, not just a dashboard.

---

# 20. Development roadmap

## Phase 1 — Agent fundamentals

Completed / nearly completed:

- Python environment
- Ollama
- Qwen
- LLM calls
- calculator tool
- tool calling
- agent loop
- timing
- streaming
- FastAPI
- basic frontend

## Phase 2 — Robust agent

Next:

- proper streaming tool-calling
- strict tool schema validation
- error handling
- conversation state
- request IDs
- structured event protocol
- logging
- tests

## Phase 3 — Documents

Then:

- PDF upload
- PDF extraction
- page metadata
- document storage
- chunking
- embeddings
- vector database
- retrieval

## Phase 4 — RAG

Then:

- semantic search
- metadata filtering
- source citations
- answer grounding
- retrieval evaluation
- hallucination checks

## Phase 5 — Statement analysis

Then:

- transaction extraction
- normalization
- structured database
- SQL/Python analysis
- categories
- summaries
- comparisons
- anomaly detection

## Phase 6 — General PDF agent

Then:

- document type detection
- multiple document types
- document-specific tools
- cross-document questions
- multi-document RAG

## Phase 7 — Production

Finally:

- authentication
- authorization
- object storage
- background workers
- PostgreSQL
- production vector DB
- monitoring
- rate limiting
- security
- deployment
- evaluation
- cost optimization

---

# 21. Claude Cowork instructions

Use the following as the persistent project context/instructions for Claude Cowork.

## Role

You are helping develop the **AI Document Agent** project described in this document.

Act as a senior AI/backend engineer and patient mentor.

The developer knows Python and Java but is learning AI agents, RAG, and LLM application architecture.

## Primary objective

Help build the project incrementally while explaining important architectural decisions.

Do not jump directly to large frameworks or complicated abstractions.

Prefer understanding first, framework second.

## Development rules

1. Work incrementally.
2. Keep the application runnable after each meaningful change.
3. Do not rewrite the entire project unnecessarily.
4. Before changing architecture, explain why.
5. Prefer small, testable functions.
6. Keep API, agent, tools, RAG, and document-processing responsibilities separated.
7. Never hard-code user questions into production application code.
8. Do not expose private chain-of-thought.
9. Show safe application status events instead:
   - thinking
   - retrieving
   - tool selected
   - tool running
   - tool completed
   - generating
   - completed
10. Treat all LLM-generated tool arguments as untrusted input.
11. Validate tool arguments before execution.
12. Prefer deterministic Python/SQL for calculations instead of asking an LLM to calculate.
13. Use RAG for evidence retrieval, not as a replacement for structured data processing.
14. Do not add MCP unless there is a concrete architectural reason.
15. Do not add LangChain/LangGraph unless the complexity of the application justifies them.
16. Keep the local development setup free whenever possible.
17. Prefer Ollama/local models during learning.
18. Keep model providers abstract enough that cloud models can be introduced later.
19. Explain errors from stack traces rather than hiding them.
20. When changing code, identify exactly which files and sections should change.

## Important current environment

Windows.

Python:

```text
3.12.14
```

Package manager:

```text
uv
```

LLM runtime:

```text
Ollama
```

Current model:

```text
qwen3:8b
```

Backend:

```text
FastAPI
Uvicorn
Pydantic
```

Frontend:

```text
HTML
CSS
Vanilla JavaScript
```

## Current architecture

```text
Browser
   ↓
FastAPI
   ↓
Agent
   ↓
Ollama / Qwen
   ↓
Tools
```

## Current user interface

The browser UI is served by FastAPI at:

```text
http://127.0.0.1:8000/
```

Swagger remains available at:

```text
http://127.0.0.1:8000/docs
```

## Current important limitation

Qwen3 8B runs on CPU on the development laptop and can be slow.

Do not assume that adding more LLM calls is acceptable.

When possible:

```text
deterministic task → Python/SQL
retrieval → vector search
reasoning → LLM
```

Minimize unnecessary LLM calls.

## How to respond to development requests

For each change:

1. Explain the goal.
2. Explain the architecture impact.
3. List files to change.
4. Provide the smallest useful code change.
5. Explain how to run it.
6. Explain how to test it.
7. Wait for confirmation before moving to a large next phase.

Do not make unrelated improvements during a focused task.

## Current next task

The immediate technical priority is:

**Finish and verify a robust streaming tool-calling agent.**

It must:

```text
User
 ↓
FastAPI
 ↓
stream_agent
 ↓
Qwen
 ↓
tool decision
 ↓
tool execution
 ↓
Qwen final response
 ↓
stream tokens/events
 ↓
Browser
```

The UI should show safe status events rather than model chain-of-thought.

After that, move to conversation state, then PDF ingestion and RAG.

---

# 22. Definition of success

The first major success milestone is:

> A user can open the browser, ask a question, the agent can decide whether to use a tool, execute the tool safely, stream status and final answer, and return a measured response without exposing private reasoning.

The next major success milestone is:

> A user can upload a PDF, ask a question about it, and receive a grounded answer with relevant source/page references.

The ultimate goal is:

> A general-purpose AI document analyst that can understand uploaded PDFs, retrieve evidence, use deterministic tools, perform structured analysis, reason over multiple documents, and answer users through a conversational interface.
