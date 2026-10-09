FROM python:3.12-slim

# Tesseract powers OCR for scanned PDFs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY static ./static
RUN uv sync --locked --no-dev

# Runtime state lives next to src/ (the app resolves paths from there).
# The app runs as an unprivileged user (uid/gid 10001) that owns only these folders.
# The production volumes are handed to that uid before the first deploy of this image.
RUN groupadd --system --gid 10001 app  && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app  && mkdir -p uploads chroma_db ocr_cache data  && chown -R app:app uploads chroma_db ocr_cache data /home/app
ENV HOME=/home/app
USER app

ENV PATH="/srv/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "ai_document_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
