"""LLM Provider — one interface, any LLM (local or cloud, free or paid).

WHY THIS EXISTS:

  agent.py needs to "call an LLM", but it should not care
  WHICH LLM. This module is the single place that knows how
  to talk to a model. agent.py only ever calls:

      llm_chat(messages, tools)    -> LLMResponse
      llm_stream(messages, tools)  -> LLMChunk, LLMChunk, ...

  Everything about providers, API keys, base URLs, retries
  and fallbacks lives here. No keys are written in code —
  they are read from environment variables (your .env file,
  which git ignores). See .env.example for every option.

THE KEY IDEA — "OpenAI-compatible" APIs:

  Almost every LLM vendor now exposes the same HTTP API
  shape that OpenAI invented (/v1/chat/completions). So one
  client library (`openai`) can talk to all of them — we just
  change the base_url, api_key and model name:

    Provider      base_url
    ----------    ------------------------------------------------
    openrouter    https://openrouter.ai/api/v1
    gemini        https://generativelanguage.googleapis.com/v1beta/openai/
    anthropic     https://api.anthropic.com/v1/          (Claude)
    openai        https://api.openai.com/v1
    groq          https://api.groq.com/openai/v1
    lmstudio      http://localhost:1234/v1               (local)
    custom        anything you put in LLM_BASE_URL       (vLLM,
                  Together, DeepSeek, Mistral, llama.cpp ...)

  Ollama (local) is special-cased: we use its NATIVE client
  because it lets us tune context size, CPU threads and
  keep-alive, and turn off Qwen3 "thinking" — which matters
  a lot for speed on a laptop CPU.

      agent.py
         | llm_chat() / llm_stream()
      llm_provider.py  (this file)
         |  LLM_PROVIDER=?
      +--+----------------------------+
      |                               |
   ollama (native)        any OpenAI-compatible API
   local, free            (cloud or local, free or paid)

CONFIGURATION (all from .env):

  LLM_PROVIDER   which backend (see table above, or "ollama")
  LLM_MODELS     comma-separated model list, tried IN ORDER.
                 If one fails (removed, rate-limited, ...) the
                 next one is used automatically.
  LLM_API_KEY    the key. Optional if you use the provider's
                 usual variable (OPENROUTER_API_KEY,
                 GEMINI_API_KEY, ANTHROPIC_API_KEY, ...).
  LLM_BASE_URL   override the URL (required for "custom").

RELIABILITY FEATURES (same mechanism as before, now for all
providers):

  1. Retry on 429 (rate limit) — wait, try the same model again.
  2. Cooldown — a model that keeps returning 429 is skipped
     for a few minutes, so later requests don't waste time.
  3. Model fallback — any other failure moves to the next model
     in LLM_MODELS.
  4. Fail fast when retrying can't help — a bad API key (401/403)
     or an unreachable server stops immediately with a clear
     message instead of trying every model.

  Errors raised to the rest of the app are LLMProviderError with
  a short, safe message. The full provider response (which can
  include account IDs) goes only to the server log.

QUICK SELF-TEST:

  uv run python -m ai_document_agent.llm_provider

  prints the active configuration and sends one tiny prompt.
"""

import json
import logging
import os
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar


# ----------------------------------------------------------
# Load .env here as well as in main.py.
#
# main.py already calls load_dotenv() before importing the
# routes, but loading it here too makes this module work on
# its own (tests, the self-test at the bottom, scripts).
# load_dotenv() never overwrites variables that are already
# set, so calling it twice is harmless.
# ----------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional for this module
    pass


logger = logging.getLogger(__name__)

T = TypeVar("T")


# ==========================================================
# Provider presets
# ==========================================================
#
# A preset is just "where is the API" + "which env variable
# usually holds its key". Adding a new OpenAI-compatible
# provider is ONE line here — no other code changes.


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str | None
    key_env_vars: tuple[str, ...] = ()
    needs_key: bool = True
    is_local: bool = False


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        "https://openrouter.ai/api/v1",
        ("OPENROUTER_API_KEY",),
    ),
    "gemini": ProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "anthropic": ProviderPreset(
        "https://api.anthropic.com/v1/",
        ("ANTHROPIC_API_KEY",),
    ),
    "openai": ProviderPreset(
        "https://api.openai.com/v1",
        ("OPENAI_API_KEY",),
    ),
    "groq": ProviderPreset(
        "https://api.groq.com/openai/v1",
        ("GROQ_API_KEY",),
    ),
    "lmstudio": ProviderPreset(
        "http://localhost:1234/v1",
        needs_key=False,
        is_local=True,
    ),
    # Any other OpenAI-compatible server. You must set
    # LLM_BASE_URL; LLM_API_KEY is optional (some local
    # servers don't need one).
    "custom": ProviderPreset(None, needs_key=False),
}

# Friendly alternative names people are likely to type.
PROVIDER_ALIASES = {
    "claude": "anthropic",
    "google": "gemini",
    "lm-studio": "lmstudio",
    "openai-compatible": "custom",
}

OLLAMA = "ollama"


# ==========================================================
# Configuration — read once from the environment
# ==========================================================


def _env_list(name: str) -> list[str]:
    """Read a comma-separated env var into a clean list."""

    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid number in %s — using %s", name, default)
        return default


_raw_provider = (os.getenv("LLM_PROVIDER") or OLLAMA).strip().lower()
LLM_PROVIDER = PROVIDER_ALIASES.get(_raw_provider, _raw_provider)
IS_OLLAMA = LLM_PROVIDER == OLLAMA

_PRESET = PROVIDER_PRESETS.get(LLM_PROVIDER)

# Models to try, in order. For Ollama we also honour the
# older OLLAMA_MODEL variable and default to qwen3:8b so an
# existing local setup keeps working with no .env changes.
if IS_OLLAMA:
    LLM_MODELS = (
        _env_list("LLM_MODELS")
        or _env_list("LLM_MODEL")
        or _env_list("OLLAMA_MODEL")
        or ["qwen3:8b"]
    )
else:
    LLM_MODELS = _env_list("LLM_MODELS") or _env_list("LLM_MODEL")

LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL", "").strip()
    or (_PRESET.base_url if _PRESET else None)
)

# Retry / fallback tuning (defaults = previous behaviour)
LLM_MAX_RETRIES = int(_env_float("LLM_MAX_RETRIES", 1))   # extra tries on 429
LLM_RETRY_DELAY = _env_float("LLM_RETRY_DELAY", 2.0)      # seconds
COOLDOWN_SECONDS = _env_float("LLM_COOLDOWN_SECONDS", 300)
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 120)              # per request

# Optional output cap. Only sent when set (or for Claude,
# whose API requires one) because some newer OpenAI models
# reject the max_tokens parameter.
_max_tokens_raw = os.getenv("LLM_MAX_TOKENS", "").strip()
LLM_MAX_TOKENS: int | None = int(_max_tokens_raw) if _max_tokens_raw else (
    4096 if LLM_PROVIDER == "anthropic" else None
)

# ---------------------------------------------------------
# Ollama tuning (only used when LLM_PROVIDER=ollama)
# ---------------------------------------------------------
#   num_ctx    context window in tokens. Smaller = faster on
#              CPU. 2048 is enough for most questions here.
#   num_thread CPU threads. Unset = let Ollama decide. Set it
#              to your physical core count if you want.
#   keep_alive keep the model in RAM between requests to
#              avoid a 10-20 s cold start every question.
#   OLLAMA_HOST (read by the ollama library itself) points at
#              a non-default server, e.g. http://192.168.1.5:11434
OLLAMA_OPTIONS: dict[str, Any] = {
    "num_ctx": int(_env_float("OLLAMA_NUM_CTX", 2048)),
}
if os.getenv("OLLAMA_NUM_THREAD"):
    OLLAMA_OPTIONS["num_thread"] = int(_env_float("OLLAMA_NUM_THREAD", 0))

OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


def _resolve_api_key() -> tuple[str, str]:
    """Find the API key. Returns (key, name_of_env_var).

    Order: LLM_API_KEY first (generic), then the provider's
    usual variable. This lets you keep several keys in .env
    and switch provider by changing only LLM_PROVIDER.
    """

    generic = os.getenv("LLM_API_KEY", "").strip()
    if generic:
        return generic, "LLM_API_KEY"

    if _PRESET:
        for var in _PRESET.key_env_vars:
            value = os.getenv(var, "").strip()
            if value:
                return value, var

    return "", ""


def _key_hint() -> str:
    """Which variable the user should set, for error messages."""

    if _PRESET and _PRESET.key_env_vars:
        return f"{_PRESET.key_env_vars[0]} (or LLM_API_KEY)"
    return "LLM_API_KEY"


def _is_local() -> bool:
    return IS_OLLAMA or bool(_PRESET and _PRESET.is_local)


# ==========================================================
# Errors
# ==========================================================


class LLMProviderError(RuntimeError):
    """A clean, user-safe error from the LLM layer.

    The message never contains API keys or raw provider
    payloads — it is safe to show in the browser. Details
    are written to the server log instead.
    """


def check_config() -> None:
    """Raise LLMProviderError if the configuration can't work.

    Called before every LLM request (cheap), so the user gets
    an actionable message instead of a confusing HTTP error.
    """

    if not IS_OLLAMA and _PRESET is None:
        supported = ", ".join([OLLAMA, *PROVIDER_PRESETS])
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Supported: {supported}."
        )

    if not LLM_MODELS:
        raise LLMProviderError(
            f"No model configured for LLM_PROVIDER={LLM_PROVIDER}. "
            f"Set LLM_MODELS in your .env (comma-separated, tried "
            f"in order). See .env.example for examples."
        )

    if not IS_OLLAMA and not LLM_BASE_URL:
        raise LLMProviderError(
            "LLM_PROVIDER=custom needs LLM_BASE_URL in your .env "
            "(e.g. http://localhost:8000/v1)."
        )

    if _PRESET and _PRESET.needs_key and not _resolve_api_key()[0]:
        raise LLMProviderError(
            f"No API key for LLM_PROVIDER={LLM_PROVIDER}. "
            f"Set {_key_hint()} in your .env."
        )


# ==========================================================
# Cooldown cache
# ==========================================================
#
# After a model exhausts its 429 retries we remember it and
# skip it for COOLDOWN_SECONDS, so the NEXT request goes
# straight to a working fallback instead of wasting ~4 s
# being rate-limited again. Lives in memory; a server
# restart clears it (rate limits reset too).

_model_cooldown: dict[str, float] = {}


def _is_model_cooled_down(model: str) -> bool:
    until = _model_cooldown.get(model)
    if until is None:
        return False

    if time.time() >= until:
        del _model_cooldown[model]
        logger.info("Cooldown expired for '%s' — will try again", model)
        return False

    logger.info(
        "Skipping '%s' (cooldown: %.0fs remaining)",
        model, until - time.time(),
    )
    return True


def _set_model_cooldown(model: str) -> None:
    _model_cooldown[model] = time.time() + COOLDOWN_SECONDS
    logger.warning(
        "Model '%s' placed in cooldown for %ds", model, COOLDOWN_SECONDS,
    )


def _models_to_try() -> list[str]:
    """Models not in cooldown, in configured order.

    If EVERY model is cooling down we try them all anyway —
    a slow answer beats an instant failure.
    """

    available = [m for m in LLM_MODELS if not _is_model_cooled_down(m)]
    if not available:
        logger.warning("All models are in cooldown — trying them anyway")
        return list(LLM_MODELS)
    return available


# ==========================================================
# Classifying failures
# ==========================================================
#
# Different SDKs raise different exception classes, but
# most carry an HTTP status_code. We look at that first and
# only fall back to searching the message text.


def _status_code(exc: Exception) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_rate_limit(exc: Exception) -> bool:
    code = _status_code(exc)
    return code == 429 or (code is None and "429" in str(exc))


def _is_auth_error(exc: Exception) -> bool:
    return _status_code(exc) in (401, 403)


def _is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, ConnectionError):
        return True
    if "Connection" in type(exc).__name__:  # openai.APIConnectionError
        return True
    text = str(exc).lower()
    return "failed to connect" in text or "connection refused" in text


def _short_error(exc: Exception | None) -> str:
    """One safe phrase describing an error (no raw payload)."""

    if exc is None:
        return "unknown error"
    code = _status_code(exc)
    if code:
        return f"HTTP {code}"
    return type(exc).__name__


def _handle_failure(model: str, attempt: int, exc: Exception) -> str:
    """Decide what to do after a failed attempt.

    Returns "retry" (same model again) or "next" (next model).
    Raises LLMProviderError when no other model can help.
    """

    if _is_auth_error(exc):
        logger.error("LLM auth failed for provider '%s': %s", LLM_PROVIDER, exc)
        raise LLMProviderError(
            f"{LLM_PROVIDER} rejected the API key "
            f"({_short_error(exc)}). Check {_key_hint()} in your .env."
        ) from exc

    if _is_connection_error(exc):
        logger.error("LLM connection failed (%s): %s", LLM_BASE_URL, exc)
        where = "Ollama" if IS_OLLAMA else LLM_BASE_URL
        hint = " Is it running?" if _is_local() else ""
        raise LLMProviderError(
            f"Could not connect to {where}.{hint}"
        ) from exc

    rate_limited = _is_rate_limit(exc)

    if rate_limited and attempt < LLM_MAX_RETRIES:
        logger.warning(
            "Model '%s' rate-limited (429). Retry %d/%d in %.1fs...",
            model, attempt + 1, LLM_MAX_RETRIES, LLM_RETRY_DELAY,
        )
        return "retry"

    # Full detail goes to the log only.
    logger.warning(
        "Model '%s' failed: %s. Trying next fallback...", model, exc,
    )
    if rate_limited:
        _set_model_cooldown(model)
    return "next"


def _all_models_failed(tried: list[str], last_error: Exception | None):
    return LLMProviderError(
        f"All models failed for LLM provider '{LLM_PROVIDER}' "
        f"(tried: {', '.join(tried)}; last error: "
        f"{_short_error(last_error)}). Check LLM_MODELS in your "
        f".env — free models are often renamed, removed or "
        f"rate-limited."
    )


def _log_success(model: str, attempt: int, streaming: bool) -> None:
    kind = "streaming" if streaming else "response"
    if attempt:
        logger.info(
            "Model '%s' succeeded on retry %d/%d (%s)",
            model, attempt, LLM_MAX_RETRIES, kind,
        )
    else:
        logger.info("LLM %s from %s:%s", kind, LLM_PROVIDER, model)


# ==========================================================
# The fallback engine (shared by every provider)
# ==========================================================
#
# These two functions contain the retry → cooldown →
# next-model loop. They take a small function that knows
# how to call ONE model once, so the loop is written once
# and reused for Ollama and every OpenAI-compatible API.


def _chat_with_fallback(call_once: Callable[[str], T]) -> T:
    last_error: Exception | None = None
    tried: list[str] = []

    for model in _models_to_try():
        tried.append(model)

        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                result = call_once(model)
                _log_success(model, attempt, streaming=False)
                return result
            except LLMProviderError:
                raise
            except Exception as exc:
                last_error = exc
                if _handle_failure(model, attempt, exc) == "retry":
                    time.sleep(LLM_RETRY_DELAY)
                    continue
                break

    raise _all_models_failed(tried, last_error)


def _stream_with_fallback(
    open_stream: Callable[[str], Iterator["LLMChunk"]],
) -> Generator["LLMChunk", None, None]:
    last_error: Exception | None = None
    tried: list[str] = []

    for model in _models_to_try():
        tried.append(model)

        for attempt in range(LLM_MAX_RETRIES + 1):
            # Once we've sent text to the user we must NOT
            # switch models — the answer would restart in the
            # middle and appear duplicated. So we track it.
            started = False
            try:
                for chunk in open_stream(model):
                    if not started:
                        _log_success(model, attempt, streaming=True)
                        started = True
                    yield chunk
                if not started:
                    _log_success(model, attempt, streaming=True)
                return
            except LLMProviderError:
                raise
            except Exception as exc:
                if started:
                    logger.error("Stream from '%s' broke mid-answer: %s", model, exc)
                    raise LLMProviderError(
                        "The model stopped responding mid-answer "
                        f"({_short_error(exc)}). Please ask again."
                    ) from exc
                last_error = exc
                if _handle_failure(model, attempt, exc) == "retry":
                    time.sleep(LLM_RETRY_DELAY)
                    continue
                break

    raise _all_models_failed(tried, last_error)


# ==========================================================
# Unified response types
# ==========================================================
#
# Ollama:  response.message.content
# OpenAI:  response.choices[0].message.content
# agent.py only ever sees these dataclasses instead.


@dataclass
class ToolCall:
    """A single tool call requested by the LLM.

    'id' links a tool result back to the call (OpenAI-style
    APIs need it; Ollama doesn't use it).
    """

    name: str
    arguments: dict
    id: str = ""


@dataclass
class LLMResponse:
    """Unified non-streaming response from any provider."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: Any = None


@dataclass
class LLMChunk:
    """A single streaming chunk from any provider.

    Content chunks arrive one by one. If the model decides
    to call tools, the COMPLETE tool_calls list arrives on
    one final chunk (we assemble the pieces internally).
    """

    content: str = ""
    tool_calls: list[ToolCall] | None = None


# ==========================================================
# System prompt prefix
# ==========================================================


def get_system_prefix() -> str:
    """Text to put at the very start of the system prompt.

    Qwen3 models understand "/no_think", which skips their
    slow internal reasoning (20-30 s saved per answer on CPU).
    Other models would just be confused by it, so by default
    we only add it when the primary model is a Qwen3 model.
    Override with LLM_SYSTEM_PREFIX in .env ("" = nothing).
    """

    custom = os.getenv("LLM_SYSTEM_PREFIX")
    if custom is not None:
        return custom + "\n" if custom and not custom.endswith("\n") else custom

    if LLM_MODELS and "qwen3" in LLM_MODELS[0].lower():
        return "/no_think\n"
    return ""


# ==========================================================
# Message format helpers
# ==========================================================
#
# Tool messages look different in Ollama vs OpenAI format.
# These helpers build the right one for the active provider
# so agent.py never needs if/else on the provider.


def make_tool_msg(tool_name: str, tool_call_id: str, content: str) -> dict:
    """Message that carries a tool's result back to the LLM."""

    if IS_OLLAMA:
        return {"role": "tool", "tool_name": tool_name, "content": content}

    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def make_assistant_msg(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> dict | Any:
    """Assistant message (the LLM's tool-call decision) for history."""

    if IS_OLLAMA:
        from ollama._types import Message as OllamaMessage

        if tool_calls:
            return OllamaMessage(
                role="assistant",
                content=content or "",
                tool_calls=[
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in tool_calls
                ],
            )
        return OllamaMessage(role="assistant", content=content)

    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]
    return msg


# ==========================================================
# Public interface — the ONLY functions agent.py calls
# ==========================================================


def llm_chat(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Send messages and get one complete response (no streaming).

    Used for tool-calling rounds where the user doesn't need
    to see tokens appear one by one.
    """

    check_config()
    call_once = _ollama_chat_once if IS_OLLAMA else _openai_chat_once
    return _chat_with_fallback(lambda model: call_once(model, messages, tools))


def llm_stream(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> Generator[LLMChunk, None, None]:
    """Stream the response token by token.

    Yields LLMChunk objects: content first, and if the model
    calls tools, one final chunk with the complete tool_calls.
    """

    check_config()
    open_stream = _ollama_stream_once if IS_OLLAMA else _openai_stream_once
    yield from _stream_with_fallback(
        lambda model: open_stream(model, messages, tools)
    )


# ==========================================================
# Ollama (native client, local)
# ==========================================================


def _ollama_chat_once(model, messages, tools) -> LLMResponse:
    from ollama import chat

    response = chat(
        model=model,
        messages=messages,
        tools=tools or [],
        think=False,
        options=OLLAMA_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    tool_calls = [
        ToolCall(name=tc.function.name, arguments=tc.function.arguments)
        for tc in (response.message.tool_calls or [])
    ]

    return LLMResponse(
        content=response.message.content or "",
        tool_calls=tool_calls,
        raw_message=response.message,
    )


def _ollama_stream_once(model, messages, tools) -> Iterator[LLMChunk]:
    from ollama import chat

    stream = chat(
        model=model,
        messages=messages,
        tools=tools or [],
        think=False,
        stream=True,
        options=OLLAMA_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    for chunk in stream:
        tool_calls = None
        if chunk.message.tool_calls:
            # Ollama sends complete tool calls in one chunk
            tool_calls = [
                ToolCall(name=tc.function.name, arguments=tc.function.arguments)
                for tc in chunk.message.tool_calls
            ]
        content = chunk.message.content or ""
        if content or tool_calls:
            yield LLMChunk(content=content, tool_calls=tool_calls)


# ==========================================================
# OpenAI-compatible APIs (OpenRouter, Gemini, Claude,
# OpenAI, Groq, LM Studio, custom ...)
# ==========================================================
#
# Differences from Ollama that we smooth over:
#   - tool arguments arrive as a JSON STRING -> we parse it
#   - in streaming, tool calls arrive in PIECES across many
#     chunks -> we accumulate and emit them once at the end
#   - some providers omit tool-call ids -> we create one so
#     the tool result can still be matched to its call

_openai_client = None


def _get_openai_client():
    """Create the OpenAI-compatible client once (lazily)."""

    global _openai_client

    if _openai_client is None:
        from openai import OpenAI

        api_key, _ = _resolve_api_key()

        # max_retries=0: the SDK would otherwise retry 429s
        # itself, multiplying with OUR retry loop (3 x 3 = 9
        # calls to a model that is rate-limited anyway).
        _openai_client = OpenAI(
            base_url=LLM_BASE_URL,
            # Local servers ignore the key, but the SDK
            # requires a non-empty string.
            api_key=api_key or "not-needed",
            max_retries=0,
            timeout=LLM_TIMEOUT,
        )
        logger.info(
            "OpenAI-compatible client ready: provider=%s url=%s",
            LLM_PROVIDER, LLM_BASE_URL,
        )

    return _openai_client


def _request_kwargs(model, messages, tools, stream=False) -> dict:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if stream:
        kwargs["stream"] = True
    # Only send tools when we have some — tools=[] makes
    # some models think they MUST call a tool.
    if tools:
        kwargs["tools"] = tools
    if LLM_MAX_TOKENS:
        kwargs["max_tokens"] = LLM_MAX_TOKENS
    return kwargs


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _openai_chat_once(model, messages, tools) -> LLMResponse:
    client = _get_openai_client()
    response = client.chat.completions.create(
        **_request_kwargs(model, messages, tools),
    )

    if not response.choices:
        raise RuntimeError("Provider returned no choices")

    message = response.choices[0].message
    tool_calls = [
        ToolCall(
            name=tc.function.name,
            arguments=_parse_args(tc.function.arguments),
            id=tc.id or f"call_{i}",
        )
        for i, tc in enumerate(message.tool_calls or [])
    ]

    return LLMResponse(
        content=message.content or "",
        tool_calls=tool_calls,
        raw_message=message,
    )


def _openai_stream_once(model, messages, tools) -> Iterator[LLMChunk]:
    client = _get_openai_client()
    stream = client.chat.completions.create(
        **_request_kwargs(model, messages, tools, stream=True),
    )

    # index -> {"id", "name", "arguments"(string so far)}
    pieces: dict[int, dict[str, str]] = {}

    for chunk in stream:
        if not chunk.choices:  # e.g. usage-only final chunk
            continue

        delta = chunk.choices[0].delta
        if delta is None:
            continue

        for tc in delta.tool_calls or []:
            idx = tc.index
            if idx is None:
                # Some providers omit the index: a new id means
                # a new call, otherwise continue the last one.
                if tc.id and all(p["id"] != tc.id for p in pieces.values()):
                    idx = len(pieces)
                else:
                    idx = max(pieces, default=0)

            acc = pieces.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                acc["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    acc["name"] = tc.function.name
                if tc.function.arguments:
                    acc["arguments"] += tc.function.arguments

        if delta.content:
            yield LLMChunk(content=delta.content)

    if pieces:
        yield LLMChunk(tool_calls=[
            ToolCall(
                name=p["name"],
                arguments=_parse_args(p["arguments"]),
                id=p["id"] or f"call_{idx}",
            )
            for idx, p in sorted(pieces.items())
        ])


# ==========================================================
# Startup logging (never logs the key itself)
# ==========================================================


def describe_config() -> dict[str, Any]:
    """Safe summary of the active configuration."""

    _, key_source = _resolve_api_key()
    return {
        "provider": LLM_PROVIDER,
        "base_url": "ollama (native)" if IS_OLLAMA else LLM_BASE_URL,
        "models": LLM_MODELS,
        "api_key": f"set via {key_source}" if key_source else "not set",
    }


logger.info("LLM config: %s", describe_config())


if __name__ == "__main__":
    # Self-test: uv run python -m ai_document_agent.llm_provider
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Config:", json.dumps(describe_config(), indent=2))
    reply = llm_chat([{"role": "user", "content": "Reply with just the word: OK"}])
    print("Reply:", reply.content.strip())
