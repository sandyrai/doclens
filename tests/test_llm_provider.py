"""Tests for llm_provider.py — no real LLM or network needed.

HOW THESE TESTS WORK:

  llm_provider reads its configuration from environment
  variables when it is imported. So each test:
    1. sets the env vars it wants (monkeypatch.setenv)
    2. reloads the module so it re-reads them
    3. replaces the "call one model once" function with a
       fake, so we can script failures (429, 404, 401 ...)
       and check the retry / cooldown / fallback behaviour.

  Run:  uv run pytest tests/test_llm_provider.py -v
"""

import importlib
from types import SimpleNamespace

import pytest


ENV_VARS = [
    "LLM_PROVIDER", "LLM_MODELS", "LLM_MODEL", "LLM_API_KEY",
    "LLM_BASE_URL", "LLM_MAX_TOKENS", "LLM_SYSTEM_PREFIX",
    "LLM_MAX_RETRIES", "LLM_RETRY_DELAY", "LLM_COOLDOWN_SECONDS",
    "OLLAMA_MODEL", "OLLAMA_NUM_CTX", "OLLAMA_NUM_THREAD",
    "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
]


@pytest.fixture
def load(monkeypatch):
    """Return a function that reloads llm_provider with given env."""

    # Don't let the developer's real .env leak into tests
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    def _load(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        import ai_document_agent.llm_provider as mod
        mod = importlib.reload(mod)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        return mod

    return _load


class FakeHTTPError(Exception):
    """Looks like the errors the openai / ollama SDKs raise."""

    def __init__(self, code, body="{'user_id': 'user_SECRET'}"):
        super().__init__(f"Error code: {code} - {body}")
        self.status_code = code


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------


def test_default_is_local_ollama(load):
    mod = load()
    assert mod.LLM_PROVIDER == "ollama"
    assert mod.LLM_MODELS == ["qwen3:8b"]
    mod.check_config()  # no key needed locally
    assert mod.get_system_prefix() == "/no_think\n"


def test_legacy_ollama_model_var_still_works(load):
    mod = load(OLLAMA_MODEL="llama3.2:3b")
    assert mod.LLM_MODELS == ["llama3.2:3b"]
    assert mod.get_system_prefix() == ""  # not a qwen3 model


def test_preset_uses_provider_specific_key(load):
    mod = load(LLM_PROVIDER="gemini", LLM_MODELS="m1, m2", GEMINI_API_KEY="k")
    assert "generativelanguage.googleapis.com" in mod.LLM_BASE_URL
    assert mod.LLM_MODELS == ["m1", "m2"]
    assert mod.describe_config()["api_key"] == "set via GEMINI_API_KEY"
    mod.check_config()


def test_generic_key_wins_and_is_never_described(load):
    mod = load(LLM_PROVIDER="openrouter", LLM_MODELS="m",
               LLM_API_KEY="generic-secret", OPENROUTER_API_KEY="other")
    desc = str(mod.describe_config())
    assert "LLM_API_KEY" in desc
    assert "generic-secret" not in desc


def test_claude_alias_and_default_max_tokens(load):
    mod = load(LLM_PROVIDER="claude", LLM_MODELS="m", ANTHROPIC_API_KEY="k")
    assert mod.LLM_PROVIDER == "anthropic"
    assert mod.LLM_MAX_TOKENS == 4096


@pytest.mark.parametrize("env, expected", [
    ({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, "LLM_MODELS"),
    ({"LLM_PROVIDER": "gemini", "LLM_MODELS": "m"}, "GEMINI_API_KEY"),
    ({"LLM_PROVIDER": "custom", "LLM_MODELS": "m"}, "LLM_BASE_URL"),
    ({"LLM_PROVIDER": "nope", "LLM_MODELS": "m"}, "Unknown LLM_PROVIDER"),
])
def test_bad_config_gives_actionable_error(load, env, expected):
    mod = load(**env)
    with pytest.raises(mod.LLMProviderError, match=expected):
        mod.check_config()


def test_local_openai_compatible_needs_no_key(load):
    mod = load(LLM_PROVIDER="lmstudio", LLM_MODELS="local-model")
    mod.check_config()
    mod = load(LLM_PROVIDER="custom", LLM_MODELS="m",
               LLM_BASE_URL="http://localhost:8000/v1")
    mod.check_config()


# ----------------------------------------------------------
# Retry / cooldown / fallback
# ----------------------------------------------------------


def _cloud(load, models="a,b,c"):
    return load(LLM_PROVIDER="openrouter", LLM_MODELS=models,
                OPENROUTER_API_KEY="k")


def test_fallback_chain_429_then_404_then_success(load, monkeypatch):
    mod = _cloud(load)
    calls = []

    def fake(model, messages, tools):
        calls.append(model)
        if model == "a":
            raise FakeHTTPError(429)
        if model == "b":
            raise FakeHTTPError(404)
        return mod.LLMResponse(content="hi from c")

    monkeypatch.setattr(mod, "_openai_chat_once", fake)
    resp = mod.llm_chat([{"role": "user", "content": "x"}])

    assert resp.content == "hi from c"
    assert calls == ["a", "a", "b", "c"]  # 429 retried once, 404 not
    assert "a" in mod._model_cooldown     # rate-limited model cooled
    assert "b" not in mod._model_cooldown

    # Next request skips 'a' entirely
    calls.clear()
    mod.llm_chat([{"role": "user", "content": "x"}])
    assert calls == ["b", "c"]


def test_all_failed_message_is_safe(load, monkeypatch):
    mod = _cloud(load, "a,b")
    monkeypatch.setattr(
        mod, "_openai_chat_once",
        lambda *a: (_ for _ in ()).throw(FakeHTTPError(404)),
    )
    with pytest.raises(mod.LLMProviderError) as err:
        mod.llm_chat([])
    msg = str(err.value)
    assert "tried: a, b" in msg and "HTTP 404" in msg
    assert "user_SECRET" not in msg  # raw provider payload not leaked


def test_bad_key_fails_fast(load, monkeypatch):
    mod = _cloud(load)
    calls = []

    def fake(model, *a):
        calls.append(model)
        raise FakeHTTPError(401)

    monkeypatch.setattr(mod, "_openai_chat_once", fake)
    with pytest.raises(mod.LLMProviderError, match="OPENROUTER_API_KEY"):
        mod.llm_chat([])
    assert calls == ["a"]  # didn't waste calls on b and c


def test_local_server_down_fails_fast(load, monkeypatch):
    mod = load()  # ollama

    def fake(model, *a):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(mod, "_ollama_chat_once", fake)
    with pytest.raises(mod.LLMProviderError, match="Is it running"):
        mod.llm_chat([])


def test_all_in_cooldown_still_tries(load, monkeypatch):
    mod = _cloud(load, "a")
    mod._set_model_cooldown("a")
    monkeypatch.setattr(mod, "_openai_chat_once",
                        lambda *a: mod.LLMResponse(content="ok"))
    assert mod.llm_chat([]).content == "ok"


# ----------------------------------------------------------
# Streaming
# ----------------------------------------------------------


def test_stream_falls_back_before_first_token(load, monkeypatch):
    mod = _cloud(load, "a,b")

    def fake(model, messages, tools):
        if model == "a":
            raise FakeHTTPError(404)
        yield mod.LLMChunk(content="Hel")
        yield mod.LLMChunk(content="lo")

    monkeypatch.setattr(mod, "_openai_stream_once", fake)
    text = "".join(c.content for c in mod.llm_stream([]))
    assert text == "Hello"


def test_stream_does_not_switch_model_mid_answer(load, monkeypatch):
    mod = _cloud(load, "a,b")
    calls = []

    def fake(model, messages, tools):
        calls.append(model)
        yield mod.LLMChunk(content="partial")
        raise FakeHTTPError(500)

    monkeypatch.setattr(mod, "_openai_stream_once", fake)
    with pytest.raises(mod.LLMProviderError, match="mid-answer"):
        list(mod.llm_stream([]))
    assert calls == ["a"]  # no duplicated answer from model b


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tc(index, id=None, name=None, args=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=args))


def test_stream_assembles_tool_call_pieces(load, monkeypatch):
    mod = _cloud(load, "a")
    chunks = [
        _delta(content="Let me check. "),
        _delta(tool_calls=[_tc(0, id="call_1", name="filter_rows")]),
        _delta(tool_calls=[_tc(0, args='{"column": ')]),
        _delta(tool_calls=[_tc(0, args='"Result"}')]),
        # provider that omits index + id (seen with some APIs)
        _delta(tool_calls=[_tc(None, id="call_2", name="count", args="{}")]),
    ]
    fake_client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: iter(chunks))))
    monkeypatch.setattr(mod, "_get_openai_client", lambda: fake_client)

    out = list(mod.llm_stream([], tools=[{"type": "function"}]))
    assert out[0].content == "Let me check. "
    calls = out[-1].tool_calls
    assert [(c.name, c.arguments, c.id) for c in calls] == [
        ("filter_rows", {"column": "Result"}, "call_1"),
        ("count", {}, "call_2"),
    ]


# ----------------------------------------------------------
# Message formats
# ----------------------------------------------------------


def test_tool_message_format_per_provider(load):
    mod = _cloud(load)
    assert mod.make_tool_msg("t", "id1", "r") == {
        "role": "tool", "tool_call_id": "id1", "content": "r"}
    mod = load(LLM_PROVIDER="ollama")
    assert mod.make_tool_msg("t", "id1", "r")["tool_name"] == "t"
