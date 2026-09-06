"""AnthropicClient with the SDK mocked, so no key and no network are needed.

The backend has been implemented and shipped since the start and exercised only
through the local Ollama model, which means the request it builds and the
response it parses have never been checked by anything. These cover the parts
that can be wrong without a live call: laziness, the request shape, and block
parsing.

What they deliberately do not prove is that the real API accepts this request.
Only a live call does that, and a test needing a paid key is a test nobody runs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reagent.agents.llm.anthropic_client import AnthropicClient


def _block(kind: str, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(type=kind, text=text)


def _client_returning(*blocks) -> MagicMock:
    fake = MagicMock()
    fake.messages.create.return_value = SimpleNamespace(content=list(blocks))
    return fake


def test_constructing_without_a_key_does_not_raise(monkeypatch):
    """Importing and constructing must work on a machine with no key, or the
    local-model path breaks for everyone who never intended to use Anthropic."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient()
    assert client._client is None


def test_the_missing_key_error_arrives_on_use_and_says_what_to_do(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient()

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        client.complete("system", "user")


def test_an_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert AnthropicClient(api_key="explicit")._api_key == "explicit"
    assert AnthropicClient()._api_key == "from-env"


def test_complete_sends_the_documented_request_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    fake = _client_returning(_block("text", "answer"))
    client = AnthropicClient(model="claude-haiku-4-5-20251001")

    with patch.object(client, "_ensure_client", return_value=fake):
        out = client.complete("system text", "user text", max_tokens=256)

    assert out == "answer"
    kwargs = fake.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == 256
    assert kwargs["system"] == "system text"
    assert kwargs["messages"] == [{"role": "user", "content": "user text"}]


def test_multiple_text_blocks_are_joined_in_order(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    fake = _client_returning(_block("text", '{"score": 0.7,'), _block("text", ' "ok": true}'))
    client = AnthropicClient()

    with patch.object(client, "_ensure_client", return_value=fake):
        assert client.complete("s", "u") == '{"score": 0.7, "ok": true}'


def test_non_text_blocks_are_skipped_rather_than_crashing(monkeypatch):
    """A response can carry thinking or tool_use blocks, which have no ``.text``.
    Reading ``.text`` off every block unconditionally would raise on those."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    fake = _client_returning(
        SimpleNamespace(type="thinking"),
        _block("text", "the score"),
        SimpleNamespace(type="tool_use"),
    )
    client = AnthropicClient()

    with patch.object(client, "_ensure_client", return_value=fake):
        assert client.complete("s", "u") == "the score"


def test_the_client_is_built_once_and_reused(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()
    made = MagicMock()
    made.messages.create.return_value = SimpleNamespace(content=[_block("text", "x")])

    with patch("anthropic.Anthropic", return_value=made) as ctor:
        client.complete("s", "u")
        client.complete("s", "u")

    assert ctor.call_count == 1


def test_it_satisfies_the_interface_the_orchestrator_calls(monkeypatch):
    """The orchestrator only ever calls ``complete(system, user, max_tokens)``.
    If that signature drifts, the Anthropic path breaks and nothing else would
    notice, because every other test runs against the local model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    from reagent.agents.llm.ollama_client import OllamaClient

    import inspect

    anthropic_sig = inspect.signature(AnthropicClient.complete)
    ollama_sig = inspect.signature(OllamaClient.complete)
    assert list(anthropic_sig.parameters) == list(ollama_sig.parameters)
