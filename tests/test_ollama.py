"""OllamaClient test with the HTTP layer mocked (no server needed)."""

from unittest.mock import patch

from reagent.agents.llm.ollama_client import OllamaClient


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


def test_ollama_complete_posts_and_parses():
    client = OllamaClient(model="qwen2.5:3b-instruct", host="http://localhost:11434")
    with patch("reagent.agents.llm.ollama_client.requests.post") as post:
        post.return_value = _FakeResponse('{"score": 0.7, "rationale": "ok"}')
        out = client.complete("system text", "user text", max_tokens=256)

    assert out == '{"score": 0.7, "rationale": "ok"}'
    args, kwargs = post.call_args
    assert args[0] == "http://localhost:11434/api/chat"
    body = kwargs["json"]
    assert body["model"] == "qwen2.5:3b-instruct"
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"] == "user text"


def test_ollama_host_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://box:9999/")
    client = OllamaClient()
    assert client.host == "http://box:9999"
