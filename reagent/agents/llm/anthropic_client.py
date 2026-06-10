"""Anthropic-backed LLM client."""

from __future__ import annotations

import os


class AnthropicClient:
    """Wraps the Anthropic SDK behind the :class:`LLMClient` protocol.

    Reads the API key from ``ANTHROPIC_API_KEY``. The client is created lazily so
    importing this module never fails just because no key is present.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Export it or put it in a .env file."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        client = self._ensure_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
