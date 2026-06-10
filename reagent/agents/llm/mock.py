"""Deterministic mock client so the agent layer runs without a key or network."""

from __future__ import annotations

import json


class MockClient:
    """Returns a fixed, well-formed JSON assessment for any prompt."""

    model = "mock"

    def __init__(self, score: float = 0.5):
        self._score = score

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return json.dumps(
            {"score": self._score, "rationale": "Mock assessment from deterministic client."}
        )
