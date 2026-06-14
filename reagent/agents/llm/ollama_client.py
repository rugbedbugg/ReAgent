"""Ollama-backed LLM client for local, offline agent inference.

Talks to a local Ollama server over its HTTP chat API, so the agent layer runs
with no API key and no per-call cost. Small instruct models (1B-3B) are enough:
each agent only reads a fact block and returns a short JSON verdict.
"""

from __future__ import annotations

import os

import requests


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str | None = None,
        timeout: int = 180,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        response = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]
