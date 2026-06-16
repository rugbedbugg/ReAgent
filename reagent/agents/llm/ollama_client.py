"""Ollama-backed LLM client for local, offline agent inference.

Talks to a local Ollama server over its HTTP chat API, so the agent layer runs
with no API key and no per-call cost. Small instruct models (1B-3B) are enough:
each agent only reads a fact block and returns a short JSON verdict.
"""

from __future__ import annotations

import os

import requests

# Constrains decoding to a valid assessment, so the model cannot drift off the
# {score, rationale} contract and score is always a bounded number.
ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str | None = None,
        timeout: int = 180,
        structured: bool = True,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.structured = structured

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        body = {
            "model": self.model,
            "stream": False,
            # Greedy decoding: these are judgments over fixed facts, so we want
            # the single most consistent answer, not sampled variety.
            "options": {"temperature": 0.0, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.structured:
            body["format"] = ASSESSMENT_SCHEMA
        response = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]
