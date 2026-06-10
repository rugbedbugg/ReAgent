"""Provider-agnostic LLM interface.

Every agent talks to a model only through :class:`LLMClient`, so a specialist
can run on any provider or model without changing its logic. This is what makes
REAGENT multi-LLM: the orchestrator can hand each agent a different client.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """A single-turn text completion with a system and user message."""

    model: str

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...
