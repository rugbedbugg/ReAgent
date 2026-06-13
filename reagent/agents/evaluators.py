"""Specialist evaluator agents.

An evaluator owns one objective. It reads that objective's fact block from
``route.features`` (produced by the deterministic features layer), asks its LLM
client to interpret those facts, and returns an :class:`Assessment`. The LLM
only interprets; the facts are the sole ground truth.
"""

from __future__ import annotations

import json
import re

from reagent.agents.llm.base import LLMClient
from reagent.agents.prompts import system_prompt, user_prompt
from reagent.core.models import Assessment, Route

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> tuple[float, str]:
    """Pull a {score, rationale} object out of the model's reply."""
    match = _JSON_RE.search(text)
    if not match:
        return 0.0, f"Unparseable response: {text[:200]}"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0, f"Malformed JSON: {text[:200]}"
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))  # clamp to the contract
    return score, str(data.get("rationale", "")).strip()


class Evaluator:
    """Judges a route along a single objective."""

    def __init__(self, objective: str, client: LLMClient):
        self.objective = objective
        self.client = client

    def evaluate(self, route: Route) -> Assessment:
        facts = route.features.get(self.objective, {})
        reply = self.client.complete(
            system=system_prompt(self.objective),
            user=user_prompt(self.objective, facts, route.target, route.num_steps),
        )
        score, rationale = _parse(reply)
        return Assessment(
            objective=self.objective,
            score=score,
            rationale=rationale,
            inputs=facts,
        )
