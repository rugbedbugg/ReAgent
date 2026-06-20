"""Hybrid evaluator: deterministic score, LLM rationale.

Measurement showed the agents track the deterministic rubric almost perfectly on
passthrough objectives but drift on the formula ones (safety, cost, efficiency)
because a small model is unreliable at arithmetic over compounding facts. Hybrid
mode takes the score from the deterministic scorer and asks the model only to
explain it, so the number is exact while the rationale stays in natural language.
"""

from __future__ import annotations

from reagent.agents.evaluators import _parse
from reagent.agents.llm.base import LLMClient
from reagent.agents.prompts import RATIONALE_SYSTEM, rationale_user_prompt
from reagent.core.models import Assessment, Route
from reagent.features.scoring import deterministic_scores


class HybridEvaluator:
    """Scores one objective deterministically and has the LLM justify it."""

    def __init__(self, objective: str, client: LLMClient):
        self.objective = objective
        self.client = client

    def evaluate(self, route: Route) -> Assessment:
        facts = route.features.get(self.objective, {})
        score = deterministic_scores(route)[self.objective]
        reply = self.client.complete(
            system=RATIONALE_SYSTEM,
            user=rationale_user_prompt(self.objective, facts, route.target, score),
        )
        _, rationale = _parse(reply)
        if not rationale:
            rationale = f"{self.objective} scores {score:.2f} from the computed facts."
        return Assessment(objective=self.objective, score=score, rationale=rationale, inputs=facts)
