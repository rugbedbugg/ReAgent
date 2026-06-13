"""Orchestrator: build the specialist team and run it over routes.

The team is a mapping of objective to LLM client, so each specialist can run on
a different model. ``DEFAULT_MODELS`` sets a sensible per-agent default; callers
can override any entry or pass fully built clients.
"""

from __future__ import annotations

from reagent.agents.evaluators import Evaluator
from reagent.agents.llm.base import LLMClient
from reagent.core.models import Route
from reagent.features.extract import OBJECTIVES, compute_features

# Feasibility is the highest-stakes judgment, so it gets the stronger model by
# default; the rest run on a cheaper model. Override per project or per key.
DEFAULT_MODELS: dict[str, str] = {
    "feasibility": "claude-sonnet-5",
    "availability": "claude-haiku-4-5-20251001",
    "cost": "claude-haiku-4-5-20251001",
    "safety": "claude-sonnet-5",
    "sustainability": "claude-haiku-4-5-20251001",
    "efficiency": "claude-haiku-4-5-20251001",
}


def build_team(client: LLMClient | None = None, hybrid: bool = False) -> dict:
    """Build one evaluator per objective.

    Pass a single ``client`` to run every agent on it (e.g. a MockClient in
    tests). With no client, each agent gets an Anthropic client on its default
    model. With ``hybrid``, objectives are scored deterministically and the LLM
    only writes the rationale.
    """
    from reagent.agents.hybrid import HybridEvaluator

    team: dict = {}
    for objective in OBJECTIVES:
        if client is not None:
            agent_client = client
        else:
            from reagent.agents.llm.anthropic_client import AnthropicClient

            agent_client = AnthropicClient(model=DEFAULT_MODELS[objective])
        team[objective] = (
            HybridEvaluator(objective, agent_client)
            if hybrid
            else Evaluator(objective, agent_client)
        )
    return team


class Orchestrator:
    def __init__(
        self,
        team: dict | None = None,
        client: LLMClient | None = None,
        retriever=None,
        hybrid: bool = False,
    ):
        self.team = team or build_team(client, hybrid=hybrid)
        self.retriever = retriever

    def assess(self, route: Route) -> Route:
        """Attach one Assessment per objective to the route.

        With a retriever configured, each disconnection is first grounded in
        retrieved precedent, which enriches the feasibility facts and becomes the
        cited evidence on the feasibility assessment.
        """
        if not route.features:
            compute_features(route)
        if self.retriever is not None:
            self.retriever.ground_route(route)
        route.assessments = [agent.evaluate(route) for agent in self.team.values()]
        if self.retriever is not None:
            self._attach_evidence(route)
        return route

    @staticmethod
    def _attach_evidence(route: Route) -> None:
        precedents = [p for rxn in route.reactions for p in rxn.metadata.get("precedents", [])]
        evidence = [
            f"template {p['template_hash'][:10]} (occurrence {p['library_occurence']}, "
            f"similarity {p['similarity']})"
            for p in precedents
        ]
        for assessment in route.assessments:
            if assessment.objective == "feasibility":
                assessment.evidence = evidence
