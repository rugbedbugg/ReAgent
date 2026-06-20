"""Hybrid-evaluator tests: score is deterministic, rationale comes from the LLM."""

import json

from reagent.agents.hybrid import HybridEvaluator
from reagent.agents.orchestrator import Orchestrator
from reagent.core.models import Molecule, Reaction, Route
from reagent.features.extract import compute_features
from reagent.features.scoring import deterministic_scores


class _WrongScoreClient:
    """Returns a deliberately wrong score to prove hybrid ignores it."""

    model = "wrong"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return json.dumps({"score": 0.123, "rationale": "explanation text"})


def _hazardous_route() -> Route:
    return Route(
        target="CC(=O)Oc1ccccc1C(=O)O",
        reactions=[Reaction(product="CC(=O)Oc1ccccc1C(=O)O", precursors=["CC(=O)Cl"],
                            metadata={"policy_probability": 0.7, "library_occurence": 100})],
        leaves=[Molecule(smiles="CC(=O)Cl", in_stock=True)],
        solved=True,
    )


def test_hybrid_uses_deterministic_score_not_llm():
    route = compute_features(_hazardous_route())
    agent = HybridEvaluator("safety", _WrongScoreClient())
    a = agent.evaluate(route)
    # hazard present -> penalized deterministic score, NOT the LLM's 0.123
    assert a.score == deterministic_scores(route)["safety"]
    assert a.score != 0.123
    assert a.score < 1.0
    assert a.rationale == "explanation text"


def test_hybrid_orchestrator_matches_reference_exactly():
    route = _hazardous_route()
    orch = Orchestrator(client=_WrongScoreClient(), hybrid=True)
    orch.assess(route)
    ref = deterministic_scores(route)
    for a in route.assessments:
        assert a.score == ref[a.objective]  # zero error by construction
