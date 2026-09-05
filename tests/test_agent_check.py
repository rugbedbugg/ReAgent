"""Agent-check harness test using a scripted mock client (no LLM/network)."""

import json

from reagent.agents.orchestrator import Orchestrator
from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.agent_check import check_agents


class _EchoDeterministicClient:
    """Mock that returns whatever score the user prompt's facts imply, so agent
    and deterministic scores agree perfectly (MAE should be ~0)."""

    model = "echo"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        # The safety objective is easy to make exact: no hazards -> 1.0 else 0.5.
        score = 0.5 if "acid_halide" in user or "hazard_count=1" in user else 1.0
        return json.dumps({"score": score, "rationale": "mock"})


def _clean_route() -> Route:
    return Route(
        target="CCN",
        reactions=[Reaction(product="CCN", precursors=["CCO"],
                            metadata={"policy_probability": 0.8, "library_occurence": 100})],
        leaves=[Molecule(smiles="CCO", in_stock=True)],
        solved=True,
    )


def test_check_agents_reports_metrics():
    orch = Orchestrator(client=_EchoDeterministicClient())
    result = check_agents(
        [("clean", "CCN")],
        planner=lambda s: [_clean_route()],
        orchestrator=orch,
        routes_per=1,
    )
    assert result["assessments"] == 7  # one per objective
    assert result["parse_failure_rate"] == 0.0
    assert "safety" in result["objective_mae"]
    # safety agent returned 1.0 and the clean route's deterministic safety is 1.0
    assert result["objective_mae"]["safety"] == 0.0
    assert result["ranking_agreement"] is None  # only one route, nothing to rank
