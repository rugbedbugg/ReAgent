"""Agent-layer tests using the mock LLM client (no key or network)."""

from reagent.agents.evaluators import Evaluator, _parse
from reagent.agents.llm.mock import MockClient
from reagent.agents.orchestrator import Orchestrator, build_team
from reagent.core.models import Molecule, Reaction, Route


def _route() -> Route:
    return Route(
        target="CC(=O)Oc1ccccc1C(=O)O",
        reactions=[
            Reaction(
                product="CC(=O)Oc1ccccc1C(=O)O",
                precursors=["CC(=O)OC(C)=O", "O=C(O)c1ccccc1O"],
                metadata={"policy_probability": 0.73, "library_occurence": 1196},
            )
        ],
        leaves=[
            Molecule(smiles="CC(=O)OC(C)=O", in_stock=True),
            Molecule(smiles="O=C(O)c1ccccc1O", in_stock=True),
        ],
        solved=True,
    )


def test_parse_variants():
    assert _parse('{"score": 0.8, "rationale": "good"}') == (0.8, "good")
    # clamped and tolerant of surrounding prose
    score, _ = _parse('Here is my answer: {"score": 1.5, "rationale": "x"} thanks')
    assert score == 1.0
    assert _parse("not json at all")[0] == 0.0


def test_single_evaluator_with_mock():
    agent = Evaluator("safety", MockClient(score=0.9))
    route = _route()
    from reagent.features.extract import compute_features

    compute_features(route)
    a = agent.evaluate(route)
    assert a.objective == "safety"
    assert a.score == 0.9
    assert a.inputs == route.features["safety"]


def test_orchestrator_full_team():
    orch = Orchestrator(client=MockClient(score=0.6))
    route = orch.assess(_route())
    assert len(route.assessments) == 6
    assert {a.objective for a in route.assessments} == set(build_team(MockClient()).keys())
    assert all(a.score == 0.6 for a in route.assessments)
    # features were auto-computed by the orchestrator
    assert route.features
