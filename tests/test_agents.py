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
    assert len(route.assessments) == 7
    assert {a.objective for a in route.assessments} == set(build_team(MockClient()).keys())
    assert all(a.score == 0.6 for a in route.assessments)
    # features were auto-computed by the orchestrator
    assert route.features


def test_rationale_does_not_credit_an_objective_that_decided_nothing():
    # Availability is 1.0 for every solved route, so it is the highest raw score
    # but cannot have carried the winner. Feasibility is what separated them.
    from reagent.agents.rationale import build_rationale
    from reagent.core.models import Assessment, Route
    from reagent.optimize.aggregate import rank_routes

    def _r(scores):
        r = Route(target="T", solved=True)
        r.assessments = [Assessment(objective=o, score=s, rationale="") for o, s in scores.items()]
        return r

    win = _r({"feasibility": 0.72, "availability": 1.0, "safety": 0.40})
    lose = _r({"feasibility": 0.05, "availability": 1.0, "safety": 0.35})
    ranked = rank_routes([win, lose])
    text = build_rationale(ranked, {id(win): 1, id(lose): 2})

    carried = next(line for line in text.splitlines() if line.startswith("Carried by:"))
    assert "feasibility" in carried
    assert "availability" not in carried
