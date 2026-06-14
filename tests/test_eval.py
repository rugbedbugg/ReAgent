"""Evaluation-layer tests: deterministic scorer and harness (offline, no backend)."""

from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.harness import evaluate
from reagent.features.scoring import deterministic_scores


def _route(target: str, precursors: list[str], hazardous: bool, prob: float, solved: bool = True) -> Route:
    return Route(
        target=target,
        reactions=[
            Reaction(
                product=target,
                precursors=precursors,
                metadata={"policy_probability": prob, "library_occurence": 100},
            )
        ],
        leaves=[Molecule(smiles=p, in_stock=True) for p in precursors],
        solved=solved,
    )


def test_deterministic_scores_penalize_hazard():
    clean = _route("CCN", ["CCO", "N"], hazardous=False, prob=0.8)
    hazardous = _route("CCN", ["CC(=O)Cl", "N"], hazardous=True, prob=0.8)
    assert deterministic_scores(clean)["safety"] == 1.0
    assert deterministic_scores(hazardous)["safety"] <= 0.5
    assert deterministic_scores(clean)["feasibility"] == 0.8


def test_harness_prefers_safer_route_at_equal_solve_rate():
    # Two candidate routes for one target: baseline (feasibility) would pick the
    # slightly higher-probability but hazardous route; REAGENT should pick safer.
    safe = _route("T", ["CCO", "OCC"], hazardous=False, prob=0.70)
    hazardous = _route("T", ["CC(=O)Cl"], hazardous=True, prob=0.72)

    result = evaluate([("t", "T")], planner=lambda s: [safe, hazardous])

    assert result["solve_rate"] == 1.0
    # REAGENT selects the safer route -> higher mean safety than baseline
    assert result["reagent_quality"]["safety"] > result["baseline_quality"]["safety"]
    assert result["per_target"][0]["changed_pick"] is True


def test_harness_counts_unsolved():
    unsolved = _route("U", ["Cc1ccccc1Br"], hazardous=False, prob=0.5, solved=False)
    result = evaluate([("u", "U")], planner=lambda s: [unsolved])
    assert result["solve_rate"] == 0.0
    assert result["per_target"][0]["solved"] is False
