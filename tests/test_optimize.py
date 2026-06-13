"""Aggregation-layer tests: weighted-sum ranking and Pareto front."""

from reagent.core.models import Assessment, Route
from reagent.optimize.aggregate import rank_routes, weighted_score
from reagent.optimize.pareto import pareto_front


def _route(target: str, scores: dict[str, float]) -> Route:
    return Route(
        target=target,
        assessments=[Assessment(objective=o, score=s, rationale="") for o, s in scores.items()],
    )


# Mirrors the two aspirin routes: acetic anhydride is safe and high-confidence;
# acetyl chloride is marginally greener but hazardous and low-confidence.
ANHYDRIDE = _route(
    "anhydride",
    {"feasibility": 0.90, "availability": 1.0, "cost": 0.8, "safety": 1.0,
     "sustainability": 0.70, "efficiency": 0.9},
)
CHLORIDE = _route(
    "chloride",
    {"feasibility": 0.20, "availability": 1.0, "cost": 0.85, "safety": 0.4,
     "sustainability": 0.80, "efficiency": 0.9},
)


def test_weighted_score_favours_safe_feasible_route():
    assert weighted_score(ANHYDRIDE) > weighted_score(CHLORIDE)


def test_rank_orders_and_records_scores():
    ranked = rank_routes([CHLORIDE, ANHYDRIDE])
    assert ranked[0] is ANHYDRIDE
    assert "weighted" in ANHYDRIDE.scores
    assert ANHYDRIDE.scores["vector"]["safety"] == 1.0


def test_pareto_keeps_both_when_they_trade_off():
    # Neither dominates: anhydride wins safety/feasibility, chloride wins
    # sustainability/cost. Both are non-dominated.
    front = pareto_front([ANHYDRIDE, CHLORIDE])
    assert len(front) == 2


def test_pareto_drops_dominated_route():
    strictly_worse = _route(
        "worse",
        {"feasibility": 0.10, "availability": 0.5, "cost": 0.5, "safety": 0.3,
         "sustainability": 0.5, "efficiency": 0.5},
    )
    front = pareto_front([ANHYDRIDE, strictly_worse])
    assert front == [ANHYDRIDE]
