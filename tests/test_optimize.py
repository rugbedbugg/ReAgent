"""Aggregation-layer tests: weighted-sum ranking and Pareto front."""

from reagent.core.models import Assessment, Route
from reagent.optimize.aggregate import normalized_vectors, rank_routes, weighted_score
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


def test_normalization_drops_objectives_that_cannot_discriminate():
    # Availability is 1.0 for every solved route by construction, so it carries
    # no comparative information and must not consume weight.
    vectors = [{"feasibility": 0.9, "availability": 1.0}, {"feasibility": 0.2, "availability": 1.0}]
    normalized = normalized_vectors(vectors)
    assert [set(v) for v in normalized] == [{"feasibility"}, {"feasibility"}]
    assert normalized[0]["feasibility"] == 1.0
    assert normalized[1]["feasibility"] == 0.0


def test_normalization_is_invariant_to_objective_scale():
    # Halving the range an objective happens to span must not change the
    # ranking: only the weights decide the trade-off.
    wide = [{"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.5}]
    narrow = [{"a": 0.0, "b": 0.0}, {"a": 0.5, "b": 0.25}]
    assert normalized_vectors(wide) == normalized_vectors(narrow)


def test_wide_range_objective_no_longer_overrides_the_weights():
    # The measured aspirin failure, with its real spreads: feasibility runs
    # 0.001-0.726 while safety runs 0.30-0.50. Under the safety-tilted profile
    # the user cares more about safety (0.28) than feasibility (0.20), yet raw
    # weighting still hands the pick to feasibility purely on range.
    safety_tilted = {"feasibility": 0.20, "safety": 0.28}
    feasible = _route("feasible", {"feasibility": 0.726, "safety": 0.30})
    safe = _route("safe", {"feasibility": 0.001, "safety": 0.50})

    assert weighted_score(feasible, safety_tilted) > weighted_score(safe, safety_tilted)
    assert rank_routes([feasible, safe], safety_tilted)[0] is safe


def test_small_spread_is_not_stretched_into_a_decision():
    # A 0.02 feasibility gap is noise; a 0.6 safety gap is not. Normalizing on
    # the realized range alone would make them count the same.
    marginally_likelier = _route("hazardous", {"feasibility": 0.72, "safety": 0.40})
    much_safer = _route("safe", {"feasibility": 0.70, "safety": 1.00})

    assert rank_routes([marginally_likelier, much_safer])[0] is much_safer


def test_rank_falls_back_to_raw_scores_for_a_lone_route():
    only = _route("only", {"feasibility": 0.5, "safety": 0.5})
    ranked = rank_routes([only])
    assert ranked[0].scores["weighted"] == only.scores["weighted_raw"] > 0.0
