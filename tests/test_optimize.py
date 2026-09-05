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


def _scored(target, vector, solved=True):
    """A route carrying a fixed assessment vector, bypassing feature extraction."""
    from reagent.core.models import Assessment

    return Route(
        target=target,
        solved=solved,
        assessments=[
            Assessment(objective=o, score=s, rationale="fixture", inputs={})
            for o, s in vector.items()
        ],
    )


def test_compromise_avoids_the_lopsided_corner_route():
    """Two specialists and one all-rounder, all non-dominated.

    A weighted sum picks whichever corner the weights happen to favour. The
    compromise rule takes the balanced route without being told a preference.
    """
    from reagent.optimize.pareto import compromise_route

    safe_only = _scored("T", {"safety": 1.0, "cost": 0.0, "feasibility": 0.0})
    cheap_only = _scored("T", {"safety": 0.0, "cost": 1.0, "feasibility": 0.0})
    balanced = _scored("T", {"safety": 0.7, "cost": 0.7, "feasibility": 1.0})

    assert compromise_route([safe_only, cheap_only, balanced]) is balanced


def test_compromise_ignores_dominated_routes():
    from reagent.optimize.pareto import compromise_route

    good = _scored("T", {"safety": 0.9, "cost": 0.9})
    worse = _scored("T", {"safety": 0.4, "cost": 0.4})
    assert compromise_route([good, worse]) is good


def test_compromise_handles_a_single_route():
    from reagent.optimize.pareto import compromise_route

    only = _scored("T", {"safety": 0.5, "cost": 0.5})
    assert compromise_route([only]) is only


def test_compromise_handles_an_empty_candidate_set():
    from reagent.optimize.pareto import compromise_route

    assert compromise_route([]) is None


def test_compromise_needs_no_weights_to_disagree_with_a_weighted_pick():
    """The point of the rule: it can differ from the weighted winner."""
    from reagent.optimize.aggregate import rank_routes
    from reagent.optimize.pareto import compromise_route

    safe_only = _scored("T", {"safety": 1.0, "cost": 0.0, "feasibility": 0.0})
    balanced = _scored("T", {"safety": 0.7, "cost": 0.7, "feasibility": 1.0})
    routes = [safe_only, balanced]

    safety_heavy = {"safety": 0.9, "cost": 0.05, "feasibility": 0.05}
    assert rank_routes(routes, weights=safety_heavy)[0] is safe_only
    assert compromise_route(routes) is balanced


def test_feasibility_is_deliberately_floored_despite_its_weight():
    """Feasibility's measured median spread is 0.025, under the 0.10 floor.

    It carries the largest weight (0.30) yet is damped on most targets. That is
    intended: a 0.02 gap between two candidates from the same policy model is
    not evidence one route is better, and stretching it to a full swing would
    let it outvote real differences elsewhere. Pinned so the floor is not
    "fixed" without evidence that small likelihood gaps mean something.
    """
    from reagent.optimize.aggregate import MIN_SPAN, normalized_vectors

    assert MIN_SPAN > 0.025, "must stay above feasibility's measured median spread"

    low, high = normalized_vectors([{"feasibility": 0.100}, {"feasibility": 0.125}])
    assert high["feasibility"] < 0.30, "a typical feasibility gap stays a near-tie"


def test_a_wide_spread_still_uses_its_own_range():
    """The floor must not damp an objective that genuinely separates candidates."""
    from reagent.optimize.aggregate import normalized_vectors

    low, high = normalized_vectors([{"safety": 0.20}, {"safety": 0.80}])
    assert low["safety"] == 0.0
    assert high["safety"] == 1.0
