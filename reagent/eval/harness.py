"""Evaluation harness: solve-rate and baseline-vs-REAGENT route quality.

Both strategies choose from the *same* candidate routes, so solve-rate is
identical by construction; what differs is which route each picks:

- baseline: highest feasibility only (what plain likelihood-driven search prefers)
- REAGENT: highest weighted multi-objective score

We then report the quality (safety, sustainability, cost) of the routes each
strategy selected. The thesis holds if REAGENT matches solve-rate while
selecting routes with better multi-objective quality.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from reagent.core.models import Route
from reagent.features.scoring import deterministic_scores
from reagent.optimize.aggregate import (
    DEFAULT_WEIGHTS,
    WEIGHT_PROFILES,
    normalized_vectors,
    route_signature,
    weighted_from_vector,
)
from reagent.optimize.pareto import compromise_route

# Re-exported: evaluate() and check-adaptive read the profiles from here, and
# they now live beside DEFAULT_WEIGHTS so `plan` can reach them too.
__all__ = ["WEIGHT_PROFILES", "evaluate", "largest_leaf_fraction", "ADVANCED_LEAF"]


def _select(routes: list[Route], weights: dict[str, float]) -> tuple[Route, Route]:
    """Return (baseline_pick, reagent_pick) over solved routes.

    ReAgent's pick uses the same candidate-normalized aggregation the CLI ranks
    with, so the comparison measures the real selection strategy.
    """
    vectors = [deterministic_scores(r) for r in routes]
    normalized = normalized_vectors(vectors)

    # Ties are common here and the search does not return routes in a stable
    # order, so ``max`` alone would pick by arrival position. The signature
    # makes the choice a property of the routes instead.
    def best(score) -> int:
        return min(range(len(routes)), key=lambda i: (-score(i), route_signature(routes[i])))

    baseline = best(lambda i: vectors[i]["feasibility"])
    reagent = best(lambda i: weighted_from_vector(normalized[i], weights))
    return routes[baseline], routes[reagent]


ADVANCED_LEAF = 0.8


def largest_leaf_fraction(route: Route) -> float:
    """Heavy atoms in the route's biggest leaf, over heavy atoms in the target.

    Reported alongside solve-rate because solve-rate cannot see the difference:
    it counts a target as solved when every leaf is purchasable, and says
    nothing about how much of the molecule was bought rather than made. This is
    the same number the ``construction`` objective scores; it is surfaced here
    so the degenerate case is visible in the evaluation output whether or not
    anyone weights that objective.
    """
    from reagent.features.extract import compute_features

    if not route.features:
        compute_features(route)
    return route.features["construction"]["largest_leaf_fraction"]


def evaluate(
    targets: list[tuple[str, str]],
    planner: Callable[[str], list[Route]],
    weights: dict[str, float] | None = None,
) -> dict:
    """Run the comparison over targets. ``planner`` maps a SMILES to its routes."""
    weights = weights or DEFAULT_WEIGHTS
    solved = 0
    lengths: list[int] = []
    leaf_fractions: list[float] = []
    base_q: dict[str, list[float]] = {"safety": [], "sustainability": [], "cost": []}
    reag_q: dict[str, list[float]] = {"safety": [], "sustainability": [], "cost": []}
    comp_q: dict[str, list[float]] = {"safety": [], "sustainability": [], "cost": []}
    comp_leaf: list[float] = []
    comp_agrees = 0
    per_target = []

    for name, smiles in targets:
        routes = planner(smiles)
        solved_routes = [r for r in routes if r.solved]
        is_solved = bool(solved_routes)
        solved += int(is_solved)
        if not is_solved:
            per_target.append({"name": name, "solved": False, "candidates": len(routes)})
            continue

        baseline, reagent = _select(solved_routes, weights)
        lengths.append(reagent.num_steps)
        leaf_fractions.append(largest_leaf_fraction(reagent))
        b_scores, r_scores = deterministic_scores(baseline), deterministic_scores(reagent)
        for obj in base_q:
            base_q[obj].append(b_scores[obj])
            reag_q[obj].append(r_scores[obj])

        # The weight-free reference. It cannot vary with the profile, so it is
        # the same line under every weighting -- which is the point: it says
        # what the candidates support before anyone expresses a preference.
        compromise = compromise_route(solved_routes) or reagent
        c_scores = deterministic_scores(compromise)
        for obj in comp_q:
            comp_q[obj].append(c_scores[obj])
        comp_leaf.append(largest_leaf_fraction(compromise))
        comp_agrees += int(compromise is reagent)
        per_target.append(
            {
                "name": name,
                "solved": True,
                "candidates": len(solved_routes),
                "baseline_safety": b_scores["safety"],
                "reagent_safety": r_scores["safety"],
                "changed_pick": baseline is not reagent,
                "largest_leaf_fraction": leaf_fractions[-1],
            }
        )

    return {
        "n_targets": len(targets),
        "solve_rate": solved / len(targets) if targets else 0.0,
        # Solve-rate counts a target as solved when every leaf is purchasable,
        # which counts buying a nearly finished molecule as success. On the
        # moderate set against a capped catalogue that is 10 of 25 targets, so
        # the honest figure is 0.60 where solve_rate reports 1.00. Reported
        # alongside rather than instead: both questions are legitimate, and
        # which one matters depends on whether you meant to build or to buy.
        "build_solve_rate": (
            sum(f < ADVANCED_LEAF for f in leaf_fractions) / len(targets)
            if targets
            else 0.0
        ),
        "avg_route_length": mean(lengths) if lengths else 0.0,
        "avg_largest_leaf_fraction": mean(leaf_fractions) if leaf_fractions else 0.0,
        "advanced_intermediate_routes": sum(f >= ADVANCED_LEAF for f in leaf_fractions),
        "baseline_quality": {o: (mean(v) if v else 0.0) for o, v in base_q.items()},
        "compromise_quality": {o: (mean(v) if v else 0.0) for o, v in comp_q.items()},
        "compromise_largest_leaf_fraction": mean(comp_leaf) if comp_leaf else 0.0,
        "compromise_agrees_with_weighted": comp_agrees,
        "reagent_quality": {o: (mean(v) if v else 0.0) for o, v in reag_q.items()},
        "per_target": per_target,
    }
