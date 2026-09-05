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
    normalized_vectors,
    weighted_from_vector,
)

# The multi-objective effect only shows when objectives beyond feasibility carry
# weight, so evaluation reports both the feasibility-led default and a profile a
# safety/green-minded chemist might set.
WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "feasibility-led": dict(DEFAULT_WEIGHTS),
    "safety-tilted": {
        "feasibility": 0.20,
        "availability": 0.05,
        "cost": 0.15,
        "safety": 0.28,
        "construction": 0.10,
        "sustainability": 0.14,
        "efficiency": 0.08,
    },
    # Genuine building-block routes to one target differ little in hazard, so
    # tilting safety moves the pick less than it looks like it should. How much
    # of the molecule a route builds *does* vary across candidates, so this is
    # the profile that shows the selection layer expressing a preference.
    "build-it-yourself": {
        "feasibility": 0.20,
        "availability": 0.05,
        "cost": 0.10,
        "safety": 0.10,
        "construction": 0.30,
        "sustainability": 0.15,
        "efficiency": 0.10,
    },
}


def _select(routes: list[Route], weights: dict[str, float]) -> tuple[Route, Route]:
    """Return (baseline_pick, reagent_pick) over solved routes.

    ReAgent's pick uses the same candidate-normalized aggregation the CLI ranks
    with, so the comparison measures the real selection strategy.
    """
    vectors = [deterministic_scores(r) for r in routes]
    normalized = normalized_vectors(vectors)
    baseline = max(range(len(routes)), key=lambda i: vectors[i]["feasibility"])
    reagent = max(range(len(routes)), key=lambda i: weighted_from_vector(normalized[i], weights))
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
        "avg_route_length": mean(lengths) if lengths else 0.0,
        "avg_largest_leaf_fraction": mean(leaf_fractions) if leaf_fractions else 0.0,
        "advanced_intermediate_routes": sum(f >= ADVANCED_LEAF for f in leaf_fractions),
        "baseline_quality": {o: (mean(v) if v else 0.0) for o, v in base_q.items()},
        "reagent_quality": {o: (mean(v) if v else 0.0) for o, v in reag_q.items()},
        "per_target": per_target,
    }
