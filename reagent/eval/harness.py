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
from dataclasses import dataclass
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
__all__ = [
    "WEIGHT_PROFILES", "evaluate", "largest_leaf_fraction", "ADVANCED_LEAF",
    "rank_routes_deterministic", "rank_baseline_feasibility", "rank_candidates",
    "RankingResult",
]


@dataclass
class RankingResult:
    """Result of ranking with metadata."""
    reagent_ranked: list[Route]
    baseline_ranked: list[Route]
    weight_vector: dict[str, float]


def rank_routes_deterministic(
    routes: list[Route],
    weights: dict[str, float] | None = None,
) -> list[Route]:
    """
    Deterministically rank routes using the same logic as harness._select.

    This is the shared ranking function used by both evaluation and literature benchmark.
    """
    weights = weights or DEFAULT_WEIGHTS
    solved = [r for r in routes if r.solved]
    if not solved:
        return []

    # Compute deterministic scores
    vectors = [deterministic_scores(r) for r in solved]
    normalized = normalized_vectors(vectors)

    # Weighted score
    scored = []
    for route, norm in zip(solved, normalized):
        score = weighted_from_vector(norm, weights)
        scored.append((route, score))

    # Sort: descending score, then ascending route_signature (tie-break)
    scored.sort(key=lambda x: (-x[1], route_signature(x[0])))

    return [r for r, _ in scored]


def rank_baseline_feasibility(routes: list[Route]) -> list[Route]:
    """Rank by raw feasibility descending, then route_signature ascending."""
    solved = [r for r in routes if r.solved]
    if not solved:
        return []

    vectors = [deterministic_scores(r) for r in solved]
    scored = []
    for route, vec in zip(solved, vectors):
        scored.append((route, vec.get("feasibility", 0.0)))

    scored.sort(key=lambda x: (-x[1], route_signature(x[0])))
    return [r for r, _ in scored]


def rank_candidates(
    routes: list[Route],
    weights: dict[str, float] | None = None,
) -> RankingResult:
    """
    Rank candidates using the authoritative deterministic method.

    Args:
        routes: The candidate routes to rank.
        weights: Optional weight vector for ReAgent ranking. Defaults to DEFAULT_WEIGHTS.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    reagent_ranked = rank_routes_deterministic(routes, weights)
    baseline_ranked = rank_baseline_feasibility(routes)

    return RankingResult(
        reagent_ranked=reagent_ranked,
        baseline_ranked=baseline_ranked,
        weight_vector=weights,
    )


def _select(routes: list[Route], weights: dict[str, float]) -> tuple[Route, Route]:
    """Return (baseline_pick, reagent_pick) over solved routes.

    ReAgent's pick uses the same candidate-normalized aggregation the CLI ranks
    with, so the comparison measures the real selection strategy.
    """
    solved = [r for r in routes if r.solved]
    if not solved:
        raise ValueError("_select called with no solved routes")

    # Use shared ranking functions with the provided weights
    ranking = rank_candidates(solved, weights)
    reagent_pick = ranking.reagent_ranked[0]

    # Baseline uses raw feasibility (from shared ranking result)
    baseline_pick = ranking.baseline_ranked[0]

    return baseline_pick, reagent_pick


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
        "reagent_weight_vector": dict(weights),
        "per_target": per_target,
    }
