"""Weighted-sum aggregation over the per-objective assessment scores.

Collapses a route's six Assessment scores into one ranking number. Weights are
tunable (the adaptive layer will learn them later); the raw score vector is kept
alongside so nothing is lost to the scalar and the Pareto view can use it.
"""

from __future__ import annotations

from reagent.core.models import Route

# Sums to 1.0. Feasibility and availability lead: a route that will not work or
# cannot be sourced is worthless regardless of how green or cheap it looks.
DEFAULT_WEIGHTS: dict[str, float] = {
    "feasibility": 0.30,
    "availability": 0.25,
    "cost": 0.15,
    "safety": 0.15,
    "sustainability": 0.08,
    "efficiency": 0.07,
}


def score_vector(route: Route) -> dict[str, float]:
    return {a.objective: a.score for a in route.assessments}


def weighted_score(route: Route, weights: dict[str, float] | None = None) -> float:
    """Weighted mean of the assessment scores over the objectives present.

    Weights are renormalized over whichever objectives the route actually has,
    so a missing agent does not silently drag the total toward zero.
    """
    weights = weights or DEFAULT_WEIGHTS
    vec = score_vector(route)
    total_w = sum(weights.get(o, 0.0) for o in vec)
    if total_w == 0.0:
        return 0.0
    return sum(weights.get(o, 0.0) * s for o, s in vec.items()) / total_w


def rank_routes(routes: list[Route], weights: dict[str, float] | None = None) -> list[Route]:
    """Return routes sorted best-first, recording scores on each route."""
    for route in routes:
        route.scores = {
            "weighted": weighted_score(route, weights),
            "vector": score_vector(route),
        }
    return sorted(routes, key=lambda r: r.scores["weighted"], reverse=True)
