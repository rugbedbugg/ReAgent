"""Pareto front over the assessment score vectors.

Instead of forcing one winner, expose the non-dominated routes: those that no
other route beats on every objective at once. All objectives are maximized
(higher assessment score = better).
"""

from __future__ import annotations

from reagent.core.models import Route
from reagent.optimize.aggregate import normalized_vectors, route_signature, score_vector


def _dominates(a: dict[str, float], b: dict[str, float], objectives: list[str]) -> bool:
    """True if a is at least as good as b everywhere and strictly better somewhere."""
    at_least = all(a.get(o, 0.0) >= b.get(o, 0.0) for o in objectives)
    strictly = any(a.get(o, 0.0) > b.get(o, 0.0) for o in objectives)
    return at_least and strictly


def pareto_front(routes: list[Route]) -> list[Route]:
    """Return the non-dominated routes, preserving input order."""
    vectors = [score_vector(r) for r in routes]
    objectives = sorted({o for v in vectors for o in v})
    front: list[Route] = []
    for i, route in enumerate(routes):
        dominated = any(
            _dominates(vectors[j], vectors[i], objectives)
            for j in range(len(routes))
            if j != i
        )
        if not dominated:
            front.append(route)
    return front


def compromise_route(routes: list[Route]) -> Route | None:
    """The non-dominated route closest to the ideal point. A diagnostic, not a
    recommendation.

    The n-dimensional reading of a knee point: normalize the objectives across
    the candidate set, then take the Pareto-front route whose vector sits
    closest to 1.0 on everything. Trading a lot of one objective for a little of
    another moves a route away from that corner, so the winner is the one making
    no such lopsided trade.

    **Measured, it is worse than the feasibility-only baseline** -- on the hard
    set at <=14 it scores safety 0.344, sustainability 0.824 and cost 0.514,
    against the baseline's 0.532 / 0.900 / 0.532 and the weighted pick's 0.628 /
    0.914 / 0.578, and it agrees with the weighted pick on 1 of 10 targets. So
    it is reported alongside the weighted rule rather than used as one.

    The reason is that an equal-distance rule is not weight-free, whatever it
    looks like: treating every axis alike *is* a uniform weight vector, which
    silently cuts feasibility from 0.30 to 1/7 and promotes the noisier proxies
    to equal standing. Balancing seven objectives beats optimizing none of them,
    and loses to optimizing the right ones. Its value here is as the control
    that shows the tuned weights are doing real work -- a question the profile
    comparisons could not answer, since those profiles largely agree with each
    other.
    """
    if not routes:
        return None

    front = pareto_front(routes)
    if len(front) == 1:
        return front[0]

    normalized = normalized_vectors([score_vector(r) for r in routes])
    by_id = {id(route): vector for route, vector in zip(routes, normalized, strict=True)}

    def distance_to_ideal(route: Route) -> float:
        vector = by_id[id(route)]
        # Every objective was constant across candidates, so nothing separates
        # them; order decides, as it does for a lone route.
        return sum((1.0 - value) ** 2 for value in vector.values()) if vector else 0.0

    return min(front, key=lambda r: (distance_to_ideal(r), route_signature(r)))
