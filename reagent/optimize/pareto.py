"""Pareto front over the assessment score vectors.

Instead of forcing one winner, expose the non-dominated routes: those that no
other route beats on every objective at once. All objectives are maximized
(higher assessment score = better).
"""

from __future__ import annotations

from reagent.core.models import Route
from reagent.optimize.aggregate import normalized_vectors, score_vector


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
    """Pick the best-balanced route on the Pareto front, using no weights at all.

    Every selection rule in this project so far needs a weight vector, and the
    hard-set measurements show weights often fail to discriminate: two quite
    different profiles picked identical routes on all ten targets, because
    genuine building-block routes to one target differ little on most
    objectives. A rule that needs no weights sidesteps that entirely, and gives
    the evaluation something to compare the weighted rule against.

    The rule is the n-dimensional reading of a knee point: normalize the
    objectives across the candidate set, then take the non-dominated route whose
    vector sits closest to the ideal point -- 1.0 on everything, which is
    normally unattainable. Trading a lot of one objective for a little of
    another moves a route away from that corner, so the winner is the one making
    no such lopsided trade.
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

    return min(front, key=distance_to_ideal)
