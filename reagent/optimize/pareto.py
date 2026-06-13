"""Pareto front over the assessment score vectors.

Instead of forcing one winner, expose the non-dominated routes: those that no
other route beats on every objective at once. All objectives are maximized
(higher assessment score = better).
"""

from __future__ import annotations

from reagent.core.models import Route
from reagent.optimize.aggregate import score_vector


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
