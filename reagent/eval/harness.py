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
from reagent.optimize.aggregate import DEFAULT_WEIGHTS

# The multi-objective effect only shows when objectives beyond feasibility carry
# weight, so evaluation reports both the feasibility-led default and a profile a
# safety/green-minded chemist might set.
WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "feasibility-led": dict(DEFAULT_WEIGHTS),
    "safety-tilted": {
        "feasibility": 0.20,
        "availability": 0.15,
        "cost": 0.15,
        "safety": 0.28,
        "sustainability": 0.14,
        "efficiency": 0.08,
    },
}


def _weighted(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.get(o, 0.0) for o in scores)
    if total_w == 0:
        return 0.0
    return sum(weights.get(o, 0.0) * s for o, s in scores.items()) / total_w


def _select(routes: list[Route], weights: dict[str, float]) -> tuple[Route, Route]:
    """Return (baseline_pick, reagent_pick) over solved routes."""
    scored = [(r, deterministic_scores(r)) for r in routes]
    baseline = max(scored, key=lambda rs: rs[1]["feasibility"])[0]
    reagent = max(scored, key=lambda rs: _weighted(rs[1], weights))[0]
    return baseline, reagent


def evaluate(
    targets: list[tuple[str, str]],
    planner: Callable[[str], list[Route]],
    weights: dict[str, float] | None = None,
) -> dict:
    """Run the comparison over targets. ``planner`` maps a SMILES to its routes."""
    weights = weights or DEFAULT_WEIGHTS
    solved = 0
    lengths: list[int] = []
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
            }
        )

    return {
        "n_targets": len(targets),
        "solve_rate": solved / len(targets) if targets else 0.0,
        "avg_route_length": mean(lengths) if lengths else 0.0,
        "baseline_quality": {o: (mean(v) if v else 0.0) for o, v in base_q.items()},
        "reagent_quality": {o: (mean(v) if v else 0.0) for o, v in reag_q.items()},
        "per_target": per_target,
    }
