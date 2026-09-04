"""Validate the LLM agent scores against the deterministic reference.

The deterministic scorer applies the agent rubrics numerically, so it is the
reference an ideal agent would match. Running the real agents over live routes
and comparing gives a measured reliability picture: how far each objective's
score drifts, how often the agent's ranking still agrees, and how often a reply
fails to parse.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from reagent.agents.orchestrator import Orchestrator
from reagent.core.models import Route
from reagent.features.scoring import deterministic_scores
from reagent.optimize.aggregate import DEFAULT_WEIGHTS, weighted_score


def _weighted_from_scores(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.get(o, 0.0) for o in scores)
    return sum(weights.get(o, 0.0) * s for o, s in scores.items()) / total_w if total_w else 0.0


def check_agents(
    targets: list[tuple[str, str]],
    planner: Callable[[str], list[Route]],
    orchestrator: Orchestrator,
    routes_per: int = 2,
    weights: dict[str, float] | None = None,
) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    errors: dict[str, list[float]] = {}
    parse_failures = 0
    total = 0
    ranking_agree = 0
    ranked_targets = 0
    per_target = []

    for name, smiles in targets:
        routes = [r for r in planner(smiles) if r.solved][:routes_per]
        if not routes:
            per_target.append({"name": name, "routes": 0})
            continue
        for route in routes:
            orchestrator.assess(route)

        for route in routes:
            reference = deterministic_scores(route)
            for a in route.assessments:
                total += 1
                if a.rationale.startswith(("Unparseable", "Malformed")):
                    parse_failures += 1
                    continue
                if a.objective in reference:
                    errors.setdefault(a.objective, []).append(abs(a.score - reference[a.objective]))

        if len(routes) > 1:
            ranked_targets += 1
            agent_top = max(routes, key=lambda r: weighted_score(r, weights))
            ref_top = max(routes, key=lambda r: _weighted_from_scores(deterministic_scores(r), weights))
            agree = agent_top is ref_top
            ranking_agree += int(agree)
            per_target.append({"name": name, "routes": len(routes), "ranking_agrees": agree})
        else:
            per_target.append({"name": name, "routes": len(routes)})

    mae = {obj: mean(errs) for obj, errs in errors.items() if errs}
    all_errs = [e for errs in errors.values() for e in errs]
    return {
        "objective_mae": mae,
        "overall_mae": mean(all_errs) if all_errs else 0.0,
        "parse_failure_rate": parse_failures / total if total else 0.0,
        "ranking_agreement": ranking_agree / ranked_targets if ranked_targets else None,
        "assessments": total,
        "per_target": per_target,
    }
