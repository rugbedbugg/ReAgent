"""Deterministic scoring of a route's objectives from its features.

Applies the objective rubrics numerically: given the raw feature facts, it
returns the exact score each rubric intends. It is used both to evaluate the
LLM agents against a reference and, in hybrid mode, as the score itself for the
formula-based objectives, leaving the LLM to supply the rationale.
"""

from __future__ import annotations

from reagent.core.models import Route
from reagent.features.extract import compute_features


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def deterministic_scores(route: Route) -> dict[str, float]:
    """Per-objective scores in 0..1 (higher better), matching the agent rubrics."""
    if not route.features:
        compute_features(route)
    f = route.features

    # The filter model's forward-plausibility gates the suggestion prior: a route
    # the model likes but that contains an implausible reaction is penalized.
    feasibility = _clamp(
        f["feasibility"]["min_policy_probability"] * f["feasibility"].get("min_filter_feasibility", 1.0)
    )
    availability = _clamp(f["availability"]["in_stock_fraction"])
    cost = _clamp(1.0 - f["cost"]["cost_proxy"] / 120.0)

    if "ghs_safety" in f["safety"]:  # real GHS data present, prefer it
        safety = _clamp(f["safety"]["ghs_safety"])
    else:
        n_hazards = len(f["safety"]["distinct_hazards"])
        safety = 1.0 if n_hazards == 0 else _clamp(0.5 - 0.1 * (n_hazards - 1))

    penalty = 0.2 if f["sustainability"]["pmi_proxy"] > 3 else 0.0
    sustainability = _clamp(f["sustainability"]["mean_step_atom_economy"] - penalty)

    steps = f["efficiency"]["num_steps"]
    convergent_bonus = 0.1 if f["efficiency"]["convergence"] > 1.0 else 0.0
    efficiency = _clamp(0.95 - 0.15 * max(0, steps - 1) + convergent_bonus)

    return {
        "feasibility": feasibility,
        "availability": availability,
        "cost": cost,
        "safety": safety,
        "sustainability": sustainability,
        "efficiency": efficiency,
    }
