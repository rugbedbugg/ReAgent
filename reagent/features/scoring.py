"""Deterministic scoring of a route's objectives from its features.

Applies the objective rubrics numerically: given the raw feature facts, it
returns the exact score each rubric intends. It is used both to evaluate the
LLM agents against a reference and, in hybrid mode, as the score itself for the
formula-based objectives, leaving the LLM to supply the rationale.
"""

from __future__ import annotations

from reagent.core.models import Route
from reagent.features.extract import compute_features

_SEVERITY_CAP = 3


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _brenk_safety(safety: dict) -> float:
    """Score the structural-alert screen from intensive facts only.

    The earlier form subtracted 0.1 per *distinct* hazard found anywhere in the
    route. That set grows with the number of molecules, so a longer route scored
    worse for being longer even when every compound it handled was equally
    benign -- measured on sertraline, the real three-step route scored 0.300
    against 0.400 for the one-step route that buys the penultimate intermediate,
    despite both handling methyl iodide and having identical hazard density.
    Safety became a proxy for shortness, which ``efficiency`` already scores, and
    it rewarded outsourcing the chemistry rather than doing it.

    So the score now depends only on how bad the worst single compound is and
    what fraction of the route is hazardous. Neither term moves when steps are
    added at constant hazard. The endpoints of the old rubric are kept: a clean
    route is categorically 1.0, any hazard caps the score at 0.6, and the floor
    is 0.1.
    """
    if not safety["max_molecule_hazards"]:
        return 1.0
    severity = min(safety["max_molecule_hazards"], _SEVERITY_CAP) / _SEVERITY_CAP
    return _clamp(0.6 - 0.35 * severity - 0.15 * safety["hazard_density"])


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
        safety = _brenk_safety(f["safety"])

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
