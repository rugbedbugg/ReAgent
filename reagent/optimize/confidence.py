"""Route confidence, derived from the single-step model's own weakest step.

Ranking always produces a best route, even when every candidate is one the model
barely believes in. Confidence flags that case so a low-trust route is labelled
rather than presented as a real answer. It reads min_policy_probability, the
weakest disconnection's model likelihood, which the search backend provides.
"""

from __future__ import annotations

from reagent.core.models import Route
from reagent.features.extract import compute_features

# (threshold on the weakest step's probability, label). Checked high to low.
BANDS: list[tuple[float, str]] = [
    (0.5, "high"),
    (0.2, "moderate"),
    (0.05, "low"),
]
TRUSTWORTHY_MIN = 0.2


def route_confidence(route: Route) -> tuple[str, float]:
    """Return a (label, weakest-step-probability) confidence read for a route."""
    if not route.features:
        compute_features(route)
    prob = route.features.get("feasibility", {}).get("min_policy_probability", 0.0)
    for threshold, label in BANDS:
        if prob >= threshold:
            return label, prob
    return "very-low", prob


def is_trustworthy(route: Route) -> bool:
    return route_confidence(route)[1] >= TRUSTWORTHY_MIN
