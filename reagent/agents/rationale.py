"""Assemble the human-readable justification for a ranked set of routes.

The narrative is built from the structured assessments already on each route, so
it reflects exactly what the agents scored and cited, with nothing re-derived or
invented. It states which route won, the objectives that carried it, why each
alternative was passed over, and the precedent behind the top route.
"""

from __future__ import annotations

from reagent.core.models import Route

# A gap this large on one objective is worth calling out as a reason to reject.
NOTABLE_GAP = 0.2


def _lead_objectives(route: Route, n: int = 3) -> list[tuple[str, float]]:
    vector = route.scores.get("vector", {})
    return sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:n]


def _rationale_for(route: Route, objective: str) -> str:
    for a in route.assessments:
        if a.objective == objective:
            return a.rationale
    return ""


def build_rationale(ranked: list[Route], numbers: dict[int, int]) -> str:
    """Return the "why this route" narrative for an already-ranked route list."""
    if not ranked:
        return "No routes to justify."

    winner = ranked[0]
    lines = [
        f"Recommended: Route {numbers[id(winner)]} "
        f"(weighted score {winner.scores['weighted']:.3f}, {winner.num_steps} steps)."
    ]

    from reagent.optimize.confidence import route_confidence

    conf, prob = route_confidence(winner)
    if prob < 0.2:
        lines.append(
            f"Caveat: {conf} confidence (weakest step {prob:.2f}); the base model "
            "distrusts this route, so the ranking below is between weak options."
        )

    leads = _lead_objectives(winner)
    strengths = ", ".join(f"{obj} {score:.2f}" for obj, score in leads)
    lines.append(f"Carried by: {strengths}.")

    winner_vec = winner.scores.get("vector", {})
    for route in ranked[1:]:
        n = numbers[id(route)]
        vec = route.scores.get("vector", {})
        shortfalls = [
            (obj, vec[obj], winner_vec.get(obj, 0.0))
            for obj in vec
            if winner_vec.get(obj, 0.0) - vec[obj] >= NOTABLE_GAP
        ]
        shortfalls.sort(key=lambda t: t[2] - t[1], reverse=True)
        if shortfalls:
            reasons = "; ".join(
                f"{obj} {loser:.2f} vs {win:.2f} ({_rationale_for(route, obj)})"
                for obj, loser, win in shortfalls[:2]
            )
            lines.append(
                f"Route {n} passed over (score {route.scores['weighted']:.3f}): {reasons}"
            )
        else:
            lines.append(
                f"Route {n} (score {route.scores['weighted']:.3f}): no single decisive "
                "weakness, edged out on the weighted total."
            )

    evidence = next((a.evidence for a in winner.assessments if a.objective == "feasibility"), [])
    if evidence:
        lines.append("Precedent for the recommended route:")
        lines.extend(f"  - {e}" for e in evidence)

    return "\n".join(lines)
