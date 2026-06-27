"""Feedback-driven objective weights.

Weights start at the defaults and shift when a user says which route they
actually preferred. The update is an exponentiated-gradient step: objectives on
which the preferred route beats the others gain weight, those it trails lose
weight, then the vector is renormalized. Learned weights persist to a JSON file
and are picked up by later planning runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from reagent.core.config import DATA_DIR
from reagent.optimize.aggregate import DEFAULT_WEIGHTS

WEIGHTS_PATH = DATA_DIR / "weights.json"


def load_weights(path: str | Path = WEIGHTS_PATH) -> dict[str, float]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights: dict[str, float], path: str | Path = WEIGHTS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2), encoding="utf-8")


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def update_from_preference(
    weights: dict[str, float],
    preferred: dict[str, float],
    others: list[dict[str, float]],
    lr: float = 0.5,
) -> dict[str, float]:
    """Nudge weights toward objectives that distinguish the preferred route."""
    updated = dict(weights)
    for objective, weight in weights.items():
        pref = preferred.get(objective, 0.0)
        rival_scores = [o.get(objective, 0.0) for o in others]
        rival = sum(rival_scores) / len(rival_scores) if rival_scores else pref
        updated[objective] = weight * math.exp(lr * (pref - rival))
    return _normalize(updated)
