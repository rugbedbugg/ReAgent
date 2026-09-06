"""Behavioural tests for the feedback loop, over a sequence rather than a step.

`test_adaptive.py` checks one update in isolation. That is not enough to know
the loop works: a step can move in the right direction every time and still
never arrive, oscillate, or settle somewhere that ranks routes worse than the
defaults did. These drive the loop over a run of targets and assert on what
comes out the far end.

Two of them exist specifically as regression guards. A weight must be free to
fall, and an objective the user keeps rejecting must be free to reach zero.
Anchoring weights to their defaults sounds like stability and is not: it turns
the update into a ratchet, and the loop stops converging. Measured, that change
took regret from 0.020 falling to 0.017 up to 0.043 rising to 0.125.
"""

import random

from reagent.adaptive.weights import update_from_preference
from reagent.eval.adaptive_check import simulate
from reagent.optimize.aggregate import DEFAULT_WEIGHTS

OBJECTIVES = list(DEFAULT_WEIGHTS)


def _vector(**overrides: float) -> dict[str, float]:
    return {o: overrides.get(o, 0.5) for o in OBJECTIVES}


def _safety_favouring_targets(n: int = 12) -> list[list[dict[str, float]]]:
    """Each target offers a safe-but-hard route, a cheap-but-hazardous one, and
    a middling one. A safety-loving user always wants the first."""
    return [
        [
            _vector(safety=0.9, cost=0.2, feasibility=0.4),
            _vector(safety=0.2, cost=0.9, feasibility=0.8),
            _vector(safety=0.5, cost=0.5, feasibility=0.6),
        ]
        for _ in range(n)
    ]


def test_weights_always_stay_on_the_simplex():
    """Whatever the feedback, the result is a probability vector."""
    random.seed(7)
    weights = dict(DEFAULT_WEIGHTS)
    for _ in range(200):
        preferred = {o: random.random() for o in OBJECTIVES}
        others = [{o: random.random() for o in OBJECTIVES} for _ in range(3)]
        weights = update_from_preference(weights, preferred, others)
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert all(w >= 0.0 for w in weights.values())


def test_a_consistently_rejected_objective_converges_toward_zero():
    """Regression guard: weights must be able to go *down*, and keep going.

    A clamp that floors every objective at its default defeats this, but not
    visibly after one step: the floored weight still shrinks once the vector is
    renormalized, because the others grew. Only the trajectory separates the
    two. Ten rounds of the same feedback takes feasibility from 0.30 to 0.0006
    under the real rule and leaves it stuck at 0.2704 under a floor, so the
    threshold below sits in the gap rather than near either value.
    """
    weights = dict(DEFAULT_WEIGHTS)
    preferred = _vector(safety=0.9, feasibility=0.1)
    others = [_vector(safety=0.1, feasibility=0.9)]

    for _ in range(10):
        weights = update_from_preference(weights, preferred, others)

    assert weights["feasibility"] < 0.05, (
        f"feasibility stalled at {weights['feasibility']:.4f}; a rejected "
        "objective is being held up by an anchor"
    )
    assert weights["safety"] > DEFAULT_WEIGHTS["safety"]


def test_a_distinguishing_objective_gains_weight():
    weights = dict(DEFAULT_WEIGHTS)
    preferred = _vector(cost=0.95)
    others = [_vector(cost=0.05), _vector(cost=0.10)]

    updated = update_from_preference(weights, preferred, others)

    assert updated["cost"] > weights["cost"]


def test_repeated_identical_feedback_converges_rather_than_oscillating():
    """The same preference, over and over, should settle. Successive steps get
    smaller and the direction never reverses."""
    weights = dict(DEFAULT_WEIGHTS)
    preferred = _vector(safety=0.9)
    others = [_vector(safety=0.1)]

    deltas = []
    for _ in range(15):
        updated = update_from_preference(weights, preferred, others)
        deltas.append(updated["safety"] - weights["safety"])
        weights = updated

    assert all(d > 0 for d in deltas), "direction reversed: that is oscillation"
    assert deltas[-1] < deltas[0], "steps are not shrinking: that is not convergence"
    assert weights["safety"] < 1.0


def test_regret_falls_and_agreement_rises_over_a_sequence():
    """The claim the whole adaptive layer rests on, and the one docs/EVALUATION.md
    reports."""
    hidden = {**{o: 0.05 for o in OBJECTIVES}, "safety": 0.80}

    result = simulate(_safety_favouring_targets(), hidden)

    assert result["rounds"] == 12
    assert result["regret_second_half"] < result["regret_first_half"]
    assert result["agreement_second_half"] > result["agreement_first_half"]


def test_learning_can_drive_a_rejected_objective_to_the_floor():
    """Feasibility starts as the heaviest objective at 0.30. A user who
    consistently picks against it must be able to strip it out entirely.

    This is the measured behaviour the evaluation depends on, not an accident:
    it is what lets the loop reach the user's route rather than a compromise
    between their preference and the shipped defaults.
    """
    hidden = {**{o: 0.05 for o in OBJECTIVES}, "safety": 0.80}

    result = simulate(_safety_favouring_targets(), hidden)
    learned = result["learned_weights"]

    assert learned["feasibility"] < 0.01
    assert learned["safety"] > DEFAULT_WEIGHTS["safety"]
