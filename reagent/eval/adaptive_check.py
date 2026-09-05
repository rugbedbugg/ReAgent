"""Does the feedback loop actually learn a user's preference?

The adaptive layer shifts objective weights toward whatever distinguished the
route a user said they preferred, and later runs plan with the shifted weights.
That is a claim about behaviour over a sequence of targets, and unit tests on a
single update step cannot check it: a step can move the weights in the right
direction every time and still never arrive, or oscillate, or drift somewhere
that ranks routes worse than the defaults did.

So this simulates a user who has a fixed hidden preference, shows them the
candidates for one target at a time, takes the route that preference would
actually choose as the feedback, and measures whether the learned weights
converge on it.

The measurement is **regret**: the hidden-utility gap between the route the
learned weights recommend and the best route available for that target. Zero
means the learner picked what the user wanted. If the loop works, regret on
later targets is lower than on earlier ones. Comparing first half against second
half, rather than to a fixed baseline, keeps targets with no good route from
counting as failures to learn.

Everything here is a pure function over score vectors, so a run is deterministic
and needs no planner once the vectors are cached.
"""

from __future__ import annotations

from statistics import mean

from reagent.adaptive.weights import update_from_preference
from reagent.optimize.aggregate import DEFAULT_WEIGHTS, normalized_vectors, weighted_from_vector


def _best_index(vectors: list[dict[str, float]], weights: dict[str, float]) -> int:
    """Index of the highest-scoring candidate under one weight vector.

    Ties break on the vector's own contents rather than on its position, for the
    same reason ranking does: ``max`` keeps the first of several equal scores, so
    position would otherwise decide, and position is an accident of the order the
    candidates arrived in.
    """
    def key(i: int) -> tuple:
        return (-weighted_from_vector(vectors[i], weights), sorted(vectors[i].items()))

    return min(range(len(vectors)), key=key)


def regret(
    vectors: list[dict[str, float]],
    chosen: int,
    hidden: dict[str, float],
) -> float:
    """How much hidden utility the chosen route gives up against the best available.

    Normalized by the spread of hidden utility across the candidates, so a target
    whose routes are nearly identical cannot dominate the average. A target with
    no spread at all contributes zero: nothing was there to get wrong.
    """
    utilities = [weighted_from_vector(v, hidden) for v in vectors]
    best, worst = max(utilities), min(utilities)
    if best <= worst:
        return 0.0
    return (best - utilities[chosen]) / (best - worst)


def simulate(
    per_target: list[list[dict[str, float]]],
    hidden: dict[str, float],
    lr: float = 0.5,
    start: dict[str, float] | None = None,
) -> dict:
    """Walk the targets in order, learning from each simulated choice.

    ``per_target`` is one candidate set per target, each a list of raw objective
    vectors. Vectors are normalized within their own candidate set, exactly as
    ranking and the stored episodes do, so the update sees comparable scales.
    """
    weights = dict(start or DEFAULT_WEIGHTS)
    trace: list[dict] = []

    for vectors in per_target:
        if len(vectors) < 2:
            continue  # nothing to choose between, and nothing to learn from
        normalized = normalized_vectors(vectors)
        if not any(normalized):
            continue  # every objective tied across the candidate set

        recommended = _best_index(normalized, weights)
        preferred = _best_index(normalized, hidden)

        trace.append(
            {
                "regret": regret(normalized, recommended, hidden),
                "agreed": recommended == preferred,
                "weights": dict(weights),
            }
        )
        weights = update_from_preference(
            weights,
            normalized[preferred],
            [v for i, v in enumerate(normalized) if i != preferred],
            lr=lr,
        )

    # Two rounds are the minimum that can show a trend: with one, the second
    # half is empty and there is nothing to compare the first against.
    if len(trace) < 2:
        return {"rounds": len(trace), "learned_weights": weights, "trace": trace}

    half = max(1, len(trace) // 2)
    first, second = trace[:half], trace[half:]
    return {
        "rounds": len(trace),
        "regret_first_half": mean(t["regret"] for t in first),
        "regret_second_half": mean(t["regret"] for t in second),
        "agreement_first_half": sum(t["agreed"] for t in first) / len(first),
        "agreement_second_half": sum(t["agreed"] for t in second) / len(second),
        "start_weights": trace[0]["weights"],
        "learned_weights": weights,
        "trace": trace,
    }
