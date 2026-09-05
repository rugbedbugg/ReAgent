"""Weighted-sum aggregation over the per-objective assessment scores.

Collapses a route's six Assessment scores into one ranking number. Weights are
tunable (the adaptive layer learns them from feedback); the raw score vector is
kept alongside so nothing is lost to the scalar and the Pareto view can use it.

Objectives are min-max normalized across the candidate set before weighting.
Raw scores are not on comparable scales: over a typical candidate set
feasibility swings several times wider than cost, safety, or efficiency, so
weighting the raw values lets the widest-ranging objective decide the ranking
regardless of what the weights say. Normalizing first is what makes a weight
mean "how much I care about this" rather than "how much this objective happens
to vary".

Rescaling is floored at ``MIN_SPAN``. Pure min-max would stretch whatever spread
a candidate set happens to show into a full 0..1 swing, so on a small set a
meaningless 0.02 feasibility gap would count as heavily as a decisive safety
gap. The floor keeps small real differences small while still removing the
scale advantage of a genuinely wide-ranging objective.
"""

from __future__ import annotations

from reagent.core.models import Route

# Sums to 1.0. Feasibility and availability lead: a route that will not work or
# cannot be sourced is worthless regardless of how green or cheap it looks.
# ``construction`` carries real weight despite being last-added: every other
# objective rewards buying an advanced intermediate, so without it the optimizer
# has no reason to prefer a synthesis over a purchase.
# ``construction`` is funded entirely out of ``availability``, which nominally
# held 0.25 but decides nothing: every *solved* route has all leaves in stock by
# definition, so availability is constant across the candidate set and the
# normalizer drops it. Taking weight from any other objective changes rankings
# that were measured and are correct -- taking it from here does not.
DEFAULT_WEIGHTS: dict[str, float] = {
    "feasibility": 0.30,
    "availability": 0.10,
    "cost": 0.15,
    "safety": 0.15,
    "construction": 0.15,
    "sustainability": 0.08,
    "efficiency": 0.07,
}


def score_vector(route: Route) -> dict[str, float]:
    return {a.objective: a.score for a in route.assessments}


# Objectives whose candidates differ by less than this are treated as nearly
# tied rather than rescaled to a full swing. The floor has to sit below the
# spread a real objective shows, or that objective is handicapped against a
# wider-ranging one and the normalization achieves nothing.
#
# Measured across the candidate sets of eight hard targets, median spread per
# objective: cost 0.179, safety 0.169, sustainability 0.154, construction 0.116,
# efficiency 0.075, feasibility 0.025, availability 0.000. So feasibility --
# which carries the largest weight -- sits *under* this floor on seven of the
# eight, and contributes a damped difference rather than its own full range.
#
# That looks like a miscalibration and was nearly "fixed" by giving feasibility
# its own lower floor. It is not one. Feasibility is a product of model
# probabilities, and a 0.02 gap between two candidates the same model produced
# is not evidence that one route is better; min-max would stretch it to a full
# swing, and at weight 0.30 it would then outvote a 0.6 difference in safety --
# see ``test_small_spread_is_not_stretched_into_a_decision``. Whether small
# likelihood differences mean anything is answerable only against reference
# routes, which this project does not have. Until then the floor stays, and
# feasibility's nominal 0.30 buys less than it appears to.
MIN_SPAN = 0.10


def normalized_vectors(vectors: list[dict[str, float]]) -> list[dict[str, float]]:
    """Min-max normalize each objective across the candidate set.

    Each objective is rescaled over the candidates actually on offer, so the
    weights decide the trade-off rather than the spread each objective happens
    to show. The divisor is floored at :data:`MIN_SPAN`, so an objective whose
    candidates barely differ contributes a correspondingly small difference
    instead of being stretched to a full 0..1 swing.

    An objective every candidate scores identically carries no comparative
    information and is dropped: among solved routes availability is 1.0 by
    construction (a route is solved iff every leaf is in stock), and on
    single-step targets efficiency is constant too. Dropping them beats adding a
    constant that dilutes the objectives which do separate the routes.
    """
    objectives = sorted({o for v in vectors for o in v})
    spans: dict[str, tuple[float, float]] = {}
    for objective in objectives:
        column = [v[objective] for v in vectors if objective in v]
        if column and max(column) > min(column):
            spans[objective] = (min(column), max(max(column) - min(column), MIN_SPAN))
    return [
        {o: (v[o] - lo) / span for o, (lo, span) in spans.items() if o in v}
        for v in vectors
    ]


def weighted_from_vector(
    vector: dict[str, float], weights: dict[str, float] | None = None
) -> float:
    """Weighted mean over whichever objectives the vector actually carries.

    Weights are renormalized over those objectives, so a missing agent -- or an
    objective dropped for being constant -- does not drag the total toward zero.
    """
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights.get(o, 0.0) for o in vector)
    if total_w == 0.0:
        return 0.0
    return sum(weights.get(o, 0.0) * s for o, s in vector.items()) / total_w


def weighted_score(route: Route, weights: dict[str, float] | None = None) -> float:
    """Absolute weighted mean of one route's raw scores.

    Standalone quality figure for a single route. Ranking a candidate set goes
    through :func:`rank_routes`, which normalizes across candidates first.
    """
    return weighted_from_vector(score_vector(route), weights)


def rank_routes(routes: list[Route], weights: dict[str, float] | None = None) -> list[Route]:
    """Return routes sorted best-first, recording scores on each route.

    Ordering uses the candidate-normalized score (``weighted``). The absolute
    figure is kept as ``weighted_raw`` and the raw vector as ``vector``, so the
    Pareto view, the rationale, and episodic memory still see real scores.
    """
    vectors = [score_vector(route) for route in routes]
    normalized = normalized_vectors(vectors)
    if not any(normalized):
        # Nothing separates the candidates (a lone route, or exact ties): fall
        # back to raw scores so the reported number stays meaningful.
        normalized = vectors
    for route, raw, norm in zip(routes, vectors, normalized, strict=True):
        route.scores = {
            "weighted": weighted_from_vector(norm, weights),
            "weighted_raw": weighted_from_vector(raw, weights),
            "vector": raw,
            "normalized": norm,
        }
    return sorted(routes, key=lambda r: r.scores["weighted"], reverse=True)
