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


def route_signature(route: Route) -> tuple:
    """A stable identity for a route, used to break scoring ties.

    The search returns the same routes in a different order from run to run --
    measured on warfarin: identical route sets, three different orderings across
    three runs. Both ``max`` and Python's stable ``sorted`` keep the first of
    several equal scores, so ties were being broken by position, and position
    was not reproducible. That is what made pick counts move by one between
    otherwise identical runs.

    Ordering on this after the score makes the choice depend on the routes
    themselves rather than on the order they arrived in. Leaves first (what you
    buy), then step count, then the reaction SMILES, so two genuinely distinct
    routes never compare equal.
    """
    return (
        tuple(sorted(m.smiles for m in route.leaves)),
        route.num_steps,
        tuple(sorted(r.rsmi or r.product for r in route.reactions)),
    )


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
    # Descending score, then ascending signature: a stable order that does not
    # depend on the order the search happened to return the routes in.
    return sorted(routes, key=lambda r: (-r.scores["weighted"], route_signature(r)))


# The multi-objective effect only shows when objectives beyond feasibility carry
# weight, so evaluation reports both the feasibility-led default and a profile a
# safety/green-minded chemist might set.
WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "feasibility-led": dict(DEFAULT_WEIGHTS),
    "safety-tilted": {
        "feasibility": 0.20,
        "availability": 0.05,
        "cost": 0.15,
        "safety": 0.28,
        "construction": 0.10,
        "sustainability": 0.14,
        "efficiency": 0.08,
    },
    # Genuine building-block routes to one target differ little in hazard, so
    # tilting safety moves the pick less than it looks like it should. How much
    # of the molecule a route builds *does* vary across candidates, so this is
    # the profile that shows the selection layer expressing a preference.
    "build-it-yourself": {
        "feasibility": 0.20,
        "availability": 0.05,
        "cost": 0.10,
        "safety": 0.10,
        "construction": 0.30,
        "sustainability": 0.15,
        "efficiency": 0.10,
    },
    # Getting the compound in hand, not making it. Cost and step count lead;
    # `construction` is deliberately near zero, since buying an advanced
    # intermediate is the point of this profile rather than a failure of it.
    "source-led": {
        "feasibility": 0.25,
        "availability": 0.10,
        "cost": 0.30,
        "safety": 0.10,
        "construction": 0.02,
        "sustainability": 0.05,
        "efficiency": 0.18,
    },
}

# A mode is a weight profile plus, for "build", a hard constraint. The weights
# alone were measured to be too weak to express the intent: at 0.30 on
# `construction` the build profile changes 13 picks against the default's 11,
# because a soft preference cannot outvote a route that is simply shorter.
# The constraint lives in the stock layer, so a leaf that is most of the target
# is not purchasable at all and the route is never proposed.
MODES: dict[str, dict] = {
    # What ships today: rank on the default weights, buy whatever is for sale.
    "balanced": {
        "weights": "feasibility-led",
        "max_leaf_fraction": None,
        "help": "Rank on the default weights. No constraint on what may be bought.",
    },
    # For someone asking how to *make* the molecule.
    "build": {
        "weights": "build-it-yourself",
        "max_leaf_fraction": 0.6,
        "help": "Build rather than buy: reject leaves larger than 60% of the target.",
    },
    # For someone asking how to *obtain* it. Buying an advanced intermediate is
    # the goal here, not a defect, so nothing is constrained.
    "source": {
        "weights": "source-led",
        "max_leaf_fraction": None,
        "help": "Obtain it cheaply: favour cost and few steps, buying freely.",
    },
}


def mode_weights(mode: str) -> dict[str, float]:
    """The starting weight vector for a mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {', '.join(sorted(MODES))}")
    return dict(WEIGHT_PROFILES[MODES[mode]["weights"]])


def mode_leaf_fraction(mode: str) -> float | None:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {', '.join(sorted(MODES))}")
    return MODES[mode]["max_leaf_fraction"]

