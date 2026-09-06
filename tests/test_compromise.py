"""Order-independence of the compromise route.

`compromise_route` is a diagnostic control, not a recommendation, but it still
has to answer the same way every time it is asked. The search returns the same
routes in a different order from run to run, and `min` keeps the first of
several equal keys, so anything that ties has to be settled on the routes
themselves rather than on the order they arrived in.

Tying on score is the interesting case, and it is the one a naive test misses:
asserting the winner is *somewhere in the input list* is true whatever the
function does. These build routes that score identically but differ in what
they buy, so a position-dependent implementation returns different routes for
different orderings and the assertions catch it.
"""

from itertools import permutations

from reagent.core.models import Assessment, Molecule, Route
from reagent.optimize.aggregate import route_signature
from reagent.optimize.pareto import compromise_route, pareto_front

TIED_SCORES = {"feasibility": 0.6, "safety": 0.6, "cost": 0.6}


def _route(leaf: str, scores: dict[str, float] | None = None) -> Route:
    """A route distinguishable only by the building block it buys."""
    return Route(
        target="T",
        solved=True,
        leaves=[Molecule(smiles=leaf, in_stock=True)],
        assessments=[
            Assessment(objective=o, score=s, rationale="")
            for o, s in (scores or TIED_SCORES).items()
        ],
    )


def test_identically_scored_routes_resolve_to_the_same_one_in_every_order():
    """The assertion a position-dependent implementation fails.

    Four routes with the same scores and different leaves. Whichever the rule
    picks, it must pick that same one from all 24 orderings.
    """
    routes = [_route(s) for s in ("CCO", "CCN", "CCC", "CCS")]

    chosen = {route_signature(compromise_route(list(order))) for order in permutations(routes)}

    assert len(chosen) == 1, f"order changed the answer: {len(chosen)} distinct winners"


def test_the_tie_is_settled_on_route_identity_not_arrival():
    """And the winner is specifically the smallest signature, not the first
    element, so the rule is reproducible across processes rather than merely
    self-consistent within one."""
    routes = [_route(s) for s in ("CCS", "CCO", "CCN")]
    expected = min(route_signature(r) for r in routes)

    for order in permutations(routes):
        assert route_signature(compromise_route(list(order))) == expected


def test_a_genuine_winner_beats_the_tie_break():
    """Identity only decides when the scores do not. A route closer to the
    ideal point must win from any position."""
    best = _route("CCO", {"feasibility": 0.9, "safety": 0.9, "cost": 0.9})
    rest = [_route("CCN"), _route("CCC")]

    for order in permutations([best, *rest]):
        assert compromise_route(list(order)) is best


def test_the_choice_always_comes_from_the_pareto_front():
    routes = [
        _route("CCO", {"feasibility": 0.9, "safety": 0.2, "cost": 0.5}),
        _route("CCN", {"feasibility": 0.2, "safety": 0.9, "cost": 0.5}),
        _route("CCC", {"feasibility": 0.1, "safety": 0.1, "cost": 0.1}),
    ]
    front = {route_signature(r) for r in pareto_front(routes)}

    assert route_signature(compromise_route(routes)) in front


def test_empty_and_single_inputs():
    assert compromise_route([]) is None
    only = _route("CCO")
    assert compromise_route([only]) is only
