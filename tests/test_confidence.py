"""Confidence-gate tests."""

from reagent.core.models import Molecule, Reaction, Route
from reagent.optimize.confidence import is_trustworthy, route_confidence


def _route(prob: float) -> Route:
    return Route(
        target="CCN",
        reactions=[Reaction(product="CCN", precursors=["CCO"],
                            metadata={"policy_probability": prob, "library_occurence": 10})],
        leaves=[Molecule(smiles="CCO", in_stock=True)],
        solved=True,
    )


def test_confidence_bands():
    assert route_confidence(_route(0.80))[0] == "high"
    assert route_confidence(_route(0.30))[0] == "moderate"
    assert route_confidence(_route(0.10))[0] == "low"
    assert route_confidence(_route(0.00))[0] == "very-low"


def test_trustworthiness_gate():
    assert is_trustworthy(_route(0.4))
    assert not is_trustworthy(_route(0.01))  # a feas 0.01 route is flagged, not trusted
