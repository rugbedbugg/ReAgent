"""Fast unit tests that don't need the downloaded model or an API key."""

from reagent.core.chem import canonical, is_valid
from reagent.core.models import Assessment, Molecule, Reaction, Route


def test_canonical_roundtrip():
    assert canonical("OCC") == "CCO"
    assert canonical("not-a-molecule") is None
    assert is_valid("c1ccccc1")
    assert not is_valid("XYZ123")


def test_route_model():
    route = Route(
        target="CC(=O)Oc1ccccc1C(=O)O",
        reactions=[Reaction(product="CC(=O)Oc1ccccc1C(=O)O", precursors=["CC(=O)O", "Oc1ccccc1C(=O)O"])],
        leaves=[Molecule(smiles="CC(=O)O", in_stock=True)],
        solved=True,
    )
    assert route.num_steps == 1
    assert route.reactions[0].precursors == ["CC(=O)O", "Oc1ccccc1C(=O)O"]


def test_assessment_bounds():
    a = Assessment(objective="safety", score=0.9, rationale="no hazardous reagents")
    assert 0.0 <= a.score <= 1.0
