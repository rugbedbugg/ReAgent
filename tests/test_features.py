"""Feature-layer tests: deterministic, no model or API key needed."""

from reagent.core.models import Molecule, Reaction, Route
from reagent.features import descriptors as d
from reagent.features.extract import compute_features


def _aspirin_route() -> Route:
    tree = {
        "type": "mol",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "children": [
            {
                "type": "reaction",
                "smiles": "rsmi",
                "children": [
                    {"type": "mol", "smiles": "CC(=O)OC(C)=O"},
                    {"type": "mol", "smiles": "O=C(O)c1ccccc1O"},
                ],
            }
        ],
    }
    return Route(
        target="CC(=O)Oc1ccccc1C(=O)O",
        reactions=[
            Reaction(
                product="CC(=O)Oc1ccccc1C(=O)O",
                precursors=["CC(=O)OC(C)=O", "O=C(O)c1ccccc1O"],
                metadata={
                    "policy_probability": 0.73,
                    "library_occurence": 1196,
                    "classification": "0.0 Unrecognized",
                },
            )
        ],
        leaves=[
            Molecule(smiles="CC(=O)OC(C)=O", in_stock=True),
            Molecule(smiles="O=C(O)c1ccccc1O", in_stock=True),
        ],
        solved=True,
        tree=tree,
    )


def test_descriptors():
    assert d.heavy_atoms("CC(=O)O") == 4
    assert d.mol_weight("O") > 17.9
    assert "acyl_halide" in d.hazard_groups("CC(=O)Cl")
    assert d.hazard_groups("CCO") == []


def test_compute_features_blocks():
    route = compute_features(_aspirin_route())
    assert set(route.features) == {
        "efficiency",
        "availability",
        "cost",
        "safety",
        "sustainability",
        "feasibility",
    }


def test_availability_and_feasibility():
    f = compute_features(_aspirin_route()).features
    assert f["availability"]["in_stock_fraction"] == 1.0
    assert f["availability"]["not_in_stock"] == []
    assert f["feasibility"]["min_policy_probability"] == 0.73
    assert f["feasibility"]["min_library_occurence"] == 1196
    assert f["efficiency"]["convergence"] == 1.0  # single linear step


def test_safety_flags_acyl_chloride():
    route = _aspirin_route()
    route.reactions[0].precursors = ["CC(=O)Cl", "O=C(O)c1ccccc1O"]
    f = compute_features(route).features
    assert "acyl_halide" in f["safety"]["distinct_hazards"]
