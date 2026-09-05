"""Objective-aware molecule costs: the hook that lets objectives steer the search."""

from reagent.search.cost import AccessibilityCost, HazardCost


class _Mol:
    """Stands in for AiZynthFinder's Molecule, which exposes ``smiles``."""

    def __init__(self, smiles):
        self.smiles = smiles


def test_a_hazardous_molecule_costs_more_than_a_benign_one():
    cost = HazardCost(weight=1.0)
    assert cost.calculate(_Mol("CI")) > cost.calculate(_Mol("CCO"))


def test_a_benign_molecule_is_free():
    assert HazardCost().calculate(_Mol("CCO")) == 0.0


def test_zero_weight_reproduces_the_default_behaviour():
    """The A/B control: weight 0 must be indistinguishable from ZeroMoleculeCost."""
    cost = HazardCost(weight=0.0)
    assert cost.calculate(_Mol("CI")) == 0.0
    assert cost.calculate(_Mol("CCO")) == 0.0


def test_hazard_cost_saturates():
    """Past a few alerts a compound needs handling precautions either way."""
    cost = HazardCost(weight=1.0, cap=2)
    assert cost.calculate(_Mol("CI")) <= 2.0


def test_an_unparseable_molecule_costs_nothing():
    """This runs inside the search loop: never raise, and never guess hazard."""
    assert HazardCost().calculate(_Mol("not-a-smiles")) == 0.0
    assert AccessibilityCost().calculate(_Mol("not-a-smiles")) == 0.0


def test_an_easy_building_block_is_free_but_a_complex_one_is_not():
    cost = AccessibilityCost(weight=1.0, free=3.0)
    assert cost.calculate(_Mol("CCO")) == 0.0
    # A polycyclic natural-product-like scaffold scores well above the free tier.
    complex_mol = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4CC(O)CC[C@]4(C)[C@H]3CC[C@]12C"
    assert cost.calculate(_Mol(complex_mol)) > 0.0


def test_costs_are_reported_for_the_search_log():
    assert "hazard" in repr(HazardCost())
    assert "accessibility" in repr(AccessibilityCost())


def test_steer_spec_parses_a_bare_objective_name():
    from reagent.cli import _steer_config

    assert _steer_config("hazard") == {"cost": "reagent.search.cost.HazardCost"}


def test_steer_spec_parses_an_explicit_weight():
    from reagent.cli import _steer_config

    config = _steer_config("accessibility:2.5")
    assert config["cost"].endswith("AccessibilityCost")
    assert config["weight"] == 2.5


def test_a_zero_weight_is_accepted_as_the_control_arm():
    from reagent.cli import _steer_config

    assert _steer_config("hazard:0")["weight"] == 0.0


def test_no_steering_leaves_the_search_untouched():
    from reagent.cli import _steer_config

    assert _steer_config(None) is None
    assert _steer_config("") is None


def test_an_unknown_objective_is_rejected():
    import click
    import pytest

    from reagent.cli import _steer_config

    with pytest.raises(click.ClickException):
        _steer_config("greenness")


def test_a_non_numeric_weight_is_rejected():
    import click
    import pytest

    from reagent.cli import _steer_config

    with pytest.raises(click.ClickException):
        _steer_config("hazard:lots")
