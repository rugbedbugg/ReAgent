"""Modes: a weight profile plus, for `build`, a hard constraint.

The weights alone were measured to be too weak to express the intent. At 0.30
on `construction` the build profile changes 13 picks against the default's 11,
because a soft preference cannot outvote a route that is simply shorter. So a
mode is not only weights: `build` also caps how much of the target a single
purchased leaf may be, enforced in the stock layer where the search can see it.
"""

import pytest

from reagent.optimize.aggregate import (
    DEFAULT_WEIGHTS,
    MODES,
    WEIGHT_PROFILES,
    mode_leaf_fraction,
    mode_weights,
)


def test_every_mode_names_a_real_profile():
    for name, spec in MODES.items():
        assert spec["weights"] in WEIGHT_PROFILES, f"{name} names a missing profile"


def test_every_profile_is_a_probability_vector():
    for name, weights in WEIGHT_PROFILES.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, f"{name} does not sum to 1"
        assert set(weights) == set(DEFAULT_WEIGHTS), f"{name} has the wrong objectives"


def test_only_build_constrains_what_may_be_bought():
    """`source` exists to buy advanced intermediates, so constraining it would
    defeat the mode. `balanced` keeps the shipped behaviour."""
    assert mode_leaf_fraction("build") == 0.6
    assert mode_leaf_fraction("source") is None
    assert mode_leaf_fraction("balanced") is None


def test_the_modes_actually_disagree_about_buying():
    """If build and source weighted `construction` alike, the modes would be
    labels rather than behaviour."""
    build = mode_weights("build")
    source = mode_weights("source")
    assert build["construction"] > source["construction"] * 10
    assert source["cost"] > build["cost"]


def test_balanced_is_the_shipped_default():
    assert mode_weights("balanced") == dict(DEFAULT_WEIGHTS)


def test_an_unknown_mode_names_the_valid_ones():
    for fn in (mode_weights, mode_leaf_fraction):
        with pytest.raises(ValueError, match="build"):
            fn("nonsense")
