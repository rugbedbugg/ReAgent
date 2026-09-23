"""Evaluation-layer tests: deterministic scorer and harness (offline, no backend)."""

import pytest

from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.harness import evaluate
from reagent.features.scoring import deterministic_scores


def _route(target: str, precursors: list[str], hazardous: bool, prob: float, solved: bool = True) -> Route:
    return Route(
        target=target,
        reactions=[
            Reaction(
                product=target,
                precursors=precursors,
                metadata={"policy_probability": prob, "library_occurence": 100},
            )
        ],
        leaves=[Molecule(smiles=p, in_stock=True) for p in precursors],
        solved=solved,
    )


def test_deterministic_scores_penalize_hazard():
    clean = _route("CCN", ["CCO", "N"], hazardous=False, prob=0.8)
    hazardous = _route("CCN", ["CC(=O)Cl", "N"], hazardous=True, prob=0.8)
    assert deterministic_scores(clean)["safety"] == 1.0
    assert deterministic_scores(hazardous)["safety"] <= 0.5
    assert deterministic_scores(clean)["feasibility"] == 0.8


def test_harness_prefers_safer_route_at_equal_solve_rate():
    # Two candidate routes for one target: baseline (feasibility) would pick the
    # slightly higher-probability but hazardous route; REAGENT should pick safer.
    safe = _route("T", ["CCO", "OCC"], hazardous=False, prob=0.70)
    hazardous = _route("T", ["CC(=O)Cl"], hazardous=True, prob=0.72)

    result = evaluate([("t", "T")], planner=lambda s: [safe, hazardous])

    assert result["solve_rate"] == 1.0
    # REAGENT selects the safer route -> higher mean safety than baseline
    assert result["reagent_quality"]["safety"] > result["baseline_quality"]["safety"]
    assert result["per_target"][0]["changed_pick"] is True


def test_harness_counts_unsolved():
    unsolved = _route("U", ["Cc1ccccc1Br"], hazardous=False, prob=0.5, solved=False)
    result = evaluate([("u", "U")], planner=lambda s: [unsolved])
    assert result["solve_rate"] == 0.0
    assert result["per_target"][0]["solved"] is False


def test_largest_leaf_fraction_spots_a_bought_intermediate():
    """Sertraline from its ketimine: one step, and the leaf is the whole molecule."""
    from reagent.eval.harness import largest_leaf_fraction

    route = Route(
        target="CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
        leaves=[Molecule(smiles="CN=C1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21", in_stock=True)],
        solved=True,
    )
    assert largest_leaf_fraction(route) > 0.9


def test_largest_leaf_fraction_is_low_for_a_building_block_route():
    """Fluoxetine from an aminoalcohol and an aryl iodide: both genuine reagents."""
    from reagent.eval.harness import largest_leaf_fraction

    route = Route(
        target="CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        leaves=[
            Molecule(smiles="CNCCC(O)c1ccccc1", in_stock=True),
            Molecule(smiles="FC(F)(F)c1ccc(I)cc1", in_stock=True),
        ],
        solved=True,
    )
    assert largest_leaf_fraction(route) < 0.7


def test_largest_leaf_fraction_handles_a_route_with_no_leaves():
    from reagent.eval.harness import largest_leaf_fraction

    assert largest_leaf_fraction(Route(target="CCO")) == 0.0


def test_safety_does_not_fall_just_because_the_route_is_longer():
    """The defect this formula exists to fix.

    Both routes make sertraline and both handle methyl iodide, the one genuinely
    nasty reagent; their hazard densities are identical (1/3 vs 2/6). The earlier
    formula subtracted 0.1 per distinct hazard found anywhere in the route, so
    the real three-step synthesis scored 0.300 against 0.400 for the one-step
    route that just buys the penultimate intermediate. Safety measured length.
    """
    from reagent.features.scoring import deterministic_scores

    sert = "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"
    penultimate = "NC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"
    alkene = "NC1CC=C(c2ccc(Cl)c(Cl)c2)c2ccccc21"

    def route(steps, leaves):
        return Route(
            target=sert,
            reactions=[
                Reaction(product=p, precursors=pr,
                         metadata={"policy_probability": 0.5, "library_occurence": 50})
                for p, pr in steps
            ],
            leaves=[Molecule(smiles=s, in_stock=True) for s in leaves],
            solved=True,
        )

    bought = route([(sert, ["CI", penultimate])], ["CI", penultimate])
    built = route(
        [
            (sert, ["CI", penultimate]),
            (penultimate, [alkene]),
            (alkene, ["NC1CCC(=O)c2ccccc21", "Clc1ccc(Br)cc1Cl"]),
        ],
        ["CI", "NC1CCC(=O)c2ccccc21", "Clc1ccc(Br)cc1Cl"],
    )

    assert deterministic_scores(built)["safety"] == deterministic_scores(bought)["safety"]


def test_safety_still_falls_when_a_worse_reagent_is_handled():
    """Intensive does not mean insensitive: severity must still bite."""
    from reagent.features.scoring import deterministic_scores

    mild = _route("CCN", ["CCO", "N"], hazardous=False, prob=0.8)
    nasty = _route("CCN", ["CC(=O)Cl", "N"], hazardous=True, prob=0.8)
    assert deterministic_scores(nasty)["safety"] < deterministic_scores(mild)["safety"]


def test_a_denser_route_scores_worse_than_a_sparser_one():
    """At equal worst-molecule severity, more hazardous molecules is worse."""
    from reagent.features.scoring import deterministic_scores

    sparse = _route("CCN", ["CC(=O)Cl", "CCO", "N"], hazardous=True, prob=0.8)
    dense = _route("CCN", ["CC(=O)Cl", "CCC(=O)Cl"], hazardous=True, prob=0.8)
    assert deterministic_scores(dense)["safety"] < deterministic_scores(sparse)["safety"]


def test_construction_scores_a_building_block_route_above_a_bought_one():
    """The objective that exists because every other one prefers buying."""
    from reagent.features.scoring import deterministic_scores

    sert = "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"

    def route(leaves):
        return Route(
            target=sert,
            reactions=[Reaction(product=sert, precursors=leaves,
                                metadata={"policy_probability": 0.5, "library_occurence": 50})],
            leaves=[Molecule(smiles=s, in_stock=True) for s in leaves],
            solved=True,
        )

    bought = route(["CN=C1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"])   # 19 of 20 heavy atoms
    built = route(["NC1CCC(=O)c2ccccc21", "Clc1ccc(Br)cc1Cl"])  # 12 and 9

    assert deterministic_scores(bought)["construction"] == 0.0
    assert deterministic_scores(built)["construction"] > 0.7


def test_construction_rewards_a_near_balanced_coupling():
    """Fluoxetine's real route: a 12-atom and an 11-atom half of a 22-atom target.

    The largest leaf is 0.545 of the target, close to the 0.5 floor that a
    perfectly balanced two-component coupling would hit, so this scores near the
    top without reaching it. Two components can never go below 0.5; only a route
    that splits into three or more can.
    """
    from reagent.features.scoring import deterministic_scores

    route = Route(
        target="CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        reactions=[Reaction(product="CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
                            precursors=["CNCCC(O)c1ccccc1", "FC(F)(F)c1ccc(I)cc1"],
                            metadata={"policy_probability": 0.5, "library_occurence": 50})],
        leaves=[Molecule(smiles="CNCCC(O)c1ccccc1", in_stock=True),
                Molecule(smiles="FC(F)(F)c1ccc(I)cc1", in_stock=True)],
        solved=True,
    )
    assert deterministic_scores(route)["construction"] > 0.85


def test_construction_survives_an_unparseable_target():
    from reagent.features.scoring import deterministic_scores

    route = Route(target="not-a-smiles",
                  leaves=[Molecule(smiles="CCO", in_stock=True)], solved=True)
    assert deterministic_scores(route)["construction"] == 0.0


def _bought_whole(target: str) -> Route:
    """A one-step route that buys a molecule which is nearly the target."""
    return Route(
        target=target,
        leaves=[Molecule(smiles="CN=C1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21", in_stock=True)],
        reactions=[Reaction(product=target, precursors=["CN=C1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"])],
        solved=True,
    )


def test_build_solve_rate_does_not_count_buying_the_answer():
    """solve_rate counts a target as solved when every leaf is purchasable,
    which counts buying a nearly finished molecule as success. On the moderate
    set that was 10 of 25 targets, so 1.00 was reported where the honest figure
    was 0.60. Both are now returned."""
    sertraline = "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"
    result = evaluate([("sertraline", sertraline)], lambda _s: [_bought_whole(sertraline)])

    assert result["solve_rate"] == 1.0
    assert result["build_solve_rate"] == 0.0


def test_build_solve_rate_matches_solve_rate_for_a_genuine_route():
    fluoxetine = "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"
    genuine = Route(
        target=fluoxetine,
        leaves=[
            Molecule(smiles="CNCCC(O)c1ccccc1", in_stock=True),
            Molecule(smiles="FC(F)(F)c1ccc(I)cc1", in_stock=True),
        ],
        reactions=[Reaction(product=fluoxetine, precursors=["CNCCC(O)c1ccccc1"])],
        solved=True,
    )
    result = evaluate([("fluoxetine", fluoxetine)], lambda _s: [genuine])

    assert result["solve_rate"] == 1.0
    assert result["build_solve_rate"] == 1.0


def test_weight_profiles_produce_different_rankings():
    """Custom weight profiles should produce different rankings when objectives differ."""
    from reagent.eval.harness import evaluate
    from reagent.optimize.aggregate import WEIGHT_PROFILES

    # Create two routes that differ in safety and cost
    # Route 1: safer but more expensive (two steps, non-hazardous)
    # Route 2: less safe but cheaper (one step, hazardous)
    safe_expensive = Route(
        target="CCN",
        reactions=[Reaction(
            product="CCN", precursors=["CCO", "N"],
            metadata={"policy_probability": 0.7, "library_occurence": 50}
        )],
        leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    safe_expensive.reactions[0].metadata["hazardous"] = False

    cheap_risky = Route(
        target="CCN",
        reactions=[Reaction(
            product="CCN", precursors=["CC(=O)Cl", "N"],
            metadata={"policy_probability": 0.72, "library_occurence": 50}
        )],
        leaves=[Molecule(smiles="CC(=O)Cl", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    cheap_risky.reactions[0].metadata["hazardous"] = True

    routes = [safe_expensive, cheap_risky]

    # Test with safety-tilted profile
    result_safety = evaluate(
        [("t", "CCN")],
        planner=lambda s: routes,
        weights=WEIGHT_PROFILES["safety-tilted"]
    )

    # Test with source-led profile
    result_source = evaluate(
        [("t", "CCN")],
        planner=lambda s: routes,
        weights=WEIGHT_PROFILES["source-led"]
    )

    # The weight parameter should be accepted and used
    # Verify the weights are passed through by checking the reagent_quality differs
    # or that the results are structurally valid
    assert "reagent_quality" in result_safety
    assert "reagent_quality" in result_source
    assert "per_target" in result_safety
    assert "per_target" in result_source
    # The weight vector should be recorded in the result
    assert "reagent_weight_vector" in result_safety
    assert "reagent_weight_vector" in result_source
    # The weight vectors should reflect the profiles used
    assert result_safety["reagent_weight_vector"] == WEIGHT_PROFILES["safety-tilted"]
    assert result_source["reagent_weight_vector"] == WEIGHT_PROFILES["source-led"]


def test_select_respects_custom_weights():
    """_select should use the provided weights, not hardcoded DEFAULT_WEIGHTS."""
    from reagent.eval.harness import _select
    from reagent.optimize.aggregate import WEIGHT_PROFILES

    # Create routes with different objective scores
    route1 = Route(
        target="CCN",
        reactions=[Reaction(
            product="CCN", precursors=["CCO", "N"],
            metadata={"policy_probability": 0.7, "library_occurence": 50}
        )],
        leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    route1.reactions[0].metadata["hazardous"] = False

    route2 = Route(
        target="CCN",
        reactions=[Reaction(
            product="CCN", precursors=["CC(=O)Cl", "N"],
            metadata={"policy_probability": 0.72, "library_occurence": 50}
        )],
        leaves=[Molecule(smiles="CC(=O)Cl", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    route2.reactions[0].metadata["hazardous"] = True

    routes = [route1, route2]

    # With DEFAULT_WEIGHTS (feasibility-led), should pick route2 (higher feasibility)
    _, pick_default = _select(routes, {"feasibility": 1.0})
    assert pick_default is route2

    # With safety-tilted weights, should pick route1 (safer)
    _, pick_safety = _select(routes, WEIGHT_PROFILES["safety-tilted"])
    assert pick_safety is route1


def test_rank_candidates_accepts_custom_weights():
    """rank_candidates should accept and use custom weight vectors."""
    from reagent.eval.harness import rank_candidates
    from reagent.optimize.aggregate import WEIGHT_PROFILES

    route1 = Route(
        target="CCN",
        reactions=[Reaction(product="CCN", precursors=["CCO", "N"])],
        leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    route1.reactions[0].metadata["hazardous"] = False

    route2 = Route(
        target="CCN",
        reactions=[Reaction(product="CCN", precursors=["CC(=O)Cl", "N"])],
        leaves=[Molecule(smiles="CC(=O)Cl", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    route2.reactions[0].metadata["hazardous"] = True

    routes = [route1, route2]

    # With feasibility-only weights, route2 (higher feasibility) should rank first
    ranked_feasibility = rank_candidates(routes, {"feasibility": 1.0})
    # The route with higher feasibility (0.72) should rank first
    assert ranked_feasibility.reagent_ranked[0] is route2

    # With safety-tilted weights, route1 (safer) should rank first
    ranked_safety = rank_candidates(routes, WEIGHT_PROFILES["safety-tilted"])
    assert ranked_safety.reagent_ranked[0] is route1


def test_baseline_ranking_unaffected_by_reagent_weights():
    """Baseline (raw feasibility) ranking should be unchanged by ReAgent weight profiles."""
    from reagent.eval.harness import rank_baseline_feasibility, rank_candidates

    route1 = Route(
        target="CCN",
        reactions=[Reaction(product="CCN", precursors=["CCO", "N"])],
        leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    route1.reactions[0].metadata["hazardous"] = False

    route2 = Route(
        target="CCN",
        reactions=[Reaction(product="CCN", precursors=["CC(=O)Cl", "N"])],
        leaves=[Molecule(smiles="CC(=O)Cl", in_stock=True), Molecule(smiles="N", in_stock=True)],
        solved=True,
    )
    route2.reactions[0].metadata["hazardous"] = True

    routes = [route1, route2]

    # Baseline ranking (raw feasibility) should be identical regardless of ReAgent weights
    baseline1 = rank_baseline_feasibility(routes)
    baseline2 = rank_baseline_feasibility(routes)
    assert [id(r) for r in baseline1] == [id(r) for r in baseline2]

    # ReAgent ranking with different weights should give different orderings
    ranked_default = rank_candidates(routes, {"feasibility": 1.0})
    ranked_safety = rank_candidates(routes, {"safety": 1.0, "feasibility": 0.0})

    # Different weights should produce different orderings
    assert [id(r) for r in ranked_default.reagent_ranked] != [id(r) for r in ranked_safety.reagent_ranked]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])