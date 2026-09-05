"""The feedback loop measured as behaviour over a sequence, not a single step."""

import random

from reagent.eval.adaptive_check import regret, simulate

SAFETY_LOVER = {"safety": 0.7, "cost": 0.15, "feasibility": 0.15}
COST_LOVER = {"cost": 0.7, "safety": 0.15, "feasibility": 0.15}


def _candidates(seed: int, n: int = 6) -> list[dict[str, float]]:
    """A candidate set where safety and cost genuinely trade off."""
    rng = random.Random(seed)
    return [
        {
            "safety": rng.random(),
            "cost": rng.random(),
            "feasibility": 0.4 + rng.random() * 0.2,
        }
        for _ in range(n)
    ]


def test_regret_is_zero_for_the_best_route():
    vectors = [{"safety": 1.0}, {"safety": 0.0}]
    assert regret(vectors, chosen=0, hidden={"safety": 1.0}) == 0.0


def test_regret_is_one_for_the_worst_route():
    vectors = [{"safety": 1.0}, {"safety": 0.0}]
    assert regret(vectors, chosen=1, hidden={"safety": 1.0}) == 1.0


def test_regret_is_zero_when_every_candidate_is_equivalent():
    """Nothing to get wrong should not count against the learner."""
    vectors = [{"safety": 0.5}, {"safety": 0.5}]
    assert regret(vectors, chosen=1, hidden={"safety": 1.0}) == 0.0


def test_learning_lowers_regret_against_a_hidden_preference():
    result = simulate([_candidates(i) for i in range(40)], hidden=SAFETY_LOVER)
    assert result["rounds"] > 20
    assert result["regret_second_half"] < result["regret_first_half"]


def test_learning_moves_weight_onto_the_preferred_objective():
    result = simulate([_candidates(i) for i in range(40)], hidden=SAFETY_LOVER)
    assert result["learned_weights"]["safety"] > result["start_weights"]["safety"]


def test_the_loop_follows_whichever_preference_it_is_shown():
    """It must not simply always favour safety -- the same run with the opposite
    hidden preference has to move weight the other way."""
    targets = [_candidates(i) for i in range(40)]
    safety = simulate(targets, hidden=SAFETY_LOVER)["learned_weights"]
    cost = simulate(targets, hidden=COST_LOVER)["learned_weights"]

    assert safety["safety"] > cost["safety"]
    assert cost["cost"] > safety["cost"]


def test_a_single_candidate_teaches_nothing():
    result = simulate([[{"safety": 0.5}]], hidden=SAFETY_LOVER)
    assert result["rounds"] == 0


def test_tied_candidates_teach_nothing():
    result = simulate([[{"safety": 0.5}, {"safety": 0.5}]], hidden=SAFETY_LOVER)
    assert result["rounds"] == 0


def test_the_simulation_does_not_depend_on_candidate_order():
    """Same tie-break bug as ranking: max() keeps the first of equal scores."""
    targets = [_candidates(i) for i in range(12)]
    shuffled = [list(reversed(c)) for c in targets]

    assert simulate(targets, hidden=SAFETY_LOVER)["learned_weights"] == \
           simulate(shuffled, hidden=SAFETY_LOVER)["learned_weights"]


def test_a_single_learning_round_reports_without_crashing():
    """Found by the order test: trace[1:] is empty at one round, and mean() raised."""
    one = [[{"safety": 0.9, "cost": 0.1}, {"safety": 0.1, "cost": 0.9}]]
    result = simulate(one, hidden=SAFETY_LOVER)

    assert result["rounds"] == 1
    assert "regret_second_half" not in result
