"""Phase 6 tests: rationale narrative, episodic memory, weight tuning."""

import math

from reagent.adaptive.memory import Episode, EpisodicMemory
from reagent.adaptive.weights import update_from_preference
from reagent.agents.rationale import build_rationale
from reagent.core.models import Assessment, Route


def _scored(target: str, vector: dict[str, float], weighted: float) -> Route:
    route = Route(
        target=target,
        assessments=[Assessment(objective=o, score=s, rationale=f"{o} note") for o, s in vector.items()],
    )
    route.scores = {"weighted": weighted, "vector": vector}
    return route


def test_build_rationale_names_winner_and_reasons():
    win = _scored("t", {"safety": 1.0, "feasibility": 0.9, "cost": 0.8}, 0.92)
    lose = _scored("t", {"safety": 0.1, "feasibility": 0.2, "cost": 0.8}, 0.35)
    numbers = {id(win): 1, id(lose): 2}
    text = build_rationale([win, lose], numbers)
    assert "Recommended: Route 1" in text
    assert "Route 2 passed over" in text
    assert "safety" in text  # the decisive shortfall is surfaced


def test_memory_roundtrip_and_similarity(tmp_path):
    mem = EpisodicMemory(path=tmp_path / "ep.jsonl")
    mem.append(Episode(target="CCO", recommended=1))
    mem.append(Episode(target="c1ccccc1", recommended=2))
    assert len(mem.all()) == 2
    assert mem.last_for("CCO").recommended == 1
    similar = mem.find_similar("CCO", k=2)
    assert similar[0][0].target == "CCO"  # identical molecule ranks first
    assert similar[0][1] == 1.0


def test_update_from_preference_shifts_and_normalizes():
    weights = {"safety": 0.5, "cost": 0.5}
    # Preferred route is much safer, equal on cost.
    preferred = {"safety": 1.0, "cost": 0.5}
    others = [{"safety": 0.1, "cost": 0.5}]
    new = update_from_preference(weights, preferred, others)
    assert new["safety"] > weights["safety"]  # safety gained weight
    assert new["cost"] < weights["cost"]
    assert math.isclose(sum(new.values()), 1.0)


def test_update_from_preference_is_a_no_op_when_nothing_distinguishes():
    """A route that ties its rivals on every objective carries no signal, so
    the weights must come back untouched rather than drifting on noise."""
    weights = {"safety": 0.5, "cost": 0.5}
    tied = {"safety": 0.9, "cost": 0.5}

    new = update_from_preference(weights, tied, [dict(tied)])

    assert new == weights


def test_update_from_preference_is_a_no_op_without_competitors():
    """With nothing to compare against there is no gap to learn from. The
    preferred route is treated as its own rival, which keeps the step at zero
    instead of dividing by an empty list."""
    weights = {"safety": 0.5, "cost": 0.5}

    new = update_from_preference(weights, {"safety": 0.9}, [])

    assert new == weights


def test_objectives_absent_from_the_preferred_route_are_scored_as_zero():
    """`preferred` need not mention every objective. A missing one reads as 0.0
    and therefore as a shortfall against rivals that do score on it, so it
    loses weight rather than being skipped."""
    weights = {"safety": 0.5, "cost": 0.5}

    new = update_from_preference(weights, {"safety": 0.9}, [{"safety": 0.9, "cost": 0.5}])

    assert new["cost"] < weights["cost"]
    assert new["safety"] > weights["safety"]
    assert math.isclose(sum(new.values()), 1.0)
