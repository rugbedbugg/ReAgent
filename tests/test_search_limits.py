"""Search-budget wiring: the wall clock must not silently cap the iteration budget.

AiZynthFinder's search loop runs ``while time_past < time_limit and i <=
iteration_limit``, so raising ``iterations`` alone measures the 120 s default
timeout rather than the budget. These tests use a stub finder, so no ONNX model
or stock file is needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reagent.singlestep.aizynth import AiZynthBackend


class _StubStrategy(SimpleNamespace):
    """One expansion/filter strategy, carrying AiZynthFinder's cutoff defaults."""

    def __init__(self):
        super().__init__(cutoff_number=50, cutoff_cumulative=0.995)

    def feasibility(self, _rxn):
        return (True, 1.0)


class _StubPolicy:
    """Selectable and subscriptable, like AiZynthFinder's policy containers.

    Strategies are held per key so attributes set on them survive lookup, which
    is how the real collection behaves and what the cutoff wiring relies on.
    """

    def __init__(self):
        self.selected = None
        self._strategies: dict = {}

    def select(self, value, *_args, **_kwargs):
        self.selected = value

    def __getitem__(self, key):
        return self._strategies.setdefault(key, _StubStrategy())


class _StubFinder:
    def __init__(self, **_kwargs):
        self.config = SimpleNamespace(search=SimpleNamespace(iteration_limit=100, time_limit=120))
        self.stock = SimpleNamespace(load=lambda *a, **k: None, select=lambda *a, **k: None)
        self.expansion_policy = _StubPolicy()
        self.filter_policy = _StubPolicy()
        self.search_stats = {}
        self.routes = SimpleNamespace(dicts=[])

    def tree_search(self):
        return 0.0

    def build_routes(self):
        return []


@pytest.fixture
def backend_factory():
    def make(**kwargs):
        with patch("aizynthfinder.aizynthfinder.AiZynthFinder", _StubFinder):
            return AiZynthBackend("unused.yml", **kwargs)

    return make


def test_time_limit_is_applied_alongside_iterations(backend_factory):
    backend = backend_factory(iterations=500, time_limit=1800)
    assert backend.search_iteration_limit == 500
    assert backend.search_time_limit == 1800


def test_iterations_alone_leaves_the_default_clock_in_place(backend_factory):
    # The regression this guards: 500 iterations under a 120 s clock never runs
    # 500 iterations, so a budget comparison measures the timeout instead.
    backend = backend_factory(iterations=500)
    assert backend.search_iteration_limit == 500
    assert backend.search_time_limit == 120


def test_reports_when_the_clock_stopped_the_search(backend_factory):
    backend = backend_factory(iterations=500, time_limit=120)
    backend._finder.search_stats = {"iterations": 87, "returned_first": False}
    backend.last_search_stats = dict(backend._finder.search_stats)
    assert backend.search_hit_time_limit is True


def test_does_not_report_a_cap_when_the_budget_was_spent(backend_factory):
    backend = backend_factory(iterations=500, time_limit=1800)
    backend.last_search_stats = {"iterations": 500, "returned_first": False}
    assert backend.search_hit_time_limit is False


def test_early_solve_is_not_reported_as_a_cap(backend_factory):
    backend = backend_factory(iterations=500, time_limit=1800)
    backend.last_search_stats = {"iterations": 12, "returned_first": True}
    assert backend.search_hit_time_limit is False


def test_single_expansion_policy_is_selected_by_name(backend_factory):
    backend = backend_factory()
    assert backend.expansion_keys == ["uspto"]
    assert backend._finder.expansion_policy.selected == ["uspto"]


def test_multiple_expansion_policies_are_selected_together(backend_factory):
    # The union of two policies' suggestions, not one replacing the other.
    backend = backend_factory(expansion=["uspto", "ringbreaker"])
    assert backend.expansion_keys == ["uspto", "ringbreaker"]
    assert backend._finder.expansion_policy.selected == ["uspto", "ringbreaker"]


def test_filter_policy_keys_off_the_primary_expansion(backend_factory):
    # The filter model is trained with the primary policy, so an ensemble must
    # not try to look it up under the secondary policy's name.
    backend = backend_factory(expansion=["uspto", "ringbreaker"])
    assert backend._finder.filter_policy.selected == "uspto"


def test_mcts_stays_a_bare_name(backend_factory):
    # AiZynthFinder special-cases "mcts"; handing it a class path would break it.
    backend = backend_factory()
    assert backend._finder.config.search.algorithm == "mcts"


def test_alternative_algorithms_resolve_to_class_paths(backend_factory):
    backend = backend_factory(algorithm="retrostar")
    assert backend._finder.config.search.algorithm == (
        "aizynthfinder.search.retrostar.search_tree.SearchTree"
    )


def test_unknown_algorithm_is_rejected(backend_factory):
    with pytest.raises(ValueError, match="Unknown search algorithm"):
        backend_factory(algorithm="nope")


def test_cutoff_number_defaults_are_left_alone(backend_factory):
    backend = backend_factory()
    assert backend._finder.expansion_policy["uspto"].cutoff_number == 50


def test_cutoff_number_is_applied_to_every_selected_policy(backend_factory):
    # The measured cap: every policy returns exactly 50 templates per molecule,
    # so the next-best disconnections never reach the search.
    backend = backend_factory(expansion=["uspto", "ringbreaker"], cutoff_number=150)
    for key in ("uspto", "ringbreaker"):
        assert backend._finder.expansion_policy[key].cutoff_number == 150


def test_cutoff_cumulative_is_applied_when_given(backend_factory):
    backend = backend_factory(cutoff_cumulative=0.999)
    assert backend._finder.expansion_policy["uspto"].cutoff_cumulative == 0.999
