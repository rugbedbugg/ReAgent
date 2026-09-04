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


class _StubPolicy:
    """Selectable and subscriptable, like AiZynthFinder's policy containers."""

    def select(self, *_args, **_kwargs):
        return None

    def __getitem__(self, _key):
        return SimpleNamespace(feasibility=lambda _rxn: (True, 1.0))


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
