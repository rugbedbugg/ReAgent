"""Pooling several searches over one single-step model.

Selection can only choose among the routes it is handed, so once solve-rate
saturates the bottleneck is candidate diversity. Different searches disagree
usefully: on naproxen at 500 iterations Retro* returns 9 solved, structurally
distinct routes against MCTS's 5.

The bug these guard against was silent. `tree_search()` builds a tree only when
there is not one already, and the tree is what binds the algorithm, so the second
pass reused the first algorithm's tree. Pooling `mcts,retrostar` returned MCTS's
5 routes instead of the union, and nothing errored. Only counting routes caught
it, which is why these drive `plan()` rather than inspect its source.
"""

import pytest

from reagent.search import ALGORITHMS, resolve
from reagent.singlestep.aizynth import AiZynthBackend


class _FakeRoutes:
    def __init__(self, dicts):
        self.dicts = dicts


class _FakeFinder:
    """Stands in for AiZynthFinder, returning a different route per algorithm.

    Records the tree state seen at each search so a missing reset is visible.
    """

    def __init__(self, per_algorithm: dict[str, list[dict]]):
        self._per_algorithm = per_algorithm
        self.config = type("C", (), {"search": type("S", (), {"algorithm": None, "iteration_limit": 100})()})()
        self.tree = None
        self.routes = _FakeRoutes([])
        self.search_stats = {"returned_first": False, "iterations": 100}
        self.searches: list[tuple[str, bool]] = []

    def tree_search(self):
        # The real one only builds a tree when there is not one already.
        self.searches.append((self.config.search.algorithm, self.tree is not None))
        if self.tree is None:
            self.tree = f"tree-for-{self.config.search.algorithm}"
        built_for = self.tree.replace("tree-for-", "")
        self.routes = _FakeRoutes(self._per_algorithm.get(built_for, []))

    def build_routes(self):
        pass


def _leaf_tree(smiles: str) -> dict:
    return {"type": "mol", "smiles": "CCO", "in_stock": True,
            "children": [{"type": "reaction", "smiles": f"{smiles}>>CCO",
                          "children": [{"type": "mol", "smiles": smiles, "in_stock": True}]}]}


def _backend(algorithms: list[str], per_algorithm: dict[str, list[dict]]) -> AiZynthBackend:
    """A backend with the heavy constructor bypassed."""
    backend = object.__new__(AiZynthBackend)
    backend.algorithms = algorithms
    backend.algorithm = algorithms[0]
    backend._finder = _FakeFinder(per_algorithm)
    backend._relative_stocks = []
    backend.last_search_stats = {}
    backend._hit_time_limit_any = False
    backend._filter = None
    return backend


PER_ALGORITHM = {
    resolve("mcts"): [_leaf_tree("CC(=O)O"), _leaf_tree("CCN")],
    resolve("retrostar"): [_leaf_tree("CC(=O)O"), _leaf_tree("CCS"), _leaf_tree("CCC")],
}


def test_pooling_returns_the_union_not_the_first_algorithm():
    """The regression. MCTS alone finds 2, Retro* alone finds 3, one shared, so
    the union is 4. Before the tree reset this returned MCTS's 2."""
    solo_mcts = _backend(["mcts"], PER_ALGORITHM).plan("CCO", max_routes=40)
    solo_retro = _backend(["retrostar"], PER_ALGORITHM).plan("CCO", max_routes=40)
    pooled = _backend(["mcts", "retrostar"], PER_ALGORITHM).plan("CCO", max_routes=40)

    assert len(solo_mcts) == 2
    assert len(solo_retro) == 3
    assert len(pooled) == 4, "pooling did not union the two searches"
    assert len(pooled) > max(len(solo_mcts), len(solo_retro))


def test_the_tree_is_rebuilt_for_each_algorithm():
    """Directly asserts the condition the bug violated: no search may begin
    with a tree left over from the previous algorithm."""
    backend = _backend(["mcts", "retrostar"], PER_ALGORITHM)
    backend.plan("CCO", max_routes=40)

    algorithms_run = [name for name, _ in backend._finder.searches]
    reused_a_tree = [reused for _, reused in backend._finder.searches]

    assert algorithms_run == [resolve("mcts"), resolve("retrostar")]
    assert reused_a_tree == [False, False]


def test_duplicate_routes_are_collapsed():
    """Both searches find CC(=O)O. It must appear once."""
    pooled = _backend(["mcts", "retrostar"], PER_ALGORITHM).plan("CCO", max_routes=40)
    leaves = [tuple(sorted(m.smiles for m in r.leaves)) for r in pooled]
    assert len(leaves) == len(set(leaves))


def test_a_single_algorithm_is_unchanged_by_the_pooling_path():
    assert len(_backend(["mcts"], PER_ALGORITHM).plan("CCO", max_routes=40)) == 2


def test_max_routes_still_caps_the_pool():
    assert len(_backend(["mcts", "retrostar"], PER_ALGORITHM).plan("CCO", max_routes=3)) == 3


def test_every_named_algorithm_resolves():
    for name in ALGORITHMS:
        assert resolve(name)


def test_an_unknown_algorithm_names_the_valid_ones():
    with pytest.raises(ValueError, match="mcts"):
        resolve("nonsense")
