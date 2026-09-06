"""Adapter around AiZynthFinder, our single-step + search backend.

AiZynthFinder gives us a pretrained USPTO template model (L2) driving an MCTS
search (L3) over a ZINC stock (L1). We run it and translate its nested route
trees into the backend-agnostic :class:`~reagent.core.models.Route`, so the rest
of REAGENT never imports AiZynthFinder directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from reagent.core.models import Molecule, Reaction, Route


def _walk(
    node: dict,
    reactions: list[Reaction],
    leaves: list[Molecule],
    parent_mol: str | None = None,
) -> None:
    """Depth-first flatten of an AiZynthFinder route tree.

    Nodes alternate mol -> reaction -> mol. A mol node with no children is a
    leaf; we record its stock status. A reaction node's children are its
    precursors and the molecule above it is the product it forms.
    """
    if node.get("type") == "mol":
        children = node.get("children") or []
        if not children:
            leaves.append(Molecule(smiles=node["smiles"], in_stock=node.get("in_stock", False)))
            return
        for child in children:  # a solved mol has exactly one reaction child
            _walk(child, reactions, leaves, parent_mol=node["smiles"])
    elif node.get("type") == "reaction":
        precursors = [c["smiles"] for c in node.get("children", [])]
        reactions.append(
            Reaction(
                product=parent_mol or "",
                precursors=precursors,
                rsmi=node.get("smiles"),
                metadata=node.get("metadata", {}),
            )
        )
        for child in node.get("children", []):
            _walk(child, reactions, leaves)


def _tree_to_route(target: str, tree: dict, solved: bool) -> Route:
    reactions: list[Reaction] = []
    leaves: list[Molecule] = []
    _walk(tree, reactions, leaves)
    return Route(target=target, reactions=reactions, leaves=leaves, solved=solved, tree=tree)


class AiZynthBackend:
    """Runs AiZynthFinder for a target and returns normalized routes."""

    def __init__(
        self,
        config_file: str | Path,
        stock: str = "zinc",
        expansion: str | Sequence[str] = "uspto",
        permissive_stock: int | None = None,
        hashed_stock: bool = False,
        stock_cache: str | Path | None = None,
        iterations: int | None = None,
        time_limit: int | None = None,
        algorithm: str = "mcts",
        molecule_cost: dict | None = None,
        cutoff_number: int | None = None,
        cutoff_cumulative: float | None = None,
        max_leaf_fraction: float | None = None,
    ):
        # Imported lazily so importing reagent.core doesn't pull in the heavy
        # AiZynthFinder stack (and its slow numba/onnx imports).
        from aizynthfinder.aizynthfinder import AiZynthFinder

        if not hashed_stock:
            self._finder = AiZynthFinder(configfile=str(config_file))
            selected = [stock]
        else:
            # The hashed catalogue only pays off if the original is never built.
            # AiZynthFinder loads every stock named in the config eagerly, and
            # that load -- pandas DataFrame and the string set alive at once --
            # is what peaks at ~4.8 GB and gets runs OOM-killed. Steady state is
            # 2.24 GB, of which the string set is 1.76 GB. Handing it a config
            # with no stock section keeps both costs off the table; the hashed
            # keys are loaded instead, at ~140 MB.
            import yaml

            from reagent.core.config import DATA_DIR
            from reagent.singlestep.stock import HashedStock, cache_path_for

            cache = (
                Path(stock_cache)
                if stock_cache
                else cache_path_for(DATA_DIR / "zinc_stock.hdf5")
            )
            if not cache.exists():
                raise FileNotFoundError(
                    f"No hashed stock cache at {cache}. Build it once with:\n"
                    "  reagent build-stock-cache"
                )
            config = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
            config.pop("stock", None)
            self._finder = AiZynthFinder(configdict=config)
            self._finder.stock.load(HashedStock(cache), "zinc_hashed")
            selected = ["zinc_hashed"]
        if permissive_stock is not None:
            from reagent.singlestep.stock import SizeStock

            self._finder.stock.load(SizeStock(max_heavy_atoms=permissive_stock), "permissive")
            selected.append("permissive")

        # A leaf may only count as purchasable if it is also small enough
        # relative to the target. Each selected stock is wrapped rather than the
        # collection, which is exact rather than convenient: selection is a
        # union, and (A | B) & Size == (A & Size) | (B & Size).
        self._relative_stocks: list = []
        if max_leaf_fraction is not None:
            from reagent.singlestep.stock import TargetRelativeStock

            wrapped_names = []
            for key in selected:
                wrapped = TargetRelativeStock(self._finder.stock[key], max_leaf_fraction)
                name = f"{key}_relative"
                self._finder.stock.load(wrapped, name)
                self._relative_stocks.append(wrapped)
                wrapped_names.append(name)
            selected = wrapped_names

        self._finder.stock.select(selected)

        # Several expansion policies can run together: the policy collection
        # concatenates the actions and priors of every selected policy, so the
        # search sees the union of their disconnections. Combining the USPTO and
        # ringbreaker models is not the same experiment as swapping one for the
        # other -- ringbreaker alone proposes ring disconnections at the expense
        # of everything else, which is why it does nothing as a replacement.
        #
        # Priors from different models are concatenated without renormalization,
        # so the combined list need not sum to one. MCTS uses priors to order
        # and weight children, which tolerates that.
        self.expansion_keys = [expansion] if isinstance(expansion, str) else list(expansion)
        self._finder.expansion_policy.select(self.expansion_keys)

        # The filter model is trained alongside the primary policy, so it keys
        # off the first-named expansion policy.
        primary = self.expansion_keys[0]
        self._filter = None
        try:
            self._finder.filter_policy.select(primary)
            self._filter = self._finder.filter_policy[primary]
        except (KeyError, ValueError):
            pass  # filter policy is optional
        # More MCTS iterations find routes the default budget misses; the gain is
        # real (measured) but linear in run time, so it stays an opt-in knob.
        #
        # The search loop stops on whichever limit binds first:
        #     while time_past < time_limit and i <= iteration_limit
        # so raising the iteration budget alone does nothing once the wall-clock
        # default (120 s) binds, which it does on a slow machine well before a
        # few hundred iterations. Raising one without the other silently
        # measures the timeout instead of the budget.
        # How many templates each expansion offers. The policy returns
        #     min(index where cumulative prior >= cutoff_cumulative, cutoff_number)
        # templates, and measurement shows the count cap is what binds: every
        # policy returns exactly its 50-template default for every molecule
        # tried, so the next-best disconnections are being discarded before the
        # search ever sees them. Raising it widens the disconnection space with
        # more of the same policy's suggestions, at a cost in branching factor,
        # run time, and memory.
        for key in self.expansion_keys:
            strategy = self._finder.expansion_policy[key]
            if cutoff_number is not None:
                strategy.cutoff_number = cutoff_number
            if cutoff_cumulative is not None:
                strategy.cutoff_cumulative = cutoff_cumulative

        from reagent.search import resolve as resolve_algorithm

        # Several searches can run over the same single-step model and their
        # results pooled. They disagree usefully: on naproxen at 500 iterations
        # Retro* returned 9 solved, structurally distinct routes against MCTS's
        # 5, overlapping only partly. Selection can only choose among what it is
        # given, so the union is strictly more to choose from. Combining
        # *searches* is a different experiment from combining expansion
        # policies, which was measured and did nothing.
        self.algorithms = [algorithm] if isinstance(algorithm, str) else list(algorithm)
        if not self.algorithms:
            raise ValueError("At least one search algorithm is required.")
        # Resolve every name now, not lazily in plan(): a typo in the second
        # algorithm would otherwise surface only after the model and stock have
        # loaded and the first search has run.
        for name in self.algorithms:
            resolve_algorithm(name)
        self.algorithm = self.algorithms[0]
        self._finder.config.search.algorithm = resolve_algorithm(self.algorithm)

        # Retro* builds every MoleculeNode with cost = molecule_cost(mol), and
        # that cost feeds the value function deciding what to expand next. So a
        # cost here steers which routes are *found*, unlike the objectives in
        # the ranking layer, which can only reorder what the search returned.
        # Ignored by every other algorithm, which has no such hook.
        if molecule_cost:
            self._finder.config.search.algorithm_config["molecule_cost"] = dict(molecule_cost)
        if iterations is not None:
            self._finder.config.search.iteration_limit = iterations
        if time_limit is not None:
            self._finder.config.search.time_limit = time_limit
        self.last_search_stats: dict = {}
        self._hit_time_limit_any = False

    def plan(self, target_smiles: str, max_routes: int = 10) -> list[Route]:
        self._finder.target_smiles = target_smiles

        # The size cap is a fraction of *this* target, so it can only be fixed
        # now. One backend plans many targets, sequentially within a worker, so
        # this is set per call rather than at construction.
        if self._relative_stocks:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(target_smiles)
            size = molecule.GetNumHeavyAtoms() if molecule else 0
            for stock in self._relative_stocks:
                stock.set_target_size(size)

        from reagent.optimize.aggregate import route_signature
        from reagent.search import resolve as resolve_algorithm

        pooled: list[Route] = []
        seen: set = set()
        self.last_search_stats = {}
        self._hit_time_limit_any = False

        for name in self.algorithms:
            self._finder.config.search.algorithm = resolve_algorithm(name)
            self._finder.tree_search()
            stats = dict(self._finder.search_stats)
            self.last_search_stats = stats
            self._hit_time_limit_any = self._hit_time_limit_any or self._stats_hit_limit(stats)
            self._finder.build_routes()

            for tree in self._finder.routes.dicts:
                route = _tree_to_route(target_smiles, tree, self._is_solved(tree))
                # Two searches over one model find the same route often enough
                # that pooling without this would just pad the candidate list
                # with duplicates and crowd out genuine alternatives.
                signature = route_signature(route)
                if signature in seen:
                    continue
                seen.add(signature)
                self._score_filter(route)
                pooled.append(route)

        # Solved routes first, so a pool that overflows max_routes does not drop
        # a solved route in favour of an unsolved one from an earlier search.
        pooled.sort(key=lambda r: (not r.solved,))
        return pooled[:max_routes]

    @property
    def search_time_limit(self) -> int:
        """Wall-clock limit the search is running under, in seconds."""
        return self._finder.config.search.time_limit

    @property
    def search_iteration_limit(self) -> int:
        """Iteration budget the search is running under."""
        return self._finder.config.search.iteration_limit

    @property
    def search_hit_time_limit(self) -> bool:
        """True if the last search stopped on the clock, not the iteration budget.

        When this is true the iteration budget was never spent, so any
        comparison across ``iterations`` values is measuring the wall-clock
        limit rather than the search budget.
        """
        # With several algorithms pooled, "the last search" is not the whole
        # story: any one of them hitting the clock invalidates a comparison
        # across iteration budgets, so the flag is sticky across the pool.
        if getattr(self, "_hit_time_limit_any", False):
            return True
        return self._stats_hit_limit(self.last_search_stats)

    def _stats_hit_limit(self, stats: dict) -> bool:
        if not stats or stats.get("returned_first"):
            return False
        return stats.get("iterations", 0) < self._finder.config.search.iteration_limit

    def _score_filter(self, route: Route) -> None:
        """Attach the filter model's forward-plausibility score to each reaction.

        The expansion policy says how likely a disconnection is to be *suggested*;
        the filter model says how plausible the resulting reaction actually is.
        The two are different signals, and a route can score well on the first
        while containing an implausible step the second catches.
        """
        if self._filter is None:
            return
        from aizynthfinder.chem import SmilesBasedRetroReaction, TreeMolecule

        for rxn in route.reactions:
            try:
                product = TreeMolecule(parent=None, smiles=rxn.product)
                retro = SmilesBasedRetroReaction(product, reactants_str=".".join(rxn.precursors))
                _, prob = self._filter.feasibility(retro)
                rxn.metadata["filter_feasibility"] = round(float(prob), 4)
            except Exception:
                pass  # skip steps the filter cannot score

    @staticmethod
    def _is_solved(tree: dict) -> bool:
        """A route is solved iff every leaf molecule is in stock."""
        stack = [tree]
        while stack:
            node = stack.pop()
            children = node.get("children") or []
            if node.get("type") == "mol" and not children and not node.get("in_stock", False):
                return False
            stack.extend(children)
        return True
