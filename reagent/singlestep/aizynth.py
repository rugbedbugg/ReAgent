"""Adapter around AiZynthFinder, our single-step + search backend.

AiZynthFinder gives us a pretrained USPTO template model (L2) driving an MCTS
search (L3) over a ZINC stock (L1). We run it and translate its nested route
trees into the backend-agnostic :class:`~reagent.core.models.Route`, so the rest
of REAGENT never imports AiZynthFinder directly.
"""

from __future__ import annotations

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
        expansion: str = "uspto",
        permissive_stock: int | None = None,
        iterations: int | None = None,
    ):
        # Imported lazily so importing reagent.core doesn't pull in the heavy
        # AiZynthFinder stack (and its slow numba/onnx imports).
        from aizynthfinder.aizynthfinder import AiZynthFinder

        self._finder = AiZynthFinder(configfile=str(config_file))
        selected = [stock]
        if permissive_stock is not None:
            from reagent.singlestep.stock import SizeStock

            self._finder.stock.load(SizeStock(max_heavy_atoms=permissive_stock), "permissive")
            selected.append("permissive")
        self._finder.stock.select(selected)
        self._finder.expansion_policy.select(expansion)
        self._filter = None
        try:
            self._finder.filter_policy.select(expansion)
            self._filter = self._finder.filter_policy[expansion]
        except (KeyError, ValueError):
            pass  # filter policy is optional
        # More MCTS iterations find routes the default budget misses; the gain is
        # real (measured) but linear in run time, so it stays an opt-in knob.
        if iterations is not None:
            self._finder.config.search.iteration_limit = iterations

    def plan(self, target_smiles: str, max_routes: int = 10) -> list[Route]:
        self._finder.target_smiles = target_smiles
        self._finder.tree_search()
        self._finder.build_routes()

        routes: list[Route] = []
        collection = self._finder.routes
        for tree in collection.dicts[:max_routes]:
            solved = self._is_solved(tree)
            route = _tree_to_route(target_smiles, tree, solved)
            self._score_filter(route)
            routes.append(route)
        return routes

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
