"""Objective-aware molecule costs, so the objectives steer the search itself.

Every objective in this project scores routes the search has *already found*.
That makes the objectives a re-ranking layer: they can pick the safest of the
candidates on offer, but they cannot cause a safer candidate to be found. If the
search never proposes a route avoiding methyl iodide, no weighting will produce
one.

Retro* takes a per-molecule cost model, and every ``MoleculeNode`` is built with
``cost = molecule_cost(mol)``, which feeds the value function that decides what
to expand next. Making a molecule expensive therefore steers the search away
from routes that pass through it, before any ranking happens.

The hook sees one molecule with no route or target context, so only per-molecule
objectives fit here: structural-alert hazard and synthetic accessibility. Atom
economy, step count and buy-versus-build are route-level and stay in the ranking
layer.

Whether steering beats re-ranking is an open question this makes testable, not
one it settles. A non-zero cost also drops Retro*'s admissibility guarantee --
the default ``ZeroMoleculeCost`` is a trivially admissible heuristic and these
are not, so the search is being guided rather than provably optimal.
"""

from __future__ import annotations

from reagent.features import descriptors as d


class HazardCost:
    """Charge a molecule for the structural alerts it carries.

    ``weight`` is the cost added per distinct Brenk alert. Retro* compares these
    against reaction costs derived from model log-probabilities, which run in
    the low single digits, so a weight near 1.0 makes one alert roughly as
    discouraging as one unlikely disconnection. Zero reproduces the default
    behaviour exactly, which is what makes an A/B measurement possible.
    """

    def __init__(self, weight: float = 1.0, cap: int = 3):
        self.weight = float(weight)
        self.cap = int(cap)

    def __repr__(self) -> str:
        return f"hazard(weight={self.weight})"

    def calculate(self, mol) -> float:
        try:
            alerts = len(d.hazard_groups(mol.smiles))
        except Exception:
            # A molecule the screen cannot parse is not evidence of hazard, and
            # this runs inside the search loop: never raise, never guess.
            return 0.0
        return self.weight * min(alerts, self.cap)


class AccessibilityCost:
    """Charge a molecule for being hard to make.

    Synthetic accessibility runs 1 (trivial) to 10 (hard). Only the excess over
    ``free`` is charged, so ordinary building blocks cost nothing and the search
    is pushed away from precursors that would themselves need a synthesis.
    """

    def __init__(self, weight: float = 0.5, free: float = 3.0):
        self.weight = float(weight)
        self.free = float(free)

    def __repr__(self) -> str:
        return f"accessibility(weight={self.weight})"

    def calculate(self, mol) -> float:
        try:
            score = d.sa_score(mol.smiles)
        except Exception:
            return 0.0
        return self.weight * max(0.0, score - self.free)
