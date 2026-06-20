"""A permissive size-based stock query.

The bundled ZINC stock is large but is still a fixed snapshot: a target's real
precursors may simply be absent from it, which caps solve-rate. This query treats
any molecule at or below a heavy-atom cutoff as purchasable, on the heuristic
that small molecules are generally commercially available. It is a deliberately
approximate stand-in for a fuller building-block catalogue, not a real one, and
is meant to be unioned with ZINC to probe how stock coverage limits planning.
"""

from __future__ import annotations

from aizynthfinder.context.stock.queries import StockQueryMixin


class SizeStock(StockQueryMixin):
    def __init__(self, max_heavy_atoms: int = 11):
        self.max_heavy_atoms = max_heavy_atoms

    def __contains__(self, mol) -> bool:
        try:
            return mol.rd_mol.GetNumHeavyAtoms() <= self.max_heavy_atoms
        except Exception:
            return False
