"""Stock queries: a permissive size heuristic, and a memory-light ZINC lookup.

AiZynthFinder's ``InMemoryInchiKeyQuery`` holds the whole catalogue as a Python
``set`` of InChI-key strings. For the bundled 17M-molecule ZINC stock that is
roughly 2.3 GB of the ~2.9 GB a planning run needs, which is most of why the
stack is memory-bound on a small machine and why searches get OOM-killed.
:class:`HashedStock` keeps the same membership test over a sorted array of
64-bit hashes instead, at about 136 MB.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from aizynthfinder.context.stock.queries import StockQueryMixin


class SizeStock(StockQueryMixin):
    """Treat any molecule at or below a heavy-atom cutoff as purchasable.

    The bundled ZINC stock is a fixed snapshot: a target's real precursors may
    simply be absent from it, which caps solve-rate. This is a deliberately
    approximate stand-in for a fuller building-block catalogue, not a real one,
    and is meant to be unioned with ZINC to probe how stock coverage limits
    planning.
    """

    def __init__(self, max_heavy_atoms: int = 11):
        self.max_heavy_atoms = max_heavy_atoms

    def __contains__(self, mol) -> bool:
        try:
            return mol.rd_mol.GetNumHeavyAtoms() <= self.max_heavy_atoms
        except Exception:
            return False


def _hash_key(key: str) -> int:
    """Stable 64-bit digest of one InChI key.

    Stable across processes and runs, unlike ``hash()``, because the result is
    persisted to a cache file and reused.
    """
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def hash_keys(keys: Iterable[str]) -> np.ndarray:
    """Sorted uint64 array of hashed keys, ready for binary search."""
    hashes = np.fromiter((_hash_key(k) for k in keys), dtype=np.uint64)
    hashes.sort()
    return hashes


def cache_path_for(stock_path: str | Path) -> Path:
    return Path(stock_path).with_suffix(".hashes.npy")


def build_hash_cache(stock_path: str | Path, cache_path: str | Path | None = None) -> Path:
    """Read the HDF5 stock once and persist its key hashes.

    The bundled file is in HDF5 *fixed* format, which pandas cannot read in
    chunks, so this pays the full string-set memory cost exactly once. Every run
    afterwards loads the cache instead.
    """
    import pandas as pd

    stock_path = Path(stock_path)
    cache_path = Path(cache_path) if cache_path else cache_path_for(stock_path)

    frame = pd.read_hdf(stock_path, key="table")
    keys = frame["inchi_key"] if "inchi_key" in getattr(frame, "columns", []) else frame
    hashes = hash_keys(str(k) for k in np.asarray(keys).ravel())
    del frame, keys

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, hashes)
    return cache_path


class HashedStock(StockQueryMixin):
    """Membership over a sorted array of 64-bit InChI-key hashes.

    Collisions are the trade: with 17M keys in a 64-bit space the chance that
    any two collide at all is about 8e-6, and a collision would make one
    molecule look purchasable when it is not. That is far safer than a Bloom
    filter, whose false-positive rate applies to every lookup and would inflate
    solve-rate -- the number this project actually reports.
    """

    def __init__(self, cache_path: str | Path):
        self._hashes: np.ndarray = np.load(Path(cache_path))

    def __contains__(self, mol) -> bool:
        try:
            key = mol.inchi_key
        except Exception:
            return False
        target = np.uint64(_hash_key(key))
        idx = int(np.searchsorted(self._hashes, target))
        return idx < self._hashes.size and self._hashes[idx] == target

    def __len__(self) -> int:
        return int(self._hashes.size)
