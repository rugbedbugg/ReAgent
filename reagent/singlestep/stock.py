"""Stock queries: a permissive size heuristic, and a memory-light ZINC lookup.

AiZynthFinder's ``InMemoryInchiKeyQuery`` holds the whole catalogue as a Python
``set`` of InChI-key strings. For the bundled 17M-molecule ZINC stock that is
roughly 2.3 GB of the ~2.9 GB a planning run needs, which is most of why the
stack is memory-bound on a small machine and why searches get OOM-killed.
:class:`HashedStock` keeps the same membership test over a sorted array of
64-bit hashes instead, at about 136 MB.

The same array is how a real vendor catalogue gets in. ZINC is a fixed snapshot
and its gaps cap solve-rate; :func:`build_catalogue_cache` hashes an Enamine or
eMolecules download into the identical format, and :func:`merge_caches` unions
the two so the search sees both through one binary search.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import itertools
import multiprocessing
import os
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
from aizynthfinder.context.stock.queries import StockQueryMixin


class SizeStock(StockQueryMixin):
    """Treat any molecule at or below a heavy-atom cutoff as purchasable.

    The bundled ZINC stock is a fixed snapshot: a target's real precursors may
    simply be absent from it, which caps solve-rate. This probes how much of that
    cap is coverage by assuming anything small is buyable. It answers a different
    question than :func:`build_catalogue_cache`, which checks a real vendor
    catalogue instead of assuming -- prefer that when one is on hand, and use
    this to bound what a better catalogue could be worth.
    """

    def __init__(self, max_heavy_atoms: int = 11):
        self.max_heavy_atoms = max_heavy_atoms

    def __contains__(self, mol) -> bool:
        try:
            return mol.rd_mol.GetNumHeavyAtoms() <= self.max_heavy_atoms
        except Exception:
            return False


class TargetRelativeStock(StockQueryMixin):
    """Purchasable, and small enough relative to the target to be a precursor.

    A catalogue cap in absolute heavy atoms cannot serve targets of different
    sizes. Measured on the widened evaluation sets: a 14-heavy-atom building
    block is 62% of the mean hard target (22.7 atoms) and 93% of the mean
    moderate one (15.0). At 14 the hard set produces genuine routes averaging
    2.00 steps, while the moderate set collapses to 1.08 steps with 10 of 25
    routes buying a nearly finished molecule.

    The cap has to be a fraction of the target, and the target is only known
    once planning starts, so it cannot live in the catalogue. It lives here
    instead, wrapping whatever stock is underneath.

    This deliberately sits in the stock layer rather than in ranking. Filtering
    degenerate routes after the search wastes the search budget on routes that
    get discarded, can leave no candidates at all when every route is
    degenerate, and leaves solve-rate counting "bought the answer" as a solve.
    A route is solved iff every leaf is in stock, so constraining stock makes
    solve-rate honest in the same stroke.
    """

    def __init__(self, inner: StockQueryMixin, max_leaf_fraction: float = 0.6):
        if not 0.0 < max_leaf_fraction <= 1.0:
            raise ValueError(
                f"max_leaf_fraction must be in (0, 1], got {max_leaf_fraction}"
            )
        self.inner = inner
        self.max_leaf_fraction = max_leaf_fraction
        self._target_heavy_atoms: int | None = None

    def set_target_size(self, heavy_atoms: int) -> None:
        """Called once per target, before the search starts."""
        self._target_heavy_atoms = heavy_atoms if heavy_atoms > 0 else None

    @property
    def max_heavy_atoms(self) -> int | None:
        """The absolute cap this fraction implies for the current target."""
        if self._target_heavy_atoms is None:
            return None
        return int(self._target_heavy_atoms * self.max_leaf_fraction)

    def __contains__(self, mol) -> bool:
        if mol not in self.inner:
            return False
        cap = self.max_heavy_atoms
        if cap is None:
            # No target set yet, so there is nothing to be relative to. Defer
            # to the inner stock rather than silently rejecting everything.
            return True
        try:
            return mol.rd_mol.GetNumHeavyAtoms() <= cap
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.inner)


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


def cache_path_for_catalogue(catalogue_path: str | Path) -> Path:
    """``version.smi.gz`` -> ``version.hashes.npy``, stripping every suffix."""
    path = Path(catalogue_path)
    while path.suffix:
        path = path.with_suffix("")
    return path.with_suffix(".hashes.npy")


def _open_text(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8", errors="replace")


def iter_catalogue_smiles(path: str | Path) -> Iterator[str]:
    """Yield one SMILES per catalogue entry, from plain or gzipped input.

    Covers the two formats vendors actually ship: delimited text whose first
    field is the structure (eMolecules' ``.smi``, most CSV exports) and SDF
    (Enamine's building-block downloads). Header rows and junk are not filtered
    here -- RDKit rejects them when the keys are computed, which is the same
    test with no second guess about the file's shape.
    """
    path = Path(path)
    stem = path.name[:-3] if path.name.endswith(".gz") else path.name
    if stem.endswith((".sdf", ".sd")):
        from rdkit import Chem

        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as handle:
            for mol in Chem.ForwardSDMolSupplier(handle):
                if mol is not None:
                    yield Chem.MolToSmiles(mol)
        return

    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield re.split(r"[\s,;]+", line, maxsplit=1)[0]


def catalogue_keys(
    smiles: str,
    max_heavy_atoms: int | None = None,
    split_salts: bool = True,
) -> tuple[str, ...]:
    """InChI keys for one catalogue entry: zero, one, or two.

    Two when the entry is a salt. A catalogue lists what ships in the bottle --
    an amine hydrochloride, say -- while a retrosynthesis asks for the free
    amine, so both the whole entry and its largest fragment are indexed. Without
    that, a shelf full of purchasable salts reads as empty stock.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()

    candidates = [mol]
    if split_salts and "." in smiles:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(fragments) > 1:
            candidates.append(max(fragments, key=lambda f: f.GetNumHeavyAtoms()))

    keys = []
    for candidate in candidates:
        if max_heavy_atoms is not None and candidate.GetNumHeavyAtoms() > max_heavy_atoms:
            continue
        try:
            key = Chem.MolToInchiKey(candidate)
        except Exception:
            continue
        if key:
            keys.append(key)
    return tuple(dict.fromkeys(keys))


def _silence_rdkit() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _hashes_for_entry(
    smiles: str,
    max_heavy_atoms: int | None = None,
    split_salts: bool = True,
) -> list[int]:
    return [_hash_key(key) for key in catalogue_keys(smiles, max_heavy_atoms, split_salts)]


_BLOCK = 1_000_000
_SLICE = 250_000


def build_catalogue_cache(
    catalogue_path: str | Path,
    cache_path: str | Path | None = None,
    *,
    max_heavy_atoms: int | None = None,
    split_salts: bool = True,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Hash a vendor catalogue into the same format :class:`HashedStock` reads.

    InChI-key generation is the whole cost -- roughly a millisecond per
    structure -- so entries are parsed in a worker pool and only their 64-bit
    digests cross the process boundary. Hashes accumulate in fixed-size uint64
    blocks rather than a Python list, which keeps a 26M-entry catalogue in a few
    hundred MB instead of a few GB.

    ``max_heavy_atoms`` is the lever that decides what kind of catalogue this
    becomes. Vendor files mix genuine building blocks with screening compounds;
    leaving the cap off makes near-complete molecules purchasable and turns
    multi-step targets into one-step ones, which flatters solve-rate without
    reflecting a route anyone would run.
    """
    _silence_rdkit()

    catalogue_path = Path(catalogue_path)
    cache_path = Path(cache_path) if cache_path else cache_path_for_catalogue(catalogue_path)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    worker = functools.partial(
        _hashes_for_entry, max_heavy_atoms=max_heavy_atoms, split_salts=split_salts
    )
    blocks: list[np.ndarray] = []
    buffer: list[int] = []
    read = kept = 0

    with multiprocessing.Pool(workers, initializer=_silence_rdkit) as pool:
        entries = iter_catalogue_smiles(catalogue_path)
        while True:
            # Fed a slice at a time on purpose. ``Pool.imap`` drains its input
            # iterable as fast as the feeder thread can run, into a queue with
            # no bound, so handing it a 78M-entry catalogue buffers gigabytes of
            # SMILES in this process -- the same unbounded-load failure that
            # :class:`HashedStock` exists to avoid.
            batch = list(itertools.islice(entries, _SLICE))
            if not batch:
                break
            read += len(batch)
            for hashes in pool.map(worker, batch, chunksize=2000):
                kept += len(hashes)
                buffer.extend(hashes)
            if len(buffer) >= _BLOCK:
                blocks.append(np.fromiter(buffer, dtype=np.uint64, count=len(buffer)))
                buffer.clear()
            if progress:
                progress(read, kept)

    if buffer:
        blocks.append(np.fromiter(buffer, dtype=np.uint64, count=len(buffer)))
    merged = np.unique(np.concatenate(blocks)) if blocks else np.empty(0, dtype=np.uint64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, merged)
    return cache_path


def merge_caches(cache_paths: Iterable[str | Path], out_path: str | Path) -> Path:
    """Union several hashed catalogues into one, deduplicated and sorted.

    Unioning at build time rather than searching several arrays at run time
    keeps the lookup a single binary search, which is what makes the stock test
    cheap enough to sit in the search's inner loop.
    """
    arrays = [np.load(Path(p)) for p in cache_paths]
    if not arrays:
        raise ValueError("No caches to merge.")
    merged = np.unique(np.concatenate(arrays))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, merged)
    return out_path


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
