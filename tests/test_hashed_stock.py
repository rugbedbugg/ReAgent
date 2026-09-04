"""Hashed stock lookup: same membership answers, a fraction of the memory."""

import numpy as np
import pytest

from reagent.singlestep.stock import HashedStock, _hash_key, cache_path_for, hash_keys


class _Mol:
    def __init__(self, key):
        self.inchi_key = key


@pytest.fixture
def stock(tmp_path):
    keys = [f"KEY{i:022d}-X" for i in range(1000)]
    path = tmp_path / "stock.hashes.npy"
    np.save(path, hash_keys(keys))
    return HashedStock(path), keys


def test_known_members_are_found(stock):
    query, keys = stock
    for key in (keys[0], keys[500], keys[-1]):
        assert _Mol(key) in query


def test_non_members_are_rejected(stock):
    query, _ = stock
    assert _Mol("ABSENTABSENTABSENTABS-Y") not in query
    # Same shape as a member key, but past the end of the catalogue.
    assert _Mol("KEY0000000000000000005000-X") not in query


def test_length_reports_the_catalogue_size(stock):
    query, keys = stock
    assert len(query) == len(keys)


def test_hashes_are_stable_across_calls():
    # The cache is persisted and reused, so the digest cannot be process-random
    # the way builtin hash() is.
    assert _hash_key("ABCDEFGHIJKLMN-OPQRSTUVWX-Y") == _hash_key("ABCDEFGHIJKLMN-OPQRSTUVWX-Y")


def test_hashes_are_sorted_for_binary_search():
    hashes = hash_keys([f"KEY{i:022d}-X" for i in range(200)])
    assert np.all(np.diff(hashes.astype(object)) > 0)


def test_a_molecule_without_an_inchi_key_is_not_in_stock(stock):
    query, _ = stock

    class _Broken:
        @property
        def inchi_key(self):
            raise ValueError("no key")

    assert _Broken() not in query


def test_cache_path_sits_beside_the_stock_file():
    assert cache_path_for("/data/zinc_stock.hdf5").name == "zinc_stock.hashes.npy"
