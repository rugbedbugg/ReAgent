"""Ingesting a vendor building-block catalogue into the hashed stock format."""

import gzip

import numpy as np
import pytest

from reagent.singlestep.stock import (
    HashedStock,
    build_catalogue_cache,
    cache_path_for_catalogue,
    catalogue_keys,
    hash_keys,
    iter_catalogue_smiles,
    merge_caches,
)

# Real building blocks, small enough that the heavy-atom cap can separate them.
BENZOIC_ACID = "OC(=O)c1ccccc1"          # 9 heavy atoms
ANILINE = "Nc1ccccc1"                    # 7 heavy atoms
ANILINE_HCL = "Cl.Nc1ccccc1"             # a salt, as a catalogue would list it
NAPROXEN = "COc1ccc2cc(C(C)C(=O)O)ccc2c1"  # 19: a drug, not a building block


class _Mol:
    """Stands in for AiZynthFinder's Molecule, which exposes ``inchi_key``."""

    def __init__(self, smiles):
        from rdkit import Chem

        self.inchi_key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def write_catalogue(path, lines):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def test_reads_the_first_field_of_a_delimited_line(tmp_path):
    path = write_catalogue(
        tmp_path / "cat.smi",
        ["isosmiles version_id parent_id", f"{ANILINE} 12345 999", f"{BENZOIC_ACID},7,7"],
    )
    assert list(iter_catalogue_smiles(path)) == ["isosmiles", ANILINE, BENZOIC_ACID]


def test_reads_gzipped_catalogues(tmp_path):
    path = write_catalogue(tmp_path / "cat.smi.gz", [ANILINE, BENZOIC_ACID])
    assert list(iter_catalogue_smiles(path)) == [ANILINE, BENZOIC_ACID]


def test_unparseable_entries_yield_no_keys():
    # The header row of a vendor file reaches RDKit; it must not become stock.
    assert catalogue_keys("isosmiles") == ()


def test_a_salt_indexes_its_free_base_too():
    keys = catalogue_keys(ANILINE_HCL)
    assert _Mol(ANILINE).inchi_key in keys, "buying the hydrochloride makes the amine available"
    assert len(keys) == 2


def test_split_salts_can_be_turned_off():
    assert _Mol(ANILINE).inchi_key not in catalogue_keys(ANILINE_HCL, split_salts=False)


def test_the_heavy_atom_cap_drops_screening_compounds():
    assert catalogue_keys(NAPROXEN, max_heavy_atoms=11) == ()
    assert catalogue_keys(BENZOIC_ACID, max_heavy_atoms=11) != ()


def test_built_cache_answers_membership(tmp_path):
    path = write_catalogue(tmp_path / "cat.smi", [ANILINE, BENZOIC_ACID, NAPROXEN])
    cache = build_catalogue_cache(path, tmp_path / "cat.hashes.npy")
    stock = HashedStock(cache)

    assert _Mol(ANILINE) in stock
    assert _Mol(BENZOIC_ACID) in stock
    assert _Mol("CCOC(=O)c1ccc(N)cc1") not in stock


def test_the_cap_is_applied_when_building(tmp_path):
    path = write_catalogue(tmp_path / "cat.smi", [ANILINE, NAPROXEN])
    cache = build_catalogue_cache(path, tmp_path / "cat.hashes.npy", max_heavy_atoms=11)
    stock = HashedStock(cache)

    assert _Mol(ANILINE) in stock
    assert _Mol(NAPROXEN) not in stock


def test_duplicate_entries_are_stored_once(tmp_path):
    path = write_catalogue(tmp_path / "cat.smi", [ANILINE] * 5 + [BENZOIC_ACID])
    cache = build_catalogue_cache(path, tmp_path / "cat.hashes.npy")
    assert len(HashedStock(cache)) == 2


def test_merging_unions_two_catalogues(tmp_path):
    zinc = tmp_path / "zinc.npy"
    np.save(zinc, hash_keys([_Mol(BENZOIC_ACID).inchi_key]))
    vendor = build_catalogue_cache(
        write_catalogue(tmp_path / "cat.smi", [ANILINE]), tmp_path / "vendor.npy"
    )

    stock = HashedStock(merge_caches([zinc, vendor], tmp_path / "merged.npy"))
    assert _Mol(ANILINE) in stock
    assert _Mol(BENZOIC_ACID) in stock
    assert len(stock) == 2


def test_merging_nothing_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        merge_caches([], tmp_path / "merged.npy")


def test_cache_path_strips_every_suffix():
    assert cache_path_for_catalogue("data/version.smi.gz").name == "version.hashes.npy"
