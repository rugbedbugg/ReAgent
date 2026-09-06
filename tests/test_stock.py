"""SizeStock permissive-stock query test."""

from rdkit import Chem

from reagent.singlestep.stock import SizeStock


class _Mol:
    def __init__(self, smiles: str):
        self.rd_mol = Chem.MolFromSmiles(smiles)


def test_size_stock_cutoff():
    stock = SizeStock(max_heavy_atoms=11)
    assert _Mol("CCO") in stock  # ethanol, 3 heavy atoms
    assert _Mol("O=C(O)c1ccccc1O") in stock  # salicylic acid, 10 heavy atoms
    assert _Mol("CC(=O)Oc1ccccc1C(=O)O") not in stock  # aspirin, 13 heavy atoms


def test_size_stock_handles_bad_mol():
    stock = SizeStock()

    class _Broken:
        rd_mol = None

    assert _Broken() not in stock  # no crash on unparseable input


class _AlwaysStock:
    """Everything is purchasable, so only the size rule can reject."""

    def __contains__(self, mol) -> bool:
        return True

    def __len__(self) -> int:
        return 1


def _mol(smiles: str):
    from aizynthfinder.chem import Molecule

    return Molecule(smiles=smiles)


def test_relative_stock_rejects_a_leaf_that_is_most_of_the_target():
    """The defect this exists for: against a catalogue capped at 14 heavy
    atoms, 10 of 25 moderate targets were 'solved' by buying a molecule that
    was most of the answer, because the cap is absolute and the targets are
    small."""
    from reagent.singlestep.stock import TargetRelativeStock

    stock = TargetRelativeStock(_AlwaysStock(), max_leaf_fraction=0.6)
    stock.set_target_size(_mol("CC(=O)Nc1ccc(O)cc1").rd_mol.GetNumHeavyAtoms())  # 11

    assert _mol("CC(=O)O") in stock  # 4 atoms, 36% of the target
    assert _mol("COc1ccc(NC(C)=O)cc1") not in stock  # 12 atoms, larger than the target


def test_relative_stock_still_defers_to_the_inner_stock():
    """Small enough is necessary, not sufficient. A molecule nobody sells is
    still not purchasable."""
    from reagent.singlestep.stock import TargetRelativeStock

    class Nothing:
        def __contains__(self, mol):
            return False

        def __len__(self):
            return 0

    stock = TargetRelativeStock(Nothing(), max_leaf_fraction=0.6)
    stock.set_target_size(20)
    assert _mol("CC(=O)O") not in stock


def test_relative_stock_defers_until_a_target_is_set():
    """Constructed before planning starts, so there is nothing to be relative
    to yet. It must not reject everything in that window."""
    from reagent.singlestep.stock import TargetRelativeStock

    stock = TargetRelativeStock(_AlwaysStock(), max_leaf_fraction=0.6)
    assert stock.max_heavy_atoms is None
    assert _mol("COc1ccc(NC(C)=O)cc1") in stock


def test_the_cap_scales_with_the_target():
    """The whole point: one fraction gives different absolute caps for
    different targets, which an absolute --max-heavy-atoms cannot."""
    from reagent.singlestep.stock import TargetRelativeStock

    stock = TargetRelativeStock(_AlwaysStock(), max_leaf_fraction=0.6)
    stock.set_target_size(15)  # mean moderate target
    assert stock.max_heavy_atoms == 9
    stock.set_target_size(23)  # mean hard target
    assert stock.max_heavy_atoms == 13


def test_an_out_of_range_fraction_is_rejected():
    import pytest

    from reagent.singlestep.stock import TargetRelativeStock

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="max_leaf_fraction"):
            TargetRelativeStock(_AlwaysStock(), max_leaf_fraction=bad)
