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
