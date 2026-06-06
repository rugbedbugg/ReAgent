"""Thin RDKit helpers. All molecule parsing/canonicalization goes through here."""

from __future__ import annotations

from rdkit import Chem, RDLogger

# RDKit is noisy about parse failures; we handle them explicitly.
RDLogger.DisableLog("rdApp.*")


def canonical(smiles: str) -> str | None:
    """Return the canonical SMILES, or ``None`` if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def is_valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)
