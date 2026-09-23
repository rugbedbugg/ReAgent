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


def m0_key(smiles: str) -> str | None:
    """
    M0 — strict molecular identity.

    Algorithm:
    1. Parse molecule with RDKit.
    2. Remove atom-map numbers only.
    3. Sanitize with RDKit.
    4. Emit canonical **isomeric** SMILES.

    M0 MUST preserve:
    - tetrahedral stereochemistry
    - alkene stereochemistry
    - formal charge
    - isotopes
    - disconnected fragments
    - tautomeric state
    - protonation state

    M0 MUST NOT:
    - neutralize
    - desalt
    - choose largest fragment
    - canonicalize tautomers
    - ignore stereochemistry
    """
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def m1_key(smiles: str) -> str | None:
    """
    M1 — stereo-agnostic diagnostic identity.

    Same connectivity/form semantics as M0, but stereochemical specification ignored.

    M1 is NOT the primary equality relation. It is used only for diagnostic analyses.
    Does not implement tautomer-insensitive or salt-parent identities.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetProp("_CIPCode", "")
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)
