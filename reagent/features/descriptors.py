"""Molecule-level deterministic descriptors computed with RDKit.

Everything here is a pure function of a SMILES string. These are the raw numbers
the agents later reason over; no interpretation happens at this layer.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

# Reactive or hazardous functional groups, matched as SMARTS. This is a
# screening heuristic, not a GHS classification: presence flags a molecule an
# expert should look at, it does not assign a hazard category.
HAZARD_SMARTS: dict[str, str] = {
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "azide": "[N-]=[N+]=N",
    "diazo": "[C]=[N+]=[N-]",
    "peroxide": "[OX2][OX2]",
    "nitro": "[NX3+](=O)[O-]",
    "isocyanate": "[NX2]=C=O",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "aldehyde": "[CX3H1](=O)[#6]",
    "michael_acceptor": "[CX3]=[CX3][CX3]=O",
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
}

_HAZARD_PATTERNS = {name: Chem.MolFromSmarts(sm) for name, sm in HAZARD_SMARTS.items()}


def mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def mol_weight(smiles: str) -> float:
    m = mol(smiles)
    return Descriptors.MolWt(m) if m is not None else 0.0


def heavy_atoms(smiles: str) -> int:
    m = mol(smiles)
    return m.GetNumHeavyAtoms() if m is not None else 0


def ring_count(smiles: str) -> int:
    m = mol(smiles)
    return Chem.rdMolDescriptors.CalcNumRings(m) if m is not None else 0


def hazard_groups(smiles: str) -> list[str]:
    """Names of hazardous/reactive groups present in the molecule."""
    m = mol(smiles)
    if m is None:
        return []
    return [name for name, patt in _HAZARD_PATTERNS.items() if patt and m.HasSubstructMatch(patt)]
