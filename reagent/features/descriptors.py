"""Molecule-level deterministic descriptors computed with RDKit.

Everything here is a pure function of a SMILES string. These are the raw numbers
the agents later reason over; no interpretation happens at this layer.
"""

from __future__ import annotations

import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalogParams

# Ertl & Schuffenhauer synthetic-accessibility score ships with RDKit's contrib
# tree; it is a published estimate of how hard a molecule is to make, which is a
# far better cost/rarity signal for a building block than raw atom count.
_SA_DIR = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _SA_DIR not in sys.path:
    sys.path.append(_SA_DIR)
try:
    import sascorer as _sascorer
except Exception:  # pragma: no cover - contrib not present in some builds
    _sascorer = None

# Structural-alert screen using the published Brenk "unwanted functionality"
# catalogue that ships with RDKit. These are medicinal-chemistry liability
# alerts (reactive, toxic, or metabolically unstable groups), not GHS reagent
# hazards: a hit flags a group worth an expert's attention, it does not assign a
# hazard category.
_ALERT_PARAMS = FilterCatalogParams()
_ALERT_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_ALERT_CATALOG = FilterCatalog.FilterCatalog(_ALERT_PARAMS)


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
    """Distinct Brenk structural-alert names present in the molecule."""
    m = mol(smiles)
    if m is None:
        return []
    return sorted({entry.GetDescription() for entry in _ALERT_CATALOG.GetMatches(m)})


def sa_score(smiles: str) -> float:
    """Synthetic-accessibility score (~1 easy to ~10 hard), or 0.0 if unavailable."""
    m = mol(smiles)
    if m is None or _sascorer is None:
        return 0.0
    try:
        return float(_sascorer.calculateScore(m))
    except Exception:
        return 0.0
