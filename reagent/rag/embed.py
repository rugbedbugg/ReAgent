"""Reaction embedding for precedent retrieval.

Uses an RDKit structural reaction fingerprint, so retrieval is chemically
grounded and runs fully offline (no model download). The embedder is a small
interface, so a text-embedding backend can be swapped in later without touching
the index or retriever.
"""

from __future__ import annotations

import numpy as np
from rdkit import RDLogger
from rdkit.Chem import DataStructs
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048


class ReactionFingerprintEmbedder:
    name = "rdkit-structural"
    n_bits = N_BITS

    def embed(self, reaction_smarts: str) -> np.ndarray | None:
        """Return a 0/1 fingerprint vector for a reaction SMARTS, or None."""
        try:
            rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        except Exception:
            return None
        if rxn is None or rxn.GetNumReactantTemplates() == 0:
            return None
        try:
            params = rdChemReactions.ReactionFingerprintParams()
            params.fpSize = self.n_bits
            fp = rdChemReactions.CreateStructuralFingerprintForReaction(rxn, params)
        except Exception:
            return None
        arr = np.zeros((self.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
