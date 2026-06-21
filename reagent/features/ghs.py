"""GHS hazard classification from PubChem, an opt-in online safety signal.

This is the real reagent-safety data the offline screens only approximate: GHS
hazard statements (H-codes) for each molecule in a route, fetched from PubChem's
free API. It is network-dependent, so every lookup is cached to disk and any
failure (offline, no CID, no GHS record) degrades gracefully to None, letting the
caller fall back to the offline Brenk screen.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from reagent.core.chem import canonical
from reagent.core.config import DATA_DIR

CACHE_PATH = DATA_DIR / "ghs_cache.json"
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"
_CODE_RE = re.compile(r"H\d{3}")

# H-code severity tiers, worst first. The worst tier a route's reagents trigger
# sets the safety score (lower = more hazardous). The floor is reserved for
# acutely lethal and confirmed CMR codes; the merely suspected/chronic codes
# (H341/H351/H360/H361), which are very common in PubChem, sit higher so they do
# not flatten every route to the floor.
SEVERITY_TIERS: list[tuple[float, set[str]]] = [
    (0.10, {"H300", "H310", "H330", "H340", "H350"}),          # fatal / confirmed mutagen or carcinogen
    (0.25, {"H301", "H311", "H331", "H341", "H351", "H360", "H370", "H372"}),  # toxic / suspected CMR / organ
    (0.45, {"H302", "H312", "H332", "H314", "H318", "H361", "H371", "H373"}),  # harmful / corrosive
    (0.65, {"H315", "H319", "H335", "H336", "H225", "H226", "H228", "H290"}),  # irritant / flammable
]


class GHSClient:
    def __init__(self, cache_path: str | Path = CACHE_PATH, timeout: int = 20, enabled: bool = True):
        self.cache_path = Path(cache_path)
        self.timeout = timeout
        self.enabled = enabled
        self._cache: dict[str, list[str]] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

    def h_codes(self, smiles: str) -> list[str] | None:
        """GHS H-codes for a molecule, or None if unavailable (offline/no record)."""
        key = canonical(smiles) or smiles
        if key in self._cache:
            return self._cache[key]
        if not self.enabled:
            return None
        try:
            cids = requests.post(
                f"{BASE}/pug/compound/smiles/cids/JSON", data={"smiles": key}, timeout=self.timeout
            ).json()["IdentifierList"]["CID"]
            payload = requests.get(
                f"{BASE}/pug_view/data/compound/{cids[0]}/JSON",
                params={"heading": "GHS Classification"},
                timeout=self.timeout,
            ).json()
            codes = sorted(set(_CODE_RE.findall(json.dumps(payload))))
        except Exception:
            return None
        self._cache[key] = codes
        self._save()
        time.sleep(0.2)  # be polite to the PubChem endpoint
        return codes


def _score_from_codes(codes: set[str]) -> float:
    for score, tier in SEVERITY_TIERS:
        if codes & tier:
            return score
    return 1.0 if not codes else 0.85  # listed but only mild/environmental


def enrich_ghs(route, client: GHSClient) -> bool:
    """Add GHS facts and a GHS safety score to ``route.features['safety']``.

    The score covers the reagents and intermediates the route actually handles,
    excluding the final target, which is common to every route to this molecule
    and would otherwise pin them all to the same floor. Returns True if any of
    those molecules had GHS data, False otherwise (caller keeps the Brenk score).
    """
    target = canonical(route.target) or route.target
    molecules: set[str] = {m.smiles for m in route.leaves}
    for rxn in route.reactions:
        molecules.add(rxn.product)
        molecules.update(rxn.precursors)
    molecules = {m for m in molecules if (canonical(m) or m) != target}

    all_codes: set[str] = set()
    covered = 0
    for smiles in molecules:
        codes = client.h_codes(smiles)
        if codes is not None:
            covered += 1
            all_codes.update(codes)
    if covered == 0:
        return False

    safety = route.features.setdefault("safety", {})
    safety["ghs_h_codes"] = sorted(all_codes)
    safety["ghs_molecules_covered"] = covered
    safety["ghs_safety"] = _score_from_codes(all_codes)
    return True
