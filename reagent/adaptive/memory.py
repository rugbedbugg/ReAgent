"""Episodic memory of past planning runs.

Each solved target is logged with the weights used, the objective vectors of its
routes, and any later feedback. Similar past targets can then be recalled by
molecular fingerprint, so a new target can warm-start from what worked before.
Storage is a JSONL file, one episode per line.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from reagent.core.config import DATA_DIR

EPISODES_PATH = DATA_DIR / "episodes.jsonl"


class Episode(BaseModel):
    target: str
    weights: dict[str, float] = Field(default_factory=dict)
    score_vectors: list[dict[str, float]] = Field(default_factory=list)
    # Candidate-normalized copies of the same vectors. The preference update
    # compares objectives against each other, so it needs the comparable
    # scale for the same reason ranking does. Empty on episodes written
    # before normalization existed; those fall back to the raw vectors.
    normalized_vectors: list[dict[str, float]] = Field(default_factory=list)
    weighted_scores: list[float] = Field(default_factory=list)
    recommended: int = 0  # 1-based route number
    feedback: int | None = None  # route number the user actually preferred
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


class EpisodicMemory:
    def __init__(self, path: str | Path = EPISODES_PATH):
        self.path = Path(path)

    def append(self, episode: Episode) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(episode.model_dump_json() + "\n")

    def all(self) -> list[Episode]:
        if not self.path.exists():
            return []
        return [
            Episode.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def find_similar(self, target: str, k: int = 3) -> list[tuple[Episode, float]]:
        """Past episodes ranked by target-molecule Tanimoto similarity."""
        query = _fingerprint(target)
        if query is None:
            return []
        scored: list[tuple[Episode, float]] = []
        for episode in self.all():
            fp = _fingerprint(episode.target)
            if fp is None:
                continue
            scored.append((episode, DataStructs.TanimotoSimilarity(query, fp)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def last_for(self, target: str) -> Episode | None:
        matches = [e for e in self.all() if e.target == target]
        return matches[-1] if matches else None
