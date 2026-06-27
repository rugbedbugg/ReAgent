"""Precedent retrieval: ground a disconnection in similar known reactions.

For each reaction in a route we take its template SMARTS (AiZynthFinder puts it
in the reaction metadata) and retrieve the most similar templates from the USPTO
corpus, along with their corpus occurrence counts. The result is attached to the
route as evidence and folded into the feasibility facts the agent reasons over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reagent.core.config import DATA_DIR
from reagent.core.models import Route
from reagent.rag.corpus import load_templates
from reagent.rag.embed import ReactionFingerprintEmbedder
from reagent.rag.index import FingerprintIndex

INDEX_PATH = DATA_DIR / "rag_index.npz"


def build_index(
    min_occurence: int = 1,
    embedder: ReactionFingerprintEmbedder | None = None,
    save_to: str | Path | None = INDEX_PATH,
    progress: bool = False,
) -> FingerprintIndex:
    """Embed the template corpus into a searchable index (built once, cached)."""
    embedder = embedder or ReactionFingerprintEmbedder()
    records = load_templates(min_occurence=min_occurence)

    vectors: list[np.ndarray] = []
    kept: list[dict] = []
    for i, rec in enumerate(records):
        vec = embedder.embed(rec["retro_template"])
        if vec is not None:
            vectors.append(vec)
            kept.append(rec)
        if progress and i % 5000 == 0:
            print(f"  embedded {i}/{len(records)} templates")

    matrix = np.vstack(vectors).astype(np.uint8)
    index = FingerprintIndex(matrix, kept)
    if save_to:
        index.save(save_to)
    return index


class PrecedentRetriever:
    def __init__(self, index: FingerprintIndex | None = None):
        self.embedder = ReactionFingerprintEmbedder()
        self.index = index or self._load_or_build()

    def _load_or_build(self) -> FingerprintIndex:
        if INDEX_PATH.exists():
            return FingerprintIndex.load(INDEX_PATH)
        return build_index()

    def retrieve_for_template(self, template_smarts: str, k: int = 3) -> list[dict]:
        """Top-k precedent templates similar to a disconnection's template."""
        query = self.embedder.embed(template_smarts)
        if query is None:
            return []
        hits = self.index.search(query, k=k)
        return [
            {
                "template_hash": rec["template_hash"],
                "library_occurence": rec["library_occurence"],
                "similarity": round(sim, 3),
            }
            for rec, sim in hits
        ]

    def ground_route(self, route: Route, k: int = 3) -> Route:
        """Attach retrieved precedents to each reaction and to feasibility facts."""
        all_precedents: list[dict] = []
        for rxn in route.reactions:
            template = rxn.metadata.get("template")
            precedents = self.retrieve_for_template(template, k=k) if template else []
            rxn.metadata["precedents"] = precedents
            all_precedents.extend(precedents)

        if route.features.get("feasibility") is not None:
            occ = [p["library_occurence"] for p in all_precedents]
            route.features["feasibility"]["precedent_count"] = len(all_precedents)
            route.features["feasibility"]["max_precedent_occurence"] = max(occ) if occ else 0
            route.features["feasibility"]["mean_precedent_similarity"] = (
                round(float(np.mean([p["similarity"] for p in all_precedents])), 3)
                if all_precedents
                else 0.0
            )
        return route
