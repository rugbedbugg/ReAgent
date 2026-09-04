"""Core data models shared across all layers.

A retrosynthetic plan is an AND-OR tree: OR-nodes are molecules, AND-nodes are
reactions. ``Route`` is a fully expanded plan whose leaf molecules are all in
stock. These models are deliberately backend-agnostic so the search engine
(AiZynthFinder today, something else tomorrow) can be swapped without touching
the agent, optimization, or RAG layers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Molecule(BaseModel):
    """A single molecule node in a route."""

    smiles: str
    in_stock: bool = False

    def __str__(self) -> str:
        return self.smiles


class Reaction(BaseModel):
    """A single retrosynthetic step: products come from precursors.

    ``smarts`` / ``rsmi`` hold the reaction SMILES when available. ``metadata``
    carries whatever the search backend attaches (template hash, model
    likelihood, classification, etc.) so downstream feature extraction can read
    it without a backend dependency.
    """

    product: str
    precursors: list[str] = Field(default_factory=list)
    rsmi: str | None = None
    metadata: dict = Field(default_factory=dict)


class Route(BaseModel):
    """A complete retrosynthetic route to in-stock building blocks.

    Represented as a flat list of reactions plus the target and the set of leaf
    molecules, which is enough for every evaluator; the nested tree is kept in
    ``tree`` for rendering and for backends that need it.
    """

    target: str
    reactions: list[Reaction] = Field(default_factory=list)
    leaves: list[Molecule] = Field(default_factory=list)
    solved: bool = False
    tree: dict | None = None
    # Populated by the features, agent, and optimization layers respectively.
    features: dict = Field(default_factory=dict)
    assessments: list[Assessment] = Field(default_factory=list)
    scores: dict = Field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.reactions)


class Assessment(BaseModel):
    """One specialist agent's judgment of a route along one objective.

    The agent produces ``score`` and ``rationale`` by *interpreting* the
    deterministic facts in ``inputs`` (computed by the features layer). It never
    computes chemistry itself; ``inputs`` and ``evidence`` are its only grounds.
    """

    objective: str
    score: float = Field(ge=0.0, le=1.0)  # normalized, higher = better
    rationale: str
    inputs: dict = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)


Route.model_rebuild()
