"""Literature reference data foundation for ReAgent benchmark.

This module provides the curated literature-reference layer, distinct from
generated routes. It implements strict molecular identity (M0), stereo-agnostic
diagnostic identity (M1), and validation invariants for reference records.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reagent.core.chem import m0_key, m1_key
from reagent.eval.reference_review import Reference as LegacyReference


class SourceType(str, Enum):
    PEER_REVIEWED_ARTICLE = "peer_reviewed_article"
    PATENT = "patent"
    REVIEW = "review"
    DATABASE = "database"
    OTHER = "other"


class RouteScope(str, Enum):
    FULL_ROUTE = "full_route"
    TERMINAL_SEGMENT = "terminal_segment"
    INTERNAL_SEGMENT = "internal_segment"


class RouteCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"


class MolecularForm(str, Enum):
    NEUTRAL = "neutral"
    SALT = "salt"
    SOLVATE = "solvate"
    MIXTURE = "mixture"
    PROTECTED = "protected"
    UNKNOWN = "unknown"


class StereoStatus(str, Enum):
    ACHIRAL = "achiral"
    RACEMIC = "racemic"
    STEREOSPECIFIC = "stereospecific"
    MIXTURE = "mixture"
    UNSPECIFIED = "unspecified"


class StereoRelation(str, Enum):
    EXACTLY_COMPATIBLE = "exactly_compatible"
    RACEMATE_COMPATIBLE = "racemate_compatible"
    REFERENCE_MORE_SPECIFIC = "reference_more_specific"
    CONFLICTING = "conflicting"
    UNRESOLVED = "unresolved"


class MoleculeRole(str, Enum):
    TARGET = "target"
    STARTING_MATERIAL = "starting_material"
    INTERMEDIATE = "intermediate"
    REAGENT = "reagent"
    CATALYST = "catalyst"
    SOLVENT = "solvent"
    BYPRODUCT = "byproduct"
    OTHER = "other"


class StructureResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    NOT_REPORTED = "not_reported"


class YieldType(str, Enum):
    ISOLATED = "isolated"
    NMR = "nmr"
    HPLC = "hplc"
    GC = "gc"
    ESTIMATED = "estimated"
    NOT_REPORTED = "not_reported"


class EvidenceLocator(BaseModel):
    """Location of evidence within a source document."""
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    example: str | None = None
    scheme: str | None = None
    page: str | None = None
    section: str | None = None
    claim: str | None = None
    paragraph: str | None = None
    curator_note: str | None = None


class Conditions(BaseModel):
    """Reaction conditions (evidence only, not used for identity/comparison)."""
    model_config = ConfigDict(extra="forbid")

    reagents: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    solvents: list[str] = Field(default_factory=list)
    temperature_c: float | None = None
    time_h: float | None = None
    pressure_atm: float | None = None
    notes: str | None = None


class Yield(BaseModel):
    """Reaction yield (evidence only)."""
    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    unit: str = "percent"
    type: YieldType = YieldType.NOT_REPORTED
    reported: bool = False


class ReferenceMolecule(BaseModel):
    """A molecule as reported in the literature source."""
    model_config = ConfigDict(extra="forbid")

    molecule_id: str = Field(min_length=1)
    reported_name: str | None = None
    reported_structure: str | None = None
    reported_smiles: str | None = None
    role: MoleculeRole = MoleculeRole.OTHER
    structure_resolution: StructureResolutionStatus = StructureResolutionStatus.NOT_REPORTED
    stereo_status: StereoStatus = StereoStatus.UNSPECIFIED
    molecular_form: MolecularForm = MolecularForm.UNKNOWN
    evidence_locators: list[EvidenceLocator] = Field(default_factory=list)
    m0: str | None = None
    m1: str | None = None

    @model_validator(mode="after")
    def compute_keys(self) -> ReferenceMolecule:
        if self.reported_smiles:
            self.m0 = m0_key(self.reported_smiles)
            self.m1 = m1_key(self.reported_smiles)
        return self


class ReferenceStep(BaseModel):
    """A transformation step in the literature reference."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    step_id: str = Field(min_length=1)
    sequence_index: int | None = None
    precursor_ids: list[str] = Field(default_factory=list)
    product_id: str = Field(min_length=1)
    evidence_locators: list[EvidenceLocator] = Field(default_factory=list)
    conditions: Conditions | None = None
    yield_: Yield | None = Field(default=None, alias="yield")
    transformation_notes: str | None = None
    ambiguity_metadata: dict = Field(default_factory=dict)
    inferred: bool = False

    @model_validator(mode="after")
    def check_evidence(self) -> ReferenceStep:
        if not self.evidence_locators:
            raise ValueError(f"Step {self.step_id} must have at least one evidence locator")
        return self

    def model_dump(self, *args, **kwargs):
        # Exclude unset/None yield field to avoid extra field validation on re-validate
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump(*args, **kwargs)


class SourceRecord(BaseModel):
    """A literature source."""
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    source_type: SourceType = SourceType.OTHER
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    assignee: str | None = None
    year: int | None = None
    doi: str | None = None
    patent_number: str | None = None
    url: str | None = None
    is_primary: bool = False


class TargetRecord(BaseModel):
    """Target molecule information."""
    model_config = ConfigDict(extra="forbid")

    reagent_target_name: str = Field(min_length=1)
    reagent_target_smiles: str = Field(min_length=1)
    evaluation_group: str | None = None
    canonical_target_smiles: str | None = None


class CurationMetadata(BaseModel):
    """Metadata about the curation process."""
    model_config = ConfigDict(extra="forbid")

    curator: str = Field(min_length=1)
    curation_date: str = Field(min_length=1)
    version: str = "1.0"
    notes: str | None = None


class StereoMetadata(BaseModel):
    """Stereochemistry metadata for the reference target."""
    model_config = ConfigDict(extra="forbid")

    target_status: StereoStatus = StereoStatus.UNSPECIFIED
    relation_to_reagent: StereoRelation = StereoRelation.UNRESOLVED
    notes: str | None = None


class LiteratureReference(BaseModel):
    """A curated literature reference for a synthetic route."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    reference_id: str = Field(min_length=1)
    target: TargetRecord
    sources: list[SourceRecord] = Field(default_factory=list)
    route_scope: RouteScope = RouteScope.FULL_ROUTE
    route_completeness: RouteCompleteness = RouteCompleteness.COMPLETE
    molecules: list[ReferenceMolecule] = Field(default_factory=list)
    steps: list[ReferenceStep] = Field(default_factory=list)
    stereo_metadata: StereoMetadata = Field(default_factory=StereoMetadata)
    ambiguities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    curation: CurationMetadata | None = None
    is_composite: bool = False
    schema_version: Literal[1] = 1

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump(*args, **kwargs)

    @model_validator(mode="after")
    def validate_reference(self) -> LiteratureReference:
        self._validate_unique_ids()
        self._validate_graph_connectivity()
        self._validate_route_boundary()
        self._validate_composite_status()
        return self

    def _validate_unique_ids(self) -> None:
        mol_ids = [m.molecule_id for m in self.molecules]
        if len(mol_ids) != len(set(mol_ids)):
            raise ValueError("Duplicate molecule_ids in reference")
        step_ids = [s.step_id for s in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Duplicate step_ids in reference")

    def _validate_graph_connectivity(self) -> None:
        mol_ids = {m.molecule_id for m in self.molecules}
        for step in self.steps:
            for pid in step.precursor_ids:
                if pid not in mol_ids:
                    raise ValueError(f"Step {step.step_id} references unknown precursor {pid}")
            if step.product_id not in mol_ids:
                raise ValueError(f"Step {step.step_id} references unknown product {step.product_id}")

        target_mol = None
        for m in self.molecules:
            if m.role == MoleculeRole.TARGET:
                target_mol = m
                break
        if target_mol is None:
            raise ValueError("No target molecule declared")

        produced = {s.product_id for s in self.steps}
        consumed = set()
        for s in self.steps:
            consumed.update(s.precursor_ids)

        leaves = consumed - produced
        for leaf_id in leaves:
            mol = next(m for m in self.molecules if m.molecule_id == leaf_id)
            if mol.role not in (MoleculeRole.STARTING_MATERIAL, MoleculeRole.REAGENT, MoleculeRole.CATALYST, MoleculeRole.SOLVENT):
                raise ValueError(f"Graph leaf {leaf_id} is not declared as starting material")

        for mol in self.molecules:
            if mol.molecule_id != target_mol.molecule_id and mol.molecule_id not in leaves:
                if mol.molecule_id not in produced:
                    raise ValueError(f"Intermediate {mol.molecule_id} not produced by any step")

        visited = set()
        def visit(node_id: str, path: set[str]) -> None:
            if node_id in path:
                raise ValueError("Reference graph contains a cycle")
            if node_id in visited:
                return
            path.add(node_id)
            for step in self.steps:
                if step.product_id == node_id:
                    for pid in step.precursor_ids:
                        visit(pid, path.copy())
            visited.add(node_id)

        visit(target_mol.molecule_id, set())

    def _validate_route_boundary(self) -> None:
        if self.route_completeness == RouteCompleteness.COMPLETE:
            produced = {s.product_id for s in self.steps}
            consumed = set()
            for s in self.steps:
                consumed.update(s.precursor_ids)
            internal = produced & consumed
            for mol_id in internal:
                # Check both producing and consuming steps
                producing_step = next((s for s in self.steps if s.product_id == mol_id), None)
                consuming_steps = [s for s in self.steps if mol_id in s.precursor_ids]
                if producing_step is None or producing_step.inferred:
                    raise ValueError(f"Graph incomplete: internal intermediate {mol_id} lacks a documented producing step")
                for cs in consuming_steps:
                    if cs.inferred:
                        raise ValueError(f"Graph incomplete: internal intermediate {mol_id} consumed by inferred step {cs.step_id}")

    def _validate_composite_status(self) -> None:
        primary_sources = [s for s in self.sources if s.is_primary]
        if len(primary_sources) > 1:
            self.is_composite = True
        elif len(primary_sources) == 1:
            source_routes = defaultdict(list)
            for step in self.steps:
                for loc in step.evidence_locators:
                    source_routes[loc.source_id].append(step.step_id)
            primary_id = primary_sources[0].source_id
            if any(sid != primary_id for sid in source_routes.keys()):
                self.is_composite = True

    def get_target_molecule(self) -> ReferenceMolecule | None:
        for m in self.molecules:
            if m.role == MoleculeRole.TARGET:
                return m
        return None

    def get_starting_materials(self) -> list[ReferenceMolecule]:
        return [m for m in self.molecules if m.role == MoleculeRole.STARTING_MATERIAL]

    def is_exact_recovery_eligible(self) -> bool:
        """Check if this reference is eligible for exact whole-route recovery."""
        if any(s.inferred for s in self.steps):
            return False
        if self.route_completeness != RouteCompleteness.COMPLETE:
            return False
        if self.route_scope != RouteScope.FULL_ROUTE:
            return False
        return True

    def graph_complete(self) -> bool:
        """Check if the encoded route graph has no internal connectivity gaps.

        This validates the internal consistency of the recorded transformations,
        NOT the completeness of the primary literature source. A curator must
        still assert RouteCompleteness.COMPLETE based on source review.

        Returns True if:
        - Every internal intermediate has a producing step
        - No producing or consuming step for an internal intermediate is marked inferred
        - Target is reachable from starting materials
        - No cycles exist
        """
        # Reuse the validation logic but return bool instead of raising
        try:
            produced = {s.product_id for s in self.steps}
            consumed = set()
            for s in self.steps:
                consumed.update(s.precursor_ids)
            internal = produced & consumed
            for mol_id in internal:
                producing_step = next((s for s in self.steps if s.product_id == mol_id), None)
                consuming_steps = [s for s in self.steps if mol_id in s.precursor_ids]
                if producing_step is None or producing_step.inferred:
                    return False
                for cs in consuming_steps:
                    if cs.inferred:
                        return False
            return True
        except Exception:
            return False

    def to_deterministic_dict(self) -> dict:
        """Serialize to a deterministic dict for hashing.

        Ordering policy:
        - Dict keys: sorted (no semantic meaning)
        - ORDERED lists (preserve order):
          * steps - sequence encodes route topology
          * precursor_ids - order can distinguish electrophile/nucleophile
          * sources - primary source first is meaningful
          * evidence_locators - order can indicate priority
          * molecules - stable order by molecule_id
          * ambiguities, limitations - sorted for determinism (no semantic order)
          * reagents, catalysts, solvents - sorted (set semantics)
          * authors - sorted (citation convention varies)
        """
        data = self.model_dump(mode="json", by_alias=True)

        def canonicalize(obj):
            if isinstance(obj, dict):
                return {k: canonicalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                # Preserve order for semantically ordered fields; sort for unordered
                if not obj:
                    return []
                # Check if list contains dicts with molecule_id (molecules)
                if all(isinstance(x, dict) and "molecule_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("molecule_id", ""))]
                # Check if list contains dicts with step_id (steps)
                if all(isinstance(x, dict) and "step_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("step_id", ""))]
                # Check if list contains dicts with source_id (sources)
                if all(isinstance(x, dict) and "source_id" in x for x in obj):
                    # Primary sources first, then by source_id
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: (not d.get("is_primary", False), d.get("source_id", "")))]
                # Check if list contains dicts with precursor_ids (step dicts)
                if all(isinstance(x, dict) and "precursor_ids" in x for x in obj):
                    return [canonicalize(x) for x in obj]  # Preserve step order
                # Check if list contains strings that look like precursor IDs (precursor_ids)
                if all(isinstance(x, str) for x in obj) and obj and all(x.startswith("mol_") for x in obj):
                    return list(obj)  # Preserve precursor order
                # Check if list contains dicts with evidence locator fields
                if all(isinstance(x, dict) and "source_id" in x and "example" in x for x in obj):
                    return [canonicalize(x) for x in obj]  # Preserve locator order
                # For other string lists: sort for determinism
                if all(isinstance(x, str) for x in obj):
                    return sorted(obj)
                # For dict lists with no recognized ID key: sort by JSON representation
                if all(isinstance(x, dict) for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: json.dumps(d, sort_keys=True))]
                # Fallback: preserve order
                return [canonicalize(x) for x in obj]
            return obj

        return canonicalize(data)

    def content_hash(self) -> str:
        """SHA256 hash of deterministic serialization."""
        return hashlib.sha256(
            json.dumps(self.to_deterministic_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


class LiteratureReferenceSet(BaseModel):
    """A collection of literature references for benchmarking."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    references: list[LiteratureReference] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_ref_ids(self) -> LiteratureReferenceSet:
        ref_ids = [r.reference_id for r in self.references]
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError("Duplicate reference_ids in reference set")
        return self

    def to_deterministic_dict(self) -> dict:
        data = self.model_dump(mode="json", by_alias=True)

        def canonicalize(obj):
            if isinstance(obj, dict):
                return {k: canonicalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                if not obj:
                    return []
                if all(isinstance(x, dict) and "reference_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("reference_id", ""))]
                if all(isinstance(x, dict) and "molecule_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("molecule_id", ""))]
                if all(isinstance(x, dict) and "step_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("step_id", ""))]
                if all(isinstance(x, dict) and "source_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: (not d.get("is_primary", False), d.get("source_id", "")))]
                if all(isinstance(x, dict) and "precursor_ids" in x for x in obj):
                    return [canonicalize(x) for x in obj]
                if all(isinstance(x, dict) and "source_id" in x and "example" in x for x in obj):
                    return [canonicalize(x) for x in obj]
                if all(isinstance(x, str) for x in obj):
                    return sorted(obj)
                if all(isinstance(x, dict) for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: json.dumps(d, sort_keys=True))]
                return [canonicalize(x) for x in obj]
            return obj

        return canonicalize(data)

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_deterministic_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


def convert_legacy_reference(ref: LegacyReference) -> LiteratureReference:
    """Convert a legacy Reference from reference_review.py to LiteratureReference.

    This is a LOSSY conversion for backward compatibility only.
    The converted record is marked as legacy/incomplete and is NOT eligible
    for exact-recovery benchmarking because critical curation metadata is absent.
    """
    from reagent.core.models import Reaction as CoreReaction

    # Build molecules from reactions
    molecules_dict: dict[str, ReferenceMolecule] = {}
    steps: list[ReferenceStep] = []

    for i, reaction in enumerate(ref.reactions):
        if not isinstance(reaction, CoreReaction):
            reaction = CoreReaction.model_validate(reaction)

        product_id = f"mol_{len(molecules_dict)}"
        if reaction.product not in molecules_dict:
            molecules_dict[reaction.product] = ReferenceMolecule(
                molecule_id=product_id,
                reported_smiles=reaction.product,
                role=MoleculeRole.TARGET if i == 0 else MoleculeRole.INTERMEDIATE,
                structure_resolution=StructureResolutionStatus.NOT_REPORTED,
                stereo_status=StereoStatus.UNSPECIFIED,
            )

        precursor_ids = []
        for prec in reaction.precursors:
            prec_id = f"mol_{len(molecules_dict)}"
            if prec not in molecules_dict:
                molecules_dict[prec] = ReferenceMolecule(
                    molecule_id=prec_id,
                    reported_smiles=prec,
                    role=MoleculeRole.STARTING_MATERIAL,
                    structure_resolution=StructureResolutionStatus.NOT_REPORTED,
                    stereo_status=StereoStatus.UNSPECIFIED,
                )
            precursor_ids.append(molecules_dict[prec].molecule_id)

        steps.append(ReferenceStep(
            step_id=f"step_{i}",
            sequence_index=i,
            precursor_ids=precursor_ids,
            product_id=molecules_dict[reaction.product].molecule_id,
            evidence_locators=[EvidenceLocator(
                source_id=ref.source,
                curator_note=f"Converted from legacy reference: {ref.locator}; verified_by={ref.verified_by}"
            )],
            inferred=False,
        ))

    return LiteratureReference(
        reference_id=ref.source,
        target=TargetRecord(
            reagent_target_name="unknown",
            reagent_target_smiles=ref.reactions[0].product if ref.reactions else "",
        ),
        sources=[SourceRecord(
            source_id=ref.source,
            source_type=SourceType.OTHER,
            title=ref.locator,
            is_primary=True,
        )],
        route_scope=RouteScope.FULL_ROUTE,
        route_completeness=RouteCompleteness.AMBIGUOUS,  # Legacy data: completeness unknown
        molecules=list(molecules_dict.values()),
        steps=steps,
        stereo_metadata=StereoMetadata(
            target_status=StereoStatus.UNSPECIFIED,
            relation_to_reagent=StereoRelation.UNRESOLVED,
            notes="Legacy conversion: stereo relationship not curated"
        ),
        is_composite=False,
    )