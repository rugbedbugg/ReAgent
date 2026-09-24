"""Graded route similarity for literature recovery benchmark.

This module implements the Genheden-Shields route similarity metric using the
official `rxnutils` (reaction_utils) implementation. It provides adapters to
convert ReAgent's curated references and generated candidates into the format
expected by the similarity engine, and computes graded similarity scores.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import rxnutils.routes.base as base
import rxnutils.routes.comparison as comp
from rdkit import RDLogger

from reagent.core.chem import m0_key, m1_key
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.literature import (
    LiteratureReferenceSet,
)
from reagent.eval.literature_derived import (
    ExactEligibility,
)
from reagent.eval.literature_mapping import (
    MappedArtifactStore,
    MappedDerivedCandidate,
    MappedDerivedReference,
    MappedDerivedRoute,
    MapperCapability,
    MapperIdentity,
    MappingStatus,
    RouteMapper,
    RxnutilsRouteMapper,
    align_to_reference,
    map_synthesis_route,
    resolve_candidate,
    resolve_reference,
    validate_route_mapping,
)

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")


class SimilarityStatus(Enum):
    """Status of a similarity comparison."""
    SUCCESS = "success"
    MAPPING_FAILED = "mapping_failed"
    MAPPER_UNAVAILABLE = "mapper_unavailable"
    REFERENCE_INELIGIBLE = "reference_ineligible"
    CANDIDATE_CONVERSION_FAILED = "candidate_conversion_failed"
    REFERENCE_CONVERSION_FAILED = "reference_conversion_failed"
    TARGET_MISMATCH = "target_mismatch"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"


def _pair_status(status: MappingStatus, side: str | None) -> tuple[SimilarityStatus, str]:
    """Pair status for a comparison the mapping layer could not align."""
    if status == MappingStatus.TARGET_MISMATCH:
        return SimilarityStatus.TARGET_MISMATCH, "root_compounds_differ"
    if status == MappingStatus.MAPPER_UNAVAILABLE:
        return SimilarityStatus.MAPPER_UNAVAILABLE, "no_mapper_available"
    if status == MappingStatus.STALE:
        return SimilarityStatus.MAPPER_UNAVAILABLE, "stale_mapping"
    if status == MappingStatus.MAPPING_FAILED:
        return SimilarityStatus.MAPPING_FAILED, "mapping_failed"
    if side == "candidate":
        return SimilarityStatus.CANDIDATE_CONVERSION_FAILED, "candidate_conversion_failed"
    return SimilarityStatus.REFERENCE_CONVERSION_FAILED, "reference_conversion_failed"


@dataclass
class SimilarityPairResult:
    """Result of comparing one reference against one candidate."""
    reference_id: str
    candidate_route_id: str
    candidate_content_hash: str
    reagent_rank: int | None
    baseline_rank: int | None

    atom_similarity: float | None
    bond_similarity: float | None
    route_similarity: float | None

    status: SimilarityStatus
    mapping_status: str
    warnings: list[str]


@dataclass
class TargetSimilarityResult:
    """Per-target aggregated similarity results."""
    target_name: str
    target_smiles: str
    target_m0: str
    target_m1: str
    retained_candidates: int
    solved_candidates: int
    eligible_reference_ids: list[str]  # CORE_EXACT_ELIGIBLE + STEREO_STRESS
    excluded_reference_ids: list[str]

    # Similarity metrics for the selected (rank 1) candidate
    selected_route_similarity: float | None
    selected_atom_similarity: float | None
    selected_bond_similarity: float | None

    # Best similarity across all retained candidates
    max_similarity_retained: float | None
    max_atom_similarity_retained: float | None
    max_bond_similarity_retained: float | None
    best_similarity_rank: int | None

    # Baseline (feasibility-ranked) selected candidate similarity
    baseline_selected_similarity: float | None
    baseline_selected_atom_similarity: float | None
    baseline_selected_bond_similarity: float | None

    # Per-reference pairwise results
    reference_matches: list  # List[SimilarityPairResult]

    # Target evaluability
    similarity_evaluable: bool = True
    mapping_failed_count: int = 0
    mapper_unavailable_count: int = 0


@dataclass
class SimilarityMetrics:
    """Aggregate similarity metrics across targets."""
    n_targets_requested: int
    n_targets_with_graded_reference: int      # CORE + STEREO_STRESS
    n_targets_similarity_evaluable: int       # Has graded ref + solved candidates + successful comparison
    n_targets_mapping_failed: int             # Has graded ref + solved candidates but ALL comparisons failed mapping
    n_targets_mapper_unavailable: int         # Has graded ref + solved candidates but ALL comparisons mapper unavailable

    # None when no target is similarity-evaluable; never zero-filled
    mean_selected_route_similarity: float | None
    median_selected_route_similarity: float | None
    mean_max_similarity_retained: float | None
    median_max_similarity_retained: float | None

    mean_selected_atom_similarity: float | None
    median_selected_atom_similarity: float | None
    mean_selected_bond_similarity: float | None
    median_selected_bond_similarity: float | None

    per_target: list = None
    schema_version: Literal[1] = 1

    def __post_init__(self):
        if self.per_target is None:
            self.per_target = []


class SimilarityProvenance:
    """Provenance information for a similarity evaluation run."""

    def __init__(
        self,
        metric_name: str,
        metric_reference_doi: str,
        metric_implementation: str,
        reaction_utils_version: str,
        reaction_utils_module: str,
        adapter_version: str,
        rdkit_version: str,
        mapping_tool: str,
        mapping_tool_version: str,
        reference_set_content_hash: str,
        checkpoint_manifest_digest: str,
        ranking_implementation_hash: str,
        similarity_evaluator_hash: str,
    ):
        self.metric_name = metric_name
        self.metric_reference_doi = metric_reference_doi
        self.metric_implementation = metric_implementation
        self.reaction_utils_version = reaction_utils_version
        self.reaction_utils_module = reaction_utils_module
        self.adapter_version = adapter_version
        self.rdkit_version = rdkit_version
        self.mapping_tool = mapping_tool
        self.mapping_tool_version = mapping_tool_version
        self.reference_set_content_hash = reference_set_content_hash
        self.checkpoint_manifest_digest = checkpoint_manifest_digest
        self.ranking_implementation_hash = ranking_implementation_hash
        self.similarity_evaluator_hash = similarity_evaluator_hash

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "metric_reference_doi": self.metric_reference_doi,
            "metric_implementation": self.metric_implementation,
            "reaction_utils_version": self.reaction_utils_version,
            "reaction_utils_module": self.reaction_utils_module,
            "adapter_version": self.adapter_version,
            "rdkit_version": self.rdkit_version,
            "mapping_tool": self.mapping_tool,
            "mapping_tool_version": self.mapping_tool_version,
            "reference_set_content_hash": self.reference_set_content_hash,
            "checkpoint_manifest_digest": self.checkpoint_manifest_digest,
            "ranking_implementation_hash": self.ranking_implementation_hash,
            "similarity_evaluator_hash": self.similarity_evaluator_hash,
        }


class SimilarityEvaluator:
    """Evaluates graded route similarity for literature recovery benchmark."""

    ADAPTER_VERSION = "1.0"
    METRIC_NAME = "Genheden-Shields"
    METRIC_REFERENCE_DOI = "10.1039/D4DD00292J"
    METRIC_IMPLEMENTATION = "rxnutils.routes.comparison.simple_route_similarity"

    def __init__(
        self,
        reference_set: LiteratureReferenceSet,
        derivation_generator_hash: str,
        mapper: RouteMapper | None = None,
        mapped_references: dict[str, MappedDerivedReference] | None = None,
        artifact_store: MappedArtifactStore | None = None,
        stored_maps_identity: MapperIdentity | None = None,
    ):
        """Mapping comes only from the shared mapping layer.

        ``mapped_references`` supplies artifacts produced elsewhere (for example in
        a dedicated mapping environment); stale ones are rejected, never used.
        ``artifact_store`` caches artifacts on disk. ``stored_maps_identity``
        records what produced maps saved on candidate trees; by default it is
        read from the checkpoint manifest.
        """
        self.reference_set = reference_set
        self.derivation_generator_hash = derivation_generator_hash
        self.mapper = mapper if mapper is not None else RxnutilsRouteMapper()
        self.mapped_references = dict(mapped_references or {})
        self.artifact_store = artifact_store
        self.stored_maps_identity = stored_maps_identity
        self._derived_refs = []
        # Already-mapped forward-form reference routes supplied directly by a caller
        self._reference_routes: dict[str, base.SynthesisRoute] = {}
        self._reference_artifacts: dict[str, MappedDerivedRoute | None] = {}

    def _derive_all_references(self):
        if self._derived_refs:
            return self._derived_refs

        from reagent.eval.literature_derived import derive_reference

        derived = []
        for ref in self.reference_set.references:
            derived_ref = derive_reference(ref, self.derivation_generator_hash)
            derived.append(derived_ref)
        self._derived_refs = derived
        return derived

    def _reference_artifact(self, derived_ref) -> MappedDerivedRoute | None:
        """The reference's mapped artifact, or None if it is not graded-eligible."""
        rid = derived_ref.reference_id
        if rid in self._reference_routes:
            return map_synthesis_route(self._reference_routes[rid], "reference", rid,
                                       MapperIdentity("caller-supplied", "unrecorded"), self.mapper)
        if rid not in self._reference_artifacts:
            original = next((r for r in self.reference_set.references if r.reference_id == rid), None)
            # Only CORE_EXACT_ELIGIBLE (M0 target) and STEREO_STRESS (M1 target) are graded
            if original is None or derived_ref.exact_eligibility not in (
                ExactEligibility.CORE_EXACT_ELIGIBLE,
                ExactEligibility.STEREO_STRESS,
            ):
                self._reference_artifacts[rid] = None
            else:
                self._reference_artifacts[rid] = resolve_reference(
                    original, derived_ref, self.mapper, self.artifact_store, self.mapped_references.get(rid),
                )
        return self._reference_artifacts[rid]

    def _get_reference_route(self, derived_ref) -> base.SynthesisRoute | None:
        """rxnutils view of the reference artifact: mapped if mapping succeeded."""
        artifact = self._reference_artifact(derived_ref)
        return artifact.synthesis_route() if artifact is not None else None

    def _check_mapper_availability(self) -> bool:
        return self.mapper.capability() == MapperCapability.AVAILABLE

    def _has_valid_atom_mappings(self, route: base.SynthesisRoute) -> bool:
        """True if the route carries a mapping that passes route-wide validation."""
        return not validate_route_mapping(route.reaction_tree)

    def _candidate_artifact(self, route_or_cand) -> MappedDerivedCandidate:
        """Mapped artifact of a generated Route (original tree first) or a DerivedCandidate."""
        from reagent.core.models import Route
        from reagent.eval.literature_derived import derive_candidate

        if isinstance(route_or_cand, Route):
            derived = derive_candidate(route_or_cand, self.derivation_generator_hash)
        else:
            derived = route_or_cand
        return resolve_candidate(route_or_cand, derived, self.mapper, self.artifact_store,
                                 self.stored_maps_identity)

    def _candidate_or_none(self, route) -> MappedDerivedCandidate | None:
        artifact = self._candidate_artifact(route)
        return artifact if artifact.route_tree is not None else None

    def _build_candidate_route(self, route_or_cand) -> base.SynthesisRoute | None:
        """rxnutils view of the candidate artifact, or None without a usable topology."""
        return self._candidate_artifact(route_or_cand).synthesis_route()

    def _compare_pair(
        self,
        ref,
        cand_route,
        candidate_route_id: str,
        candidate_content_hash: str,
        reagent_rank,
        baseline_rank,
    ):
        """Compare a reference against one candidate through the shared mapping layer.

        Args:
            ref: DerivedReference object
            cand_route: the candidate's MappedDerivedCandidate, or an already-built
                forward-form SynthesisRoute, or None if it could not be converted
            candidate_route_id: Identifier for the candidate route
            candidate_content_hash: Content hash for provenance
            reagent_rank: Rank in reagent ranking (or None)
            baseline_rank: Rank in baseline ranking (or None)
        """
        def result(status, mapping_status, notes, atom=None, bond=None, total=None):
            return SimilarityPairResult(
                reference_id=ref.reference_id,
                candidate_route_id=candidate_route_id,
                candidate_content_hash=candidate_content_hash,
                reagent_rank=reagent_rank,
                baseline_rank=baseline_rank,
                atom_similarity=atom,
                bond_similarity=bond,
                route_similarity=total,
                status=status,
                mapping_status=mapping_status,
                warnings=notes,
            )

        ref_artifact = self._reference_artifact(ref)
        if ref_artifact is None or ref_artifact.route_tree is None:
            reason = ref_artifact.failure_reason if ref_artifact is not None else ref.exclusion_reasons
            return result(SimilarityStatus.REFERENCE_CONVERSION_FAILED, "reference_conversion_failed",
                          [f"Reference conversion failed: {reason}"])

        if isinstance(cand_route, base.SynthesisRoute):
            cand_route = map_synthesis_route(cand_route, "candidate", candidate_route_id,
                                             MapperIdentity("caller-supplied", "unrecorded"), self.mapper)
        if cand_route is None or cand_route.route_tree is None:
            return result(SimilarityStatus.CANDIDATE_CONVERSION_FAILED, "candidate_conversion_failed",
                          ["Candidate route is None"])

        try:
            alignment = align_to_reference(ref_artifact, cand_route)
        except Exception as e:
            warnings.warn(f"Target alignment failed: {e}")
            return result(SimilarityStatus.MAPPING_FAILED, f"exception: {e}", [f"Target alignment raised: {e}"])
        if alignment.status != MappingStatus.SUCCESS:
            status, mapping_status = _pair_status(alignment.status, alignment.side)
            return result(status, mapping_status, [alignment.reason])

        # Both routes now share the reference target's atom numbering
        ref_route, cand_route = alignment.reference_route, alignment.candidate_route
        try:
            sim_matrix = comp.simple_route_similarity([ref_route, cand_route])
            if sim_matrix is None or sim_matrix.size == 0:
                return result(SimilarityStatus.MAPPING_FAILED, "similarity_computation_failed",
                              ["similarity computation returned empty matrix"])

            if sim_matrix.shape == (2, 2):
                route_sim = float(sim_matrix[0, 1])
            else:
                route_sim = None

            bond_sim = comp.simple_bond_forming_similarity([ref_route, cand_route])
            atom_sim = comp.atom_matching_bonanza_similarity([ref_route, cand_route])

            bond_val = float(bond_sim[0, 1]) if bond_sim is not None and bond_sim.shape == (2, 2) else None
            atom_val = float(atom_sim[0, 1]) if atom_sim is not None and atom_sim.shape == (2, 2) else None

            return result(SimilarityStatus.SUCCESS, "success", [], atom=atom_val, bond=bond_val, total=route_sim)
        except Exception as e:
            warnings.warn(f"Similarity computation failed: {e}")
            return result(SimilarityStatus.MAPPING_FAILED, f"exception: {str(e)}",
                          [f"Similarity computation raised: {e}"])

    def evaluate(
        self,
        candidate_checkpoint: Checkpoint,
    ):
        """Evaluate graded route similarity for all targets in the reference set.

        This computes Genheden-Shields similarity metrics (total, atom, bond)
        for each reference-candidate pair, aggregates per target, and reports
        aggregate statistics. Does NOT compute exact recovery metrics (Recovery@k)
        which are the domain of Phase 2.

        Target grouping uses explicit target association from reference records
        (reagent_target_name + reagent_target_smiles), not M0 grouping.
        This ensures stereo-stress references (M1-compatible) are not pre-filtered.
        """
        if self.stored_maps_identity is None:
            self.stored_maps_identity = _stored_maps_identity(candidate_checkpoint)
        # Prime the lazy derivation cache. Everything below reads
        # self._derived_refs directly, so without this the cache stays empty and
        # every reference silently drops out before eligibility is ever assessed.
        self._derive_all_references()

        # Group derived references by explicit target association from reference records
        # Key: (reagent_target_name, reagent_target_smiles) tuple
        refs_by_target = {}
        for ref in self.reference_set.references:
            target_key = (ref.target.reagent_target_name, ref.target.reagent_target_smiles)
            refs_by_target.setdefault(target_key, []).append(ref)

        per_target_results = []

        for (target_name, target_smiles), refs in refs_by_target.items():
            target_m0 = m0_key(target_smiles) or ""
            target_m1 = m1_key(target_smiles) or ""

            # Find derived refs for these references
            derived_refs_for_target = [
                dref for dref in self._derived_refs
                if dref.reference_id in [r.reference_id for r in refs]
            ]

            # Load candidates from checkpoint using target M0
            result = candidate_checkpoint.load(target_m0)
            if result is None:
                per_target_results.append(TargetSimilarityResult(
                    target_name=target_name,
                    target_smiles=target_smiles,
                    target_m0=target_m0,
                    target_m1=target_m1,
                    retained_candidates=0,
                    solved_candidates=0,
                    eligible_reference_ids=[],
                    excluded_reference_ids=[r.reference_id for r in derived_refs_for_target],
                    selected_route_similarity=None,
                    selected_atom_similarity=None,
                    selected_bond_similarity=None,
                    max_similarity_retained=None,
                    max_atom_similarity_retained=None,
                    max_bond_similarity_retained=None,
                    best_similarity_rank=None,
                    baseline_selected_similarity=None,
                    baseline_selected_atom_similarity=None,
                    baseline_selected_bond_similarity=None,
                    reference_matches=[],
                    similarity_evaluable=False,
                    mapping_failed_count=0,
                ))
                continue

            routes = result.routes
            retained_candidates = len(routes)
            solved_routes = [r for r in routes if r.solved]
            solved_candidates = len(solved_routes)

            # Separate references by eligibility
            core_refs = [r for r in derived_refs_for_target if r.exact_eligibility == ExactEligibility.CORE_EXACT_ELIGIBLE]
            stereo_stress_refs = [r for r in derived_refs_for_target if r.exact_eligibility == ExactEligibility.STEREO_STRESS]
            excluded_refs = [r for r in derived_refs_for_target if r.exact_eligibility == ExactEligibility.EXCLUDED]
            graded_refs = core_refs + stereo_stress_refs  # Both are similarity-evaluable

            if not solved_routes:
                per_target_results.append(TargetSimilarityResult(
                    target_name=target_name,
                    target_smiles=target_smiles,
                    target_m0=target_m0,
                    target_m1=target_m1,
                    retained_candidates=retained_candidates,
                    solved_candidates=0,
                    eligible_reference_ids=[r.reference_id for r in graded_refs],
                    excluded_reference_ids=[r.reference_id for r in excluded_refs],
                    selected_route_similarity=None,
                    selected_atom_similarity=None,
                    selected_bond_similarity=None,
                    max_similarity_retained=None,
                    max_atom_similarity_retained=None,
                    max_bond_similarity_retained=None,
                    best_similarity_rank=None,
                    baseline_selected_similarity=None,
                    baseline_selected_atom_similarity=None,
                    baseline_selected_bond_similarity=None,
                    reference_matches=[],
                    similarity_evaluable=False,
                    mapping_failed_count=0,
                ))
                continue

            # Rank solved candidates
            from reagent.eval.harness import rank_candidates
            ranking = rank_candidates(solved_routes)
            reagent_ranked = ranking.reagent_ranked
            baseline_ranked = ranking.baseline_ranked

            # Build SynthesisRoute objects for all solved candidates (reagent ranking).
            # A candidate that cannot be converted keeps its rank and is reported,
            # so no later candidate is promoted into its slot.
            reagent_built = [self._candidate_or_none(r) for r in reagent_ranked]

            if all(c is None for c in reagent_built):
                per_target_results.append(TargetSimilarityResult(
                    target_name=target_name,
                    target_smiles=target_smiles,
                    target_m0=target_m0,
                    target_m1=target_m1,
                    retained_candidates=retained_candidates,
                    solved_candidates=solved_candidates,
                    eligible_reference_ids=[r.reference_id for r in graded_refs],
                    excluded_reference_ids=[r.reference_id for r in excluded_refs],
                    selected_route_similarity=None,
                    selected_atom_similarity=None,
                    selected_bond_similarity=None,
                    max_similarity_retained=None,
                    max_atom_similarity_retained=None,
                    max_bond_similarity_retained=None,
                    best_similarity_rank=None,
                    baseline_selected_similarity=None,
                    baseline_selected_atom_similarity=None,
                    baseline_selected_bond_similarity=None,
                    reference_matches=[],
                    similarity_evaluable=False,
                    mapping_failed_count=0,
                ))
                continue

            baseline_selected = self._candidate_or_none(baseline_ranked[0])

            # Compare each graded reference against all valid candidates
            reference_matches = []
            mapping_failed_count = 0
            mapper_unavailable_count = 0

            # Pre-build DerivedCandidates for content hashes
            # We need these for provenance tracking
            from reagent.eval.literature_derived import derive_candidate
            derived_cands_for_hash = [derive_candidate(r, self.derivation_generator_hash) for r in reagent_ranked]
            baseline_derived_cands = [derive_candidate(r, self.derivation_generator_hash) for r in baseline_ranked]

            for ref in graded_refs:
                is_stereo_stress = ref.exact_eligibility == ExactEligibility.STEREO_STRESS

                ref_artifact = self._reference_artifact(ref)
                if ref_artifact is None or ref_artifact.route_tree is None:
                    continue

                # For STEREO_STRESS, verify M1 target compatibility before comparing
                if is_stereo_stress:
                    if ref.target_m0 and target_m0:
                        ref_m1 = m1_key(ref.target_m0) if hasattr(ref, 'target_m0') and ref.target_m0 else None
                        if ref_m1 and ref_m1 != target_m1:
                            continue

                # Compare against reagent-ranked candidates
                best_sim_for_ref = None

                for rank, (cand_route, cand_derived) in enumerate(zip(reagent_built, derived_cands_for_hash), 1):
                    comp_result = self._compare_pair(
                        ref, cand_route,
                        candidate_route_id=cand_derived.route_id,
                        candidate_content_hash=cand_derived.content_hash(),
                        reagent_rank=rank, baseline_rank=None
                    )
                    reference_matches.append(comp_result)

                    if comp_result.status == SimilarityStatus.MAPPING_FAILED:
                        mapping_failed_count += 1
                    elif comp_result.status == SimilarityStatus.MAPPER_UNAVAILABLE:
                        mapper_unavailable_count += 1
                    elif comp_result.status == SimilarityStatus.SUCCESS and comp_result.route_similarity is not None:
                        if best_sim_for_ref is None or comp_result.route_similarity > best_sim_for_ref:
                            best_sim_for_ref = comp_result.route_similarity

                # Also compare against baseline-selected candidate (rank 1 in baseline)
                baseline_sim = None
                baseline_atom = None
                baseline_bond = None
                if baseline_ranked:
                    cand_derived = baseline_derived_cands[0]
                    comp_result = self._compare_pair(
                        ref, baseline_selected,
                        candidate_route_id=cand_derived.route_id,
                        candidate_content_hash=cand_derived.content_hash(),
                        reagent_rank=None, baseline_rank=1
                    )
                    reference_matches.append(comp_result)
                    if comp_result.status == SimilarityStatus.SUCCESS and comp_result.route_similarity is not None:
                        baseline_sim = comp_result.route_similarity
                        baseline_atom = comp_result.atom_similarity
                        baseline_bond = comp_result.bond_similarity
                    elif comp_result.status == SimilarityStatus.MAPPING_FAILED:
                        mapping_failed_count += 1
                    elif comp_result.status == SimilarityStatus.MAPPER_UNAVAILABLE:
                        mapper_unavailable_count += 1

            # Aggregate per-target metrics
            # Selected route similarity = similarity of rank 1 (reagent) candidate
            selected_sim = None
            selected_atom = None
            selected_bond = None
            if reagent_built:
                # Find the best similarity for the rank 1 candidate across all references
                rank1_sims = [m.route_similarity for m in reference_matches
                              if m.reagent_rank == 1 and m.status == SimilarityStatus.SUCCESS and m.route_similarity is not None]
                rank1_atoms = [m.atom_similarity for m in reference_matches
                               if m.reagent_rank == 1 and m.status == SimilarityStatus.SUCCESS and m.atom_similarity is not None]
                rank1_bonds = [m.bond_similarity for m in reference_matches
                               if m.reagent_rank == 1 and m.status == SimilarityStatus.SUCCESS and m.bond_similarity is not None]
                if rank1_sims:
                    selected_sim = max(rank1_sims)
                    selected_atom = max(rank1_atoms) if rank1_atoms else None
                    selected_bond = max(rank1_bonds) if rank1_bonds else None

            # Max similarity retained = best similarity across all retained candidates and all references
            all_sims = [m.route_similarity for m in reference_matches
                        if m.status == SimilarityStatus.SUCCESS and m.route_similarity is not None]
            all_atoms = [m.atom_similarity for m in reference_matches
                         if m.status == SimilarityStatus.SUCCESS and m.atom_similarity is not None]
            all_bonds = [m.bond_similarity for m in reference_matches
                         if m.status == SimilarityStatus.SUCCESS and m.bond_similarity is not None]
            max_sim = max(all_sims) if all_sims else None
            max_atom = max(all_atoms) if all_atoms else None
            max_bond = max(all_bonds) if all_bonds else None
            best_rank = None
            if max_sim is not None:
                # Best ReAgent rank among candidates achieving max_sim. Every solved
                # route has a ReAgent rank, so baseline-only pairs are not needed.
                best_rank = min((m.reagent_rank for m in reference_matches
                                 if m.status == SimilarityStatus.SUCCESS and m.route_similarity == max_sim
                                 and m.reagent_rank is not None), default=None)

            # Baseline selected similarity (rank 1 in baseline ranking)
            baseline_sim = None
            baseline_atom = None
            baseline_bond = None
            if baseline_ranked:
                baseline_sims = [m.route_similarity for m in reference_matches
                                 if m.baseline_rank == 1 and m.status == SimilarityStatus.SUCCESS and m.route_similarity is not None]
                baseline_atoms = [m.atom_similarity for m in reference_matches
                                  if m.baseline_rank == 1 and m.status == SimilarityStatus.SUCCESS and m.atom_similarity is not None]
                baseline_bonds = [m.bond_similarity for m in reference_matches
                                  if m.baseline_rank == 1 and m.status == SimilarityStatus.SUCCESS and m.bond_similarity is not None]
                if baseline_sims:
                    baseline_sim = max(baseline_sims)
                    baseline_atom = max(baseline_atoms) if baseline_atoms else None
                    baseline_bond = max(baseline_bonds) if baseline_bonds else None

            # Target is similarity-evaluable if at least one graded reference had a successful comparison
            similarity_evaluable = any(m.status == SimilarityStatus.SUCCESS for m in reference_matches)

            per_target_results.append(TargetSimilarityResult(
                target_name=target_name,
                target_smiles=target_smiles,
                target_m0=target_m0,
                target_m1=target_m1,
                retained_candidates=retained_candidates,
                solved_candidates=solved_candidates,
                eligible_reference_ids=[r.reference_id for r in graded_refs],
                excluded_reference_ids=[r.reference_id for r in excluded_refs],
                selected_route_similarity=selected_sim,
                selected_atom_similarity=selected_atom,
                selected_bond_similarity=selected_bond,
                max_similarity_retained=max_sim,
                max_atom_similarity_retained=max_atom,
                max_bond_similarity_retained=max_bond,
                best_similarity_rank=best_rank,
                baseline_selected_similarity=baseline_sim,
                baseline_selected_atom_similarity=baseline_atom,
                baseline_selected_bond_similarity=baseline_bond,
                reference_matches=reference_matches,
                similarity_evaluable=similarity_evaluable,
                mapping_failed_count=mapping_failed_count,
                mapper_unavailable_count=mapper_unavailable_count,
            ))

        # Aggregate across targets
        # The unit of aggregation is the target, not the reference
        n_requested = len(refs_by_target)
        n_with_graded_ref = len([t for t in per_target_results if t.eligible_reference_ids])
        n_evaluable = len([t for t in per_target_results if t.similarity_evaluable])
        n_mapping_failed = len([t for t in per_target_results if t.mapping_failed_count > 0 and t.mapper_unavailable_count == 0])
        n_mapper_unavailable = len([t for t in per_target_results if t.mapper_unavailable_count > 0])

        def mean_attr(attr):
            targets = [t for t in per_target_results if t.similarity_evaluable and getattr(t, attr) is not None]
            if not targets:
                return None
            return sum(getattr(t, attr) for t in targets) / len(targets)

        def median_attr(attr):
            vals = sorted(getattr(t, attr) for t in per_target_results
                          if t.similarity_evaluable and getattr(t, attr) is not None)
            if not vals:
                return None
            n = len(vals)
            if n % 2 == 1:
                return vals[n // 2]
            return (vals[n // 2 - 1] + vals[n // 2]) / 2

        metrics = SimilarityMetrics(
            n_targets_requested=n_requested,
            n_targets_with_graded_reference=n_with_graded_ref,
            n_targets_similarity_evaluable=n_evaluable,
            n_targets_mapping_failed=n_mapping_failed,
            n_targets_mapper_unavailable=n_mapper_unavailable,
            mean_selected_route_similarity=mean_attr("selected_route_similarity"),
            median_selected_route_similarity=median_attr("selected_route_similarity"),
            mean_max_similarity_retained=mean_attr("max_similarity_retained"),
            median_max_similarity_retained=median_attr("max_similarity_retained"),
            mean_selected_atom_similarity=mean_attr("selected_atom_similarity"),
            median_selected_atom_similarity=median_attr("selected_atom_similarity"),
            mean_selected_bond_similarity=mean_attr("selected_bond_similarity"),
            median_selected_bond_similarity=median_attr("selected_bond_similarity"),
            per_target=per_target_results,
        )

        return metrics


def _stored_maps_identity(checkpoint: Checkpoint) -> MapperIdentity:
    """Attribute maps saved on candidate trees to the AiZynthFinder that searched."""
    try:
        manifest = json.loads((Path(checkpoint.directory) / "manifest.json").read_text(encoding="utf-8"))
        version = manifest.get("dependencies", {}).get("aizynthfinder") or "unrecorded"
    except (OSError, ValueError, AttributeError):
        version = "unrecorded"
    return MapperIdentity("aizynthfinder-template-application", f"aizynthfinder=={version}")


def run_similarity_benchmark(
    reference_set: LiteratureReferenceSet,
    candidate_checkpoint: Checkpoint,
    derivation_generator_hash: str | None = None,
):
    if derivation_generator_hash is None:
        derivation_generator_hash = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()[:16]

    evaluator = SimilarityEvaluator(reference_set, derivation_generator_hash)
    return evaluator.evaluate(candidate_checkpoint)
