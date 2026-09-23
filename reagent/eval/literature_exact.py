"""Exact graph comparison and recovery metrics for literature benchmark.

This module implements the primary exact-recovery evaluation:
- Graph-exact route comparison (topology + M0 molecule identity)
- Deterministic ranking of solved retained candidates
- Recovery@k metrics with proper denominators
- Multiple reference aggregation
- Provenance tracking
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from reagent.core.chem import m0_key
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.harness import (
    DEFAULT_WEIGHTS,
    rank_candidates,
)
from reagent.eval.literature import (
    LiteratureReferenceSet,
    RouteCompleteness,
    RouteScope,
    StereoRelation,
)
from reagent.eval.literature_derived import (
    DerivedCandidate,
    DerivedReference,
    ExactEligibility,
    derive_candidate,
    derive_reference,
)


class GraphEquality(str, Enum):
    EXACT = "exact"
    NOT_EXACT = "not_exact"
    INELIGIBLE = "ineligible"


class GraphComparisonResult(BaseModel):
    """Result of comparing two derived routes for graph-exact equality."""
    model_config = ConfigDict(extra="forbid")

    equality: GraphEquality
    target_m0_match: bool
    graph_signature_match: bool
    details: dict = Field(default_factory=dict)


def compare_graph_exact(
    ref: DerivedReference,
    cand: DerivedCandidate,
) -> GraphComparisonResult:
    """
    Compare a derived reference and candidate for graph-exact equality.

    Requirements for EXACT:
    1. Same strict M0 target
    2. Same transformation topology (graph_signature)
    3. Both eligible for comparison
    """
    if ref.exact_eligibility == ExactEligibility.EXCLUDED:
        return GraphComparisonResult(
            equality=GraphEquality.INELIGIBLE,
            target_m0_match=False,
            graph_signature_match=False,
            details={"ref_exclusion": ref.exclusion_reasons},
        )

    if cand.graph_signature is None:
        return GraphComparisonResult(
            equality=GraphEquality.NOT_EXACT,
            target_m0_match=ref.target_m0 == cand.target_m0,
            graph_signature_match=False,
            details={"cand_errors": cand.derivation_errors},
        )

    target_match = ref.target_m0 == cand.target_m0
    signature_match = ref.graph_signature == cand.graph_signature

    return GraphComparisonResult(
        equality=GraphEquality.EXACT if (target_match and signature_match) else GraphEquality.NOT_EXACT,
        target_m0_match=target_match,
        graph_signature_match=signature_match,
        details={},
    )


class ExactReferenceMatch(BaseModel):
    """Record of an exact match between reference and candidate."""
    model_config = ConfigDict(extra="forbid")

    reference_id: str
    candidate_route_id: str
    reagent_rank: int | None
    baseline_rank: int | None
    is_core_exact: bool


class TargetRecoveryResult(BaseModel):
    """Per-target exact recovery result."""
    model_config = ConfigDict(extra="forbid")

    target_name: str
    target_smiles: str
    retained_candidates: int
    solved_candidates: int
    eligible_reference_ids: list[str]
    excluded_reference_ids: list[str]
    exact_recovered: bool
    best_exact_rank: int | None
    baseline_best_exact_rank: int | None
    recovery_at_1: bool
    recovery_at_3: bool
    recovery_at_5: bool
    recovery_at_10: bool
    recovery_at_retained: bool
    unsolved_exact_match_present: bool
    reference_matches: list[ExactReferenceMatch] = Field(default_factory=list)


class ExactRecoveryMetrics(BaseModel):
    """Aggregate exact recovery metrics across targets."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    n_targets_requested: int
    n_targets_with_core_exact_reference: int
    n_targets_without_core_exact_reference: int
    n_targets_solved: int
    recovery_at_1: float
    recovery_at_3: float
    recovery_at_5: float
    recovery_at_10: float
    recovery_at_retained: float
    per_target: list[TargetRecoveryResult] = Field(default_factory=list)


class RankingProvenance(BaseModel):
    """Provenance for deterministic ranking."""
    model_config = ConfigDict(extra="forbid")

    weight_vector: dict[str, float]
    ranking_implementation_hash: str
    evaluator_implementation_hash: str
    rdkit_version: str
    numpy_version: str


class ExactBenchmarkProvenance(BaseModel):
    """Full provenance for an exact benchmark run."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    literature_benchmark_version: str = "1.0"
    reference_set_content_hash: str
    derivation_schema_version: int
    candidate_checkpoint_path: str
    checkpoint_manifest_digest: str
    ranking_provenance: RankingProvenance
    evaluator_code_hash: str


class ExactBenchmarkResult(BaseModel):
    """Complete exact benchmark result with provenance."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    provenance: ExactBenchmarkProvenance
    metrics: ExactRecoveryMetrics
    reagent_weight_vector: dict[str, float]
    baseline_weight_vector: dict[str, float]


# --- Deterministic Ranking Helpers ---
# Use shared ranking functions from harness:
# rank_routes_deterministic, rank_baseline_feasibility, rank_candidates, RankingResult


def get_implementation_hashes() -> dict[str, str]:
    """Get hashes of key implementation files for provenance."""
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for folder in ("eval", "features", "optimize", "core"):
        for p in sorted((root / folder).glob("*.py")):
            rel = p.relative_to(root)
            hashes[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def get_derivation_generator_hash() -> str:
    """Hash of the derivation generator code."""
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_exact_recovery_benchmark(
    reference_set: LiteratureReferenceSet,
    candidate_checkpoint_path: Path,
    derivation_generator_hash: str | None = None,
) -> ExactBenchmarkResult:
    """
    Run the complete exact recovery benchmark.

    Args:
        reference_set: The curated literature references
        candidate_checkpoint_path: Path to checkpoint directory with generated routes
        derivation_generator_hash: Optional hash of derivation code (defaults to this module)

    Returns:
        ExactBenchmarkResult with metrics and provenance
    """
    import numpy
    import rdkit

    if derivation_generator_hash is None:
        derivation_generator_hash = get_derivation_generator_hash()

    # Load checkpoint
    checkpoint = Checkpoint.open_readonly(candidate_checkpoint_path)
    manifest = json.loads((candidate_checkpoint_path / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()

    # Derive references
    derived_refs: list[DerivedReference] = []
    for ref in reference_set.references:
        derived_refs.append(derive_reference(ref, derivation_generator_hash))

    # Group by target
    refs_by_target: dict[str, list[DerivedReference]] = {}
    for dref in derived_refs:
        # Use target M0 as grouping key
        refs_by_target.setdefault(dref.target_m0, []).append(dref)

    # Process each target
    per_target_results: list[TargetRecoveryResult] = []

    for target_m0, refs in refs_by_target.items():
        # Find target name from original references
        target_name = ""
        target_smiles = ""
        for ref in reference_set.references:
            if m0_key(ref.target.reagent_target_smiles) == target_m0:
                target_name = ref.target.reagent_target_name
                target_smiles = ref.target.reagent_target_smiles
                break

        # Load candidates from checkpoint
        result = checkpoint.load(target_m0)
        if result is None:
            # No candidates generated for this target
            per_target_results.append(TargetRecoveryResult(
                target_name=target_name,
                target_smiles=target_smiles,
                retained_candidates=0,
                solved_candidates=0,
                eligible_reference_ids=[r.reference_id for r in refs if r.exact_eligibility == ExactEligibility.CORE_EXACT_ELIGIBLE],
                excluded_reference_ids=[r.reference_id for r in refs if r.exact_eligibility == ExactEligibility.EXCLUDED],
                exact_recovered=False,
                best_exact_rank=None,
                baseline_best_exact_rank=None,
                recovery_at_1=False,
                recovery_at_3=False,
                recovery_at_5=False,
                recovery_at_10=False,
                recovery_at_retained=False,
                unsolved_exact_match_present=False,
            ))
            continue

        routes = result.routes
        retained_candidates = len(routes)
        solved_routes = [r for r in routes if r.solved]
        solved_candidates = len(solved_routes)

        if not solved_routes:
            # No solved routes
            eligible_refs = [r for r in refs if r.exact_eligibility == ExactEligibility.CORE_EXACT_ELIGIBLE]
            excluded_refs = [r for r in refs if r.exact_eligibility == ExactEligibility.EXCLUDED]
            stereo_stress_refs = [r for r in refs if r.exact_eligibility == ExactEligibility.STEREO_STRESS]

            per_target_results.append(TargetRecoveryResult(
                target_name=target_name,
                target_smiles=target_smiles,
                retained_candidates=retained_candidates,
                solved_candidates=0,
                eligible_reference_ids=[r.reference_id for r in eligible_refs],
                excluded_reference_ids=[r.reference_id for r in excluded_refs + stereo_stress_refs],
                exact_recovered=False,
                best_exact_rank=None,
                baseline_best_exact_rank=None,
                recovery_at_1=False,
                recovery_at_3=False,
                recovery_at_5=False,
                recovery_at_10=False,
                recovery_at_retained=False,
                unsolved_exact_match_present=False,
            ))
            continue

        # Rank candidates
        ranking = rank_candidates(solved_routes)
        reagent_ranked = ranking.reagent_ranked
        baseline_ranked = ranking.baseline_ranked

        # Derive candidates
        derived_cands = [derive_candidate(r, derivation_generator_hash) for r in reagent_ranked]

        # Check for exact matches with each reference
        reference_matches: list[ExactReferenceMatch] = []
        unsolved_match_present = False

        # Also check unsolved retained routes for diagnostic
        unsolved_routes = [r for r in routes if not r.solved]
        unsolved_cands = [derive_candidate(r, derivation_generator_hash) for r in unsolved_routes]

        for ref in refs:
            if ref.exact_eligibility not in (ExactEligibility.CORE_EXACT_ELIGIBLE, ExactEligibility.STEREO_STRESS):
                continue

            is_core = ref.exact_eligibility == ExactEligibility.CORE_EXACT_ELIGIBLE

            # Check solved candidates
            reagent_rank = None
            baseline_rank = None

            for rank, cand in enumerate(derived_cands, 1):
                comp = compare_graph_exact(ref, cand)
                if comp.equality == GraphEquality.EXACT:
                    reagent_rank = rank
                    break

            for rank, cand in enumerate(baseline_ranked, 1):
                derived = derive_candidate(cand, derivation_generator_hash)
                comp = compare_graph_exact(ref, derived)
                if comp.equality == GraphEquality.EXACT:
                    baseline_rank = rank
                    break

            # Check unsolved for diagnostic
            if reagent_rank is None:
                for ucand in unsolved_cands:
                    comp = compare_graph_exact(ref, ucand)
                    if comp.equality == GraphEquality.EXACT:
                        unsolved_match_present = True
                        break

            if reagent_rank is not None:
                reference_matches.append(ExactReferenceMatch(
                    reference_id=ref.reference_id,
                    candidate_route_id=cand.route_id,
                    reagent_rank=reagent_rank,
                    baseline_rank=baseline_rank,
                    is_core_exact=is_core,
                ))

        # Determine target-level recovery (core exact eligible only)
        core_matches = [m for m in reference_matches if m.is_core_exact]
        exact_recovered = len(core_matches) > 0
        best_exact_rank = min((m.reagent_rank for m in core_matches), default=None)
        baseline_best_exact_rank = min((m.baseline_rank for m in core_matches if m.baseline_rank is not None), default=None)

        eligible_ref_ids = [r.reference_id for r in refs if r.exact_eligibility == ExactEligibility.CORE_EXACT_ELIGIBLE]
        excluded_ref_ids = [r.reference_id for r in refs if r.exact_eligibility == ExactEligibility.EXCLUDED]

        # Recovery@k
        recovery_at_1 = best_exact_rank is not None and best_exact_rank <= 1
        recovery_at_3 = best_exact_rank is not None and best_exact_rank <= 3
        recovery_at_5 = best_exact_rank is not None and best_exact_rank <= 5
        recovery_at_10 = best_exact_rank is not None and best_exact_rank <= 10
        recovery_at_retained = best_exact_rank is not None

        per_target_results.append(TargetRecoveryResult(
            target_name=target_name,
            target_smiles=target_smiles,
            retained_candidates=retained_candidates,
            solved_candidates=solved_candidates,
            eligible_reference_ids=eligible_ref_ids,
            excluded_reference_ids=excluded_ref_ids,
            exact_recovered=exact_recovered,
            best_exact_rank=best_exact_rank,
            baseline_best_exact_rank=baseline_best_exact_rank,
            recovery_at_1=recovery_at_1,
            recovery_at_3=recovery_at_3,
            recovery_at_5=recovery_at_5,
            recovery_at_10=recovery_at_10,
            recovery_at_retained=recovery_at_retained,
            unsolved_exact_match_present=unsolved_match_present,
            reference_matches=reference_matches,
        ))

    # Aggregate metrics
    n_requested = len(reference_set.references)
    n_with_core = len([r for r in reference_set.references if r.route_completeness == RouteCompleteness.COMPLETE and r.route_scope == RouteScope.FULL_ROUTE and r.graph_complete() and not any(s.inferred for s in r.steps) and r.stereo_metadata.relation_to_reagent in (StereoRelation.EXACTLY_COMPATIBLE, StereoRelation.RACEMATE_COMPATIBLE)])
    n_without_core = n_requested - n_with_core

    targets_with_core_results = [t for t in per_target_results if t.eligible_reference_ids]
    n_solved = len([t for t in per_target_results if t.solved_candidates > 0])

    def rate(attr: str) -> float:
        if not targets_with_core_results:
            return 0.0
        return sum(getattr(t, attr) for t in targets_with_core_results) / len(targets_with_core_results)

    metrics = ExactRecoveryMetrics(
        n_targets_requested=n_requested,
        n_targets_with_core_exact_reference=len(targets_with_core_results),
        n_targets_without_core_exact_reference=n_without_core,
        n_targets_solved=n_solved,
        recovery_at_1=rate("recovery_at_1"),
        recovery_at_3=rate("recovery_at_3"),
        recovery_at_5=rate("recovery_at_5"),
        recovery_at_10=rate("recovery_at_10"),
        recovery_at_retained=rate("recovery_at_retained"),
        per_target=per_target_results,
    )

    # Provenance
    impl_hashes = get_implementation_hashes()
    evaluator_hash = hashlib.sha256(
        json.dumps(impl_hashes, sort_keys=True).encode()
    ).hexdigest()

    ranking_prov = RankingProvenance(
        weight_vector=dict(DEFAULT_WEIGHTS),
        ranking_implementation_hash=hashlib.sha256(
            (Path(__file__).resolve().parent / "harness.py").read_bytes()
        ).hexdigest(),
        evaluator_implementation_hash=evaluator_hash,
        rdkit_version=rdkit.__version__,
        numpy_version=numpy.__version__,
    )

    provenance = ExactBenchmarkProvenance(
        reference_set_content_hash=reference_set.content_hash(),
        derivation_schema_version=1,
        candidate_checkpoint_path=str(candidate_checkpoint_path.resolve()),
        checkpoint_manifest_digest=checkpoint_digest,
        ranking_provenance=ranking_prov,
        evaluator_code_hash=evaluator_hash,
    )

    return ExactBenchmarkResult(
        provenance=provenance,
        metrics=metrics,
        reagent_weight_vector=dict(DEFAULT_WEIGHTS),
        baseline_weight_vector={"feasibility": 1.0},
    )


if __name__ == "__main__":
    # For testing
    pass