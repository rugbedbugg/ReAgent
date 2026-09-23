"""Graded route similarity for literature recovery benchmark.

This module implements the Genheden-Shields route similarity metric using the
official `rxnutils` (reaction_utils) implementation. It provides adapters to
convert ReAgent's curated references and generated candidates into the format
expected by the similarity engine, and computes graded similarity scores.
"""

from __future__ import annotations

import functools
import hashlib
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import rxnutils.routes.base as base
import rxnutils.routes.comparison as comp
import rxnutils.routes.readers as readers
from rdkit import RDLogger

from reagent.core.chem import m0_key, m1_key
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.literature import (
    LiteratureReference,
    LiteratureReferenceSet,
    MoleculeRole,
)
from reagent.eval.literature_derived import (
    ExactEligibility,
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

    mean_selected_route_similarity: float
    median_selected_route_similarity: float
    mean_max_similarity_retained: float
    median_max_similarity_retained: float

    mean_selected_atom_similarity: float
    median_selected_atom_similarity: float
    mean_selected_bond_similarity: float
    median_selected_bond_similarity: float

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
    ):
        self.reference_set = reference_set
        self.derivation_generator_hash = derivation_generator_hash
        self._derived_refs = []
        self._reference_routes = {}

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

    def _get_reference_route(self, derived_ref):
        if derived_ref.reference_id in self._reference_routes:
            return self._reference_routes[derived_ref.reference_id]

        try:
            route = self._build_reference_route(derived_ref)
            self._reference_routes[derived_ref.reference_id] = route
            return route
        except Exception as e:
            warnings.warn(f"Failed to build reference route for {derived_ref.reference_id}: {e}")
            return None

    def _build_reference_route(self, derived_ref):
        # Build route for CORE_EXACT_ELIGIBLE (M0 target match) or STEREO_STRESS (M1 target match)
        if derived_ref.exact_eligibility not in (
            ExactEligibility.CORE_EXACT_ELIGIBLE,
            ExactEligibility.STEREO_STRESS,
        ):
            return None

        tree = self._reference_to_tree(derived_ref)
        if tree is None:
            return None

        try:
            route = readers.read_aizynthfinder_dict(tree)
            # Attempt to assign atom mapping for similarity computation
            # This will fail gracefully if no mapper is available
            self._ensure_route_mapping(route)
            return route
        except Exception as e:
            warnings.warn(f"Failed to parse reference route {derived_ref.reference_id}: {e}")
            return None

    def _ensure_route_mapping(self, route: base.SynthesisRoute) -> bool:
        """
        Ensure the route has atom mappings for similarity computation.

        Returns True if mapping succeeded or was already present, False if mapping failed.
        Does not raise - mapping failures are handled by the similarity computation.
        """
        try:
            # Check if route already has meaningful atom mappings
            if route.mapped_root_smiles and any(
                ":" in mol for mol in route.mapped_root_smiles.split(".")
            ):
                return True

            # Try to assign mapping using available mapper
            route.assign_atom_mapping(overwrite=True, only_rxnmapper=False)
            return True
        except Exception:
            # Mapping failed - similarity computation will handle this
            return False

    def _has_valid_atom_mappings(self, route: base.SynthesisRoute) -> bool:
        """Check if a route has valid atom mappings for similarity computation.

        A route has valid mappings if its atom_mapped_reaction_smiles contains
        atom map numbers (indicated by :digit pattern). This does NOT attempt
        to assign mappings - it only checks existing mappings.
        """
        try:
            mapped_reactions = route.atom_mapped_reaction_smiles()
            if not mapped_reactions:
                return False
            import re
            atom_map_pattern = re.compile(r':\d+')
            for rxn_smi in mapped_reactions:
                if not atom_map_pattern.search(rxn_smi):
                    return False
            return True
        except Exception:
            return False

    def _check_mapper_availability(self) -> bool:
        """Check if any supported atom mapper (NameRxn or RxnMapper) is available."""
        import importlib.util
        if importlib.util.find_spec("namrxn") is not None:
            return True
        if importlib.util.find_spec("rxnmapper") is not None:
            return True
        return False

    def _reference_to_tree(self, derived_ref):
        # Use the original LiteratureReference for tree building since it has
        # the full graph structure with molecule_ids and roles
        original_ref = None
        for ref in self.reference_set.references:
            if ref.reference_id == derived_ref.reference_id:
                original_ref = ref
                break

        if original_ref is None:
            return None

        return self._build_tree_from_original_ref(original_ref)

    def _build_tree_from_original_ref(self, ref: LiteratureReference):
        """Build AiZynthFinder tree from original LiteratureReference."""
        if not ref.molecules or not ref.steps:
            return None

        # Build molecule_id -> M0 mapping
        mol_id_to_m0 = {}
        for m in ref.molecules:
            if m.reported_smiles:
                m0 = m0_key(m.reported_smiles)
                if m0:
                    mol_id_to_m0[m.molecule_id] = m0

        if not mol_id_to_m0:
            return None

        # Find target molecule
        target_mol_id = None
        for m in ref.molecules:
            if m.role == MoleculeRole.TARGET:
                target_mol_id = m.molecule_id
                break

        if not target_mol_id or target_mol_id not in mol_id_to_m0:
            return None

        # Build the reaction subtree for the target
        target_m0 = mol_id_to_m0[target_mol_id]
        reaction_subtree = self._build_reaction_subtree(target_mol_id, ref, mol_id_to_m0, set())

        if reaction_subtree is None:
            # Target has no producing step - it's a leaf
            return {"type": "mol", "smiles": target_m0}

        # Wrap in mol node (target molecule)
        return {
            "type": "mol",
            "smiles": target_m0,
            "children": [reaction_subtree]
        }

    def _build_reaction_subtree(
        self,
        mol_id: str,
        ref: LiteratureReference,
        mol_id_to_m0: dict[str, str],
        visited: set[str]
    ):
        """Build reaction subtree for a molecule that has a producing step."""
        if mol_id in visited:
            return None
        visited.add(mol_id)

        m0 = mol_id_to_m0.get(mol_id)
        if not m0:
            return None

        # Find producing step
        producing_step = None
        for step in ref.steps:
            if step.product_id == mol_id:
                producing_step = step
                break

        if not producing_step:
            return None

        precursor_ids = producing_step.precursor_ids
        if not precursor_ids:
            return None

        reaction_children = []
        for pid in precursor_ids:
            child_m0 = mol_id_to_m0.get(pid, "")
            if not child_m0:
                continue

            # Check if precursor has a producing step (i.e., is an intermediate)
            has_producer = any(s.product_id == pid for s in ref.steps)

            if has_producer:
                # Precursor is an intermediate - build its reaction subtree
                child_subtree = self._build_reaction_subtree(pid, ref, mol_id_to_m0, visited.copy())
                if child_subtree:
                    reaction_children.append({
                        "type": "mol",
                        "smiles": child_m0,
                        "children": [child_subtree]
                    })
            else:
                # Precursor is a starting material (leaf)
                reaction_children.append({"type": "mol", "smiles": child_m0})

        if not reaction_children:
            return None

        # Build reaction SMILES and generate atom-mapped version
        precursor_m0s = [mol_id_to_m0.get(pid, "") for pid in precursor_ids if mol_id_to_m0.get(pid, "")]
        reaction_smiles = ".".join(precursor_m0s) + ">>" + m0
        mapped_rsmi = self._generate_atom_mapped_rsmi(reaction_smiles)
        if mapped_rsmi is None:
            mapped_rsmi = reaction_smiles

        return {
            "type": "reaction",
            "smiles": reaction_smiles,
            "metadata": {"mapped_reaction_smiles": mapped_rsmi},
            "children": reaction_children
        }

    def _generate_atom_mapped_rsmi(self, reaction_smiles: str) -> str | None:
        """Generate atom-mapped reaction SMILES from unmapped reaction SMILES using RDKit."""
        try:
            from rdkit import Chem

            # Parse reaction SMILES
            rxn = Chem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
            if rxn is None:
                return None

            # Try to map the reaction
            # First, ensure reactants and products have valid molecules
            reactants = [Chem.MolFromSmiles(smi) for smi in reaction_smiles.split(">>")[0].split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in reaction_smiles.split(">>")[1].split(".")]

            # Filter out None molecules
            reactants = [m for m in reactants if m is not None]
            products = [m for m in products if m is not None]

            if not reactants or not products:
                return None

            # Create a new reaction with these molecules
            rxn = Chem.rdChemReactions.ChemicalReaction()
            for r in reactants:
                rxn.AddReactantTemplate(r)
            for p in products:
                rxn.AddProductTemplate(p)

            # Try to assign atom maps
            try:
                rxn.Initialize()
                # Get the atom-mapped reaction SMILES
                mapped_rsmi = Chem.rdChemReactions.ReactionToSmarts(rxn)
                return mapped_rsmi
            except Exception:
                return None

        except Exception:
            return None

    def _build_candidate_route(self, route_or_cand) -> base.SynthesisRoute | None:
        """Build SynthesisRoute from a generated Route or DerivedCandidate, preferring the original Route.tree."""
        # Check if it's a Route object (has tree attribute) or DerivedCandidate
        from reagent.core.models import Route
        if isinstance(route_or_cand, Route):
            route = route_or_cand
            if route.tree and isinstance(route.tree, dict):
                # Use authoritative original tree topology
                tree = route.tree
            else:
                # Fallback: reconstruct from flat representation
                from reagent.eval.literature_derived import derive_candidate
                cand_derived = derive_candidate(route, self.derivation_generator_hash)
                tree = self._candidate_to_tree(cand_derived)
                if tree is None:
                    return None
        else:
            # It's a DerivedCandidate
            tree = self._candidate_to_tree(route_or_cand)
            if tree is None:
                return None

        try:
            return readers.read_aizynthfinder_dict(tree)
        except Exception as e:
            warnings.warn(f"Failed to parse candidate route: {e}")
            return None

    def _candidate_to_tree(self, candidate):
        """Alias for _build_tree_from_flat for backward compatibility."""
        return self._build_tree_from_flat(candidate)

    def _build_tree_from_flat(self, candidate):
        """Build AiZynthFinder tree dict from DerivedCandidate flat representation.

        This reconstructs topology from the flat step list. Since DerivedCandidate
        preserves graph_signature (which encodes full topology), we can verify
        the reconstruction is faithful, but the tree structure itself is rebuilt.

        For true authoritative topology, the original Route.tree should be passed
        through the derivation pipeline. This is a known limitation.
        """
        if not candidate.steps:
            return None

        step_precursor_ids = []
        step_product_ids = []
        starting_material_ids = set()

        # Assign synthetic node IDs
        node_counter = [0]

        def get_node_id():
            node_counter[0] += 1
            return f"n{node_counter[0]}"

        # First pass: collect all unique M0s and assign node IDs
        # We need to build the tree structure from the steps
        # Use the same algorithm as _build_signature_from_reactions in literature_derived.py
        m0_lookup = {}

        # Assign node IDs to leaves first
        leaf_node_ids = {}
        smiles_to_node_id = {}
        for i, leaf_m0 in enumerate(candidate.leaf_m0s):
            leaf_id = f"leaf_{i}"
            m0_lookup[leaf_id] = leaf_m0
            leaf_node_ids[leaf_m0] = leaf_id
            smiles_to_node_id[leaf_m0] = leaf_id
            starting_material_ids.add(leaf_id)

        # Assign node IDs to reactions and their products
        for i, step in enumerate(candidate.steps):
            product_id = f"prod_{i}"
            m0_lookup[product_id] = step.product_m0
            step_product_ids.append(product_id)

            precursor_ids = []
            for prec_m0 in step.precursor_m0s:
                if prec_m0 in smiles_to_node_id:
                    precursor_id = smiles_to_node_id[prec_m0]
                else:
                    precursor_id = f"prec_{len(m0_lookup)}"
                    m0_lookup[precursor_id] = prec_m0
                    smiles_to_node_id[prec_m0] = precursor_id
                precursor_ids.append(precursor_id)

            step_precursor_ids.append(tuple(precursor_ids))

        # Build producers map
        producers: dict[str, list[int]] = {}
        for i, prod_id in enumerate(step_product_ids):
            producers.setdefault(prod_id, []).append(i)

        # Check for genuine multiple-producer ambiguity
        for prod_id, step_indices in producers.items():
            if len(step_indices) > 1:
                warnings.warn(f"Candidate {candidate.route_id}: multiple producers for {prod_id}")
                return None

        # Find root: product not used as precursor anywhere
        all_precursors = set()
        for precs in step_precursor_ids:
            all_precursors.update(precs)
        root_candidates = set(step_product_ids) - all_precursors
        if not root_candidates:
            return None
        root_id = root_candidates.pop()

        # Build tree recursively

        @functools.cache
        def build_reaction_subtree(node_id: str) -> dict | None:
            """Build reaction subtree for a node that has a producing step."""
            if node_id in starting_material_ids:
                return None  # Starting materials don't have reaction subtrees

            prod_steps = producers.get(node_id)
            if not prod_steps:
                return None
            step_idx = prod_steps[0]
            precursor_ids = step_precursor_ids[step_idx]

            reaction_children = []
            for pid in precursor_ids:
                child_m0 = m0_lookup.get(pid, "")
                if not child_m0:
                    continue

                # Check if precursor has a producer (is an intermediate)
                if pid in producers:
                    # Intermediate - build its reaction subtree
                    child_subtree = build_reaction_subtree(pid)
                    if child_subtree:
                        reaction_children.append({
                            "type": "mol",
                            "smiles": child_m0,
                            "children": [child_subtree]
                        })
                else:
                    # Starting material (leaf)
                    reaction_children.append({"type": "mol", "smiles": child_m0})

            if not reaction_children:
                return None

            # Build reaction SMILES and generate atom-mapped version
            precursor_m0s = [m0_lookup.get(pid, "") for pid in precursor_ids if m0_lookup.get(pid, "")]
            reaction_smiles = ".".join(precursor_m0s) + ">>" + m0_lookup.get(node_id, "")
            mapped_rsmi = self._generate_atom_mapped_rsmi(reaction_smiles)
            if mapped_rsmi is None:
                mapped_rsmi = reaction_smiles

            return {
                "type": "reaction",
                "smiles": reaction_smiles,
                "metadata": {"mapped_reaction_smiles": mapped_rsmi},
                "children": reaction_children,
            }

        # Build the reaction subtree for the root
        root_m0 = m0_lookup.get(root_id, "")
        reaction_subtree = build_reaction_subtree(root_id)

        if reaction_subtree is None:
            # Root has no producing step - it's a leaf
            return {"type": "mol", "smiles": root_m0}

        # Wrap in mol node (target molecule)
        return {
            "type": "mol",
            "smiles": root_m0,
            "children": [reaction_subtree]
        }

    def _compare_pair(
        self,
        ref,
        cand_route: base.SynthesisRoute,
        candidate_route_id: str,
        candidate_content_hash: str,
        reagent_rank,
        baseline_rank,
    ):
        """Compare a reference route against a pre-built candidate SynthesisRoute.

        Args:
            ref: DerivedReference object
            cand_route: Pre-built SynthesisRoute for the candidate
            candidate_route_id: Identifier for the candidate route
            candidate_content_hash: Content hash for provenance
            reagent_rank: Rank in reagent ranking (or None)
            baseline_rank: Rank in baseline ranking (or None)
        """

        ref_route = self._get_reference_route(ref)
        if ref_route is None:
            return SimilarityPairResult(
                reference_id=ref.reference_id,
                candidate_route_id=candidate_route_id,
                candidate_content_hash=candidate_content_hash,
                reagent_rank=reagent_rank,
                baseline_rank=baseline_rank,
                atom_similarity=None,
                bond_similarity=None,
                route_similarity=None,
                status=SimilarityStatus.REFERENCE_CONVERSION_FAILED,
                mapping_status="reference_conversion_failed",
                warnings=[f"Reference conversion failed: {ref.exclusion_reasons}"],
            )

        if cand_route is None:
            return SimilarityPairResult(
                reference_id=ref.reference_id,
                candidate_route_id=candidate_route_id,
                candidate_content_hash=candidate_content_hash,
                reagent_rank=reagent_rank,
                baseline_rank=baseline_rank,
                atom_similarity=None,
                bond_similarity=None,
                route_similarity=None,
                status=SimilarityStatus.CANDIDATE_CONVERSION_FAILED,
                mapping_status="candidate_conversion_failed",
                warnings=["Candidate route is None"],
            )

        # Check if routes have valid atom mappings before computing similarity
        ref_has_mappings = self._has_valid_atom_mappings(ref_route)
        cand_has_mappings = self._has_valid_atom_mappings(cand_route)

        if not ref_has_mappings or not cand_has_mappings:
            mapper_available = self._check_mapper_availability()
            if not mapper_available:
                return SimilarityPairResult(
                    reference_id=ref.reference_id,
                    candidate_route_id=candidate_route_id,
                    candidate_content_hash=candidate_content_hash,
                    reagent_rank=reagent_rank,
                    baseline_rank=baseline_rank,
                    atom_similarity=None,
                    bond_similarity=None,
                    route_similarity=None,
                    status=SimilarityStatus.MAPPER_UNAVAILABLE,
                    mapping_status="no_mapper_available",
                    warnings=["No atom mapper (NameRxn/RxnMapper) available for auto-mapping"],
                )
            else:
                return SimilarityPairResult(
                    reference_id=ref.reference_id,
                    candidate_route_id=candidate_route_id,
                    candidate_content_hash=candidate_content_hash,
                    reagent_rank=reagent_rank,
                    baseline_rank=baseline_rank,
                    atom_similarity=None,
                    bond_similarity=None,
                    route_similarity=None,
                    status=SimilarityStatus.MAPPING_FAILED,
                    mapping_status="route_lacks_atom_mappings",
                    warnings=["Route lacks atom mappings and auto-mapping failed"],
                )

        try:
            sim_matrix = comp.simple_route_similarity([ref_route, cand_route])
            if sim_matrix is None or sim_matrix.size == 0:
                return SimilarityPairResult(
                    reference_id=ref.reference_id,
                    candidate_route_id=candidate_route_id,
                    candidate_content_hash=candidate_content_hash,
                    reagent_rank=reagent_rank,
                    baseline_rank=baseline_rank,
                    atom_similarity=None,
                    bond_similarity=None,
                    route_similarity=None,
                    status=SimilarityStatus.MAPPING_FAILED,
                    mapping_status="similarity_computation_failed",
                    warnings=["similarity computation returned empty matrix"],
                )

            if sim_matrix.shape == (2, 2):
                route_sim = float(sim_matrix[0, 1])
            else:
                route_sim = None

            bond_sim = comp.simple_bond_forming_similarity([ref_route, cand_route])
            atom_sim = comp.atom_matching_bonanza_similarity([ref_route, cand_route])

            bond_val = float(bond_sim[0, 1]) if bond_sim is not None and bond_sim.shape == (2, 2) else None
            atom_val = float(atom_sim[0, 1]) if atom_sim is not None and atom_sim.shape == (2, 2) else None

            return SimilarityPairResult(
                reference_id=ref.reference_id,
                candidate_route_id=candidate_route_id,
                candidate_content_hash=candidate_content_hash,
                reagent_rank=reagent_rank,
                baseline_rank=baseline_rank,
                atom_similarity=atom_val,
                bond_similarity=bond_val,
                route_similarity=route_sim,
                status=SimilarityStatus.SUCCESS,
                mapping_status="success",
                warnings=[],
            )
        except Exception as e:
            warnings.warn(f"Similarity computation failed: {e}")
            return SimilarityPairResult(
                reference_id=ref.reference_id,
                candidate_route_id=candidate_route_id,
                candidate_content_hash=candidate_content_hash,
                reagent_rank=reagent_rank,
                baseline_rank=baseline_rank,
                atom_similarity=None,
                bond_similarity=None,
                route_similarity=None,
                status=SimilarityStatus.MAPPING_FAILED,
                mapping_status=f"exception: {str(e)}",
                warnings=[f"Similarity computation raised: {e}"],
            )

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

            # Build SynthesisRoute objects for all solved candidates (reagent ranking)
            derived_cands = [self._build_candidate_route(r) for r in reagent_ranked]
            valid_cands = [(r, c) for r, c in zip(reagent_ranked, derived_cands) if c is not None]

            if not valid_cands:
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

            # Build SynthesisRoute objects for baseline ranking
            baseline_cands = [self._build_candidate_route(r) for r in baseline_ranked]
            valid_baseline = [(r, c) for r, c in zip(baseline_ranked, baseline_cands) if c is not None]

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

                ref_route = self._get_reference_route(ref)
                if ref_route is None:
                    continue

                # For STEREO_STRESS, verify M1 target compatibility before comparing
                if is_stereo_stress:
                    if ref.target_m0 and target_m0:
                        ref_m1 = m1_key(ref.target_m0) if hasattr(ref, 'target_m0') and ref.target_m0 else None
                        if ref_m1 and ref_m1 != target_m1:
                            continue

                # Compare against reagent-ranked candidates
                best_sim_for_ref = None

                for rank, (orig_route, cand_route) in enumerate(valid_cands, 1):
                    cand_derived = derived_cands_for_hash[rank - 1]  # Same index
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
                if valid_baseline:
                    orig_route, cand_route = valid_baseline[0]
                    cand_derived = baseline_derived_cands[0]
                    comp_result = self._compare_pair(
                        ref, cand_route,
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
            if valid_cands:
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
                # Find the rank of the candidate that achieved max_sim
                best_match = max((m for m in reference_matches
                                  if m.status == SimilarityStatus.SUCCESS and m.route_similarity == max_sim),
                                 key=lambda m: m.reagent_rank or 999, default=None)
                if best_match:
                    best_rank = best_match.reagent_rank

            # Baseline selected similarity (rank 1 in baseline ranking)
            baseline_sim = None
            baseline_atom = None
            baseline_bond = None
            if valid_baseline:
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
        n_requested = len(self.reference_set.references)
        n_with_graded_ref = len([t for t in per_target_results if t.eligible_reference_ids])
        n_evaluable = len([t for t in per_target_results if t.similarity_evaluable])
        n_mapping_failed = len([t for t in per_target_results if t.mapping_failed_count > 0 and t.mapper_unavailable_count == 0])
        n_mapper_unavailable = len([t for t in per_target_results if t.mapper_unavailable_count > 0])

        def mean_attr(attr):
            targets = [t for t in per_target_results if t.similarity_evaluable and getattr(t, attr) is not None]
            if not targets:
                return 0.0
            return sum(getattr(t, attr) for t in targets) / len(targets)

        def median_attr(attr):
            vals = sorted(getattr(t, attr) for t in per_target_results
                          if t.similarity_evaluable and getattr(t, attr) is not None)
            if not vals:
                return 0.0
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
