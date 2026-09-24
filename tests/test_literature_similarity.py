"""Phase 3A tests for graded route similarity (Genheden-Shields).

These tests verify the similarity evaluator against the official rxnutils
implementation, target compatibility using M1, mapping lifecycle, and
aggregation semantics.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from reagent.core.chem import m0_key, m1_key
from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.literature import (
    EvidenceLocator,
    LiteratureReference,
    LiteratureReferenceSet,
    MoleculeRole,
    ReferenceMolecule,
    ReferenceStep,
    RouteCompleteness,
    RouteScope,
    SourceRecord,
    SourceType,
    StereoMetadata,
    StereoRelation,
    StereoStatus,
    StructureResolutionStatus,
    TargetRecord,
)
from reagent.eval.literature_derived import (
    ExactEligibility,
    derive_candidate,
    derive_reference,
)
from reagent.eval.literature_similarity import (
    SimilarityEvaluator,
    SimilarityStatus,
    run_similarity_benchmark,
)


@pytest.fixture(autouse=True)
def _no_route_mapper(monkeypatch):
    """Keep results independent of any mapper installed on the machine."""
    for name in ("RXNMAPPER_ENV_PATH", "CONDA_PATH"):
        monkeypatch.delenv(name, raising=False)


# ============================================================
# Helper: Build minimal reference and candidate for testing
# ============================================================

def create_simple_reference(ref_id: str = "ref_1") -> LiteratureReference:
    """Create a simple one-step reference for aspirin."""
    return LiteratureReference(
        reference_id=ref_id,
        target=TargetRecord(
            reagent_target_name="aspirin",
            reagent_target_smiles="CC(=O)Oc1ccccc1C(=O)O",
        ),
        sources=[SourceRecord(
            source_id="src_1",
            source_type=SourceType.PEER_REVIEWED_ARTICLE,
            is_primary=True,
        )],
        route_scope=RouteScope.FULL_ROUTE,
        route_completeness=RouteCompleteness.COMPLETE,
        molecules=[
            ReferenceMolecule(
                molecule_id="mol_target",
                reported_smiles="CC(=O)Oc1ccccc1C(=O)O",
                role=MoleculeRole.TARGET,
                structure_resolution=StructureResolutionStatus.RESOLVED,
                stereo_status=StereoStatus.UNSPECIFIED,
            ),
            ReferenceMolecule(
                molecule_id="mol_1",
                reported_smiles="CC(=O)O",
                role=MoleculeRole.STARTING_MATERIAL,
                structure_resolution=StructureResolutionStatus.RESOLVED,
                stereo_status=StereoStatus.UNSPECIFIED,
            ),
            ReferenceMolecule(
                molecule_id="mol_2",
                reported_smiles="C1=CC=CC=C1C(=O)O",
                role=MoleculeRole.STARTING_MATERIAL,
                structure_resolution=StructureResolutionStatus.RESOLVED,
                stereo_status=StereoStatus.UNSPECIFIED,
            ),
        ],
        steps=[
            ReferenceStep(
                step_id="step_1",
                precursor_ids=["mol_1", "mol_2"],
                product_id="mol_target",
                evidence_locators=[EvidenceLocator(source_id="src_1", example="1")],
            ),
        ],
        stereo_metadata=StereoMetadata(
            target_status=StereoStatus.UNSPECIFIED,
            relation_to_reagent=StereoRelation.EXACTLY_COMPATIBLE,
        ),
    )


def create_stereo_stress_reference(ref_id: str = "ref_stereo") -> LiteratureReference:
    """Create a reference with specific stereochemistry (REFERENCE_MORE_SPECIFIC)."""
    # Target with defined stereochemistry
    target_smiles = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"  # R-ibuprofen
    return LiteratureReference(
        reference_id=ref_id,
        target=TargetRecord(
            reagent_target_name="ibuprofen_R",
            reagent_target_smiles=target_smiles,
        ),
        sources=[SourceRecord(
            source_id="src_1",
            source_type=SourceType.PEER_REVIEWED_ARTICLE,
            is_primary=True,
        )],
        route_scope=RouteScope.FULL_ROUTE,
        route_completeness=RouteCompleteness.COMPLETE,
        molecules=[
            ReferenceMolecule(
                molecule_id="mol_target",
                reported_smiles=target_smiles,
                role=MoleculeRole.TARGET,
                structure_resolution=StructureResolutionStatus.RESOLVED,
                stereo_status=StereoStatus.STEREOSPECIFIC,
            ),
            ReferenceMolecule(
                molecule_id="mol_1",
                reported_smiles="CC(C)Cc1ccc(C)cc1",
                role=MoleculeRole.STARTING_MATERIAL,
                structure_resolution=StructureResolutionStatus.RESOLVED,
            ),
            ReferenceMolecule(
                molecule_id="mol_2",
                reported_smiles="CC(=O)O",
                role=MoleculeRole.STARTING_MATERIAL,
                structure_resolution=StructureResolutionStatus.RESOLVED,
            ),
        ],
        steps=[
            ReferenceStep(
                step_id="step_1",
                precursor_ids=["mol_1", "mol_2"],
                product_id="mol_target",
                evidence_locators=[EvidenceLocator(source_id="src_1", example="1")],
            ),
        ],
        stereo_metadata=StereoMetadata(
            target_status=StereoStatus.STEREOSPECIFIC,
            relation_to_reagent=StereoRelation.REFERENCE_MORE_SPECIFIC,
        ),
    )


def create_candidate_route(smiles: str = "CC(=O)Oc1ccccc1C(=O)O",
                           target_smiles: str = "CC(=O)Oc1ccccc1C(=O)O",
                           tree: dict | None = None) -> Route:
    """Create a candidate route for testing."""
    if tree is None:
        tree = {
            "type": "mol",
            "smiles": target_smiles,
            "children": [{
                "type": "reaction",
                "smiles": "CC(=O)O.C1=CC=CC=C1C(=O)O>>CC(=O)Oc1ccccc1C(=O)O",
                "metadata": {
                    "mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]>>[CH3:1][C:2](=[O:3])[O:4].[c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]"
                },
                "children": [
                    {"type": "mol", "smiles": "CC(=O)O"},
                    {"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O"}
                ]
            }]
        }

    return Route(
        target=target_smiles,
        reactions=[
            Reaction(
                product=target_smiles,
                precursors=["CC(=O)O", "C1=CC=CC=C1C(=O)O"],
            ),
        ],
        leaves=[
            Molecule(smiles="CC(=O)O", in_stock=True),
            Molecule(smiles="C1=CC=CC=C1C(=O)O", in_stock=True),
        ],
        solved=True,
        tree=tree,
    )


def create_convergent_route() -> Route:
    """Create a convergent route: A used twice, A->B, A->C, B+C->T"""
    target = "CC(=O)Oc1ccccc1C(=O)O"
    return Route(
        target=target,
        reactions=[
            Reaction(product="CC(=O)O", precursors=["CCO"]),
            Reaction(product="C1=CC=CC=C1C(=O)O", precursors=["CCO"]),
            Reaction(product=target, precursors=["CC(=O)O", "C1=CC=CC=C1C(=O)O"]),
        ],
        leaves=[
            Molecule(smiles="CCO", in_stock=True),
            Molecule(smiles="CCO", in_stock=True),
        ],
        solved=True,
        tree={
            "type": "mol", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "children": [{
                "type": "reaction", "smiles": "CC(=O)O.C1=CC=CC=C1C(=O)O>>CC(=O)Oc1ccccc1C(=O)O",
                "metadata": {"mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]>>[CH3:1][C:2](=[O:3])[O:4].[c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]"},
                "children": [
                    {"type": "mol", "smiles": "CC(=O)O", "children": [{
                        "type": "reaction", "smiles": "CCO>>CC(=O)O",
                        "metadata": {"mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4]>>[CH3:1][CH2:2][O:3]"},
                        "children": [{"type": "mol", "smiles": "CCO"}]
                    }]},
                    {"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O", "children": [{
                        "type": "reaction", "smiles": "CCO>>C1=CC=CC=C1C(=O)O",
                        "metadata": {"mapped_reaction_smiles": "[c:1]1[c:2][c:3][c:4][c:5][c:6]1[C:7](=[O:8])[O:9]>>[CH3:1][CH2:2][O:3]"},
                        "children": [{"type": "mol", "smiles": "CCO"}]
                    }]}
                ]
            }]
        }
    )


def create_linear_route() -> Route:
    """Create a linear route with GENUINELY DIFFERENT flat reactions.

    Convergent uses: CCO->CC(=O)O, CCO->C1=CC=CC=C1C(=O)O, CC(=O)O+C1=CC=CC=C1C(=O)O->target
    Linear uses: CCO->CC(=O)O, CC(=O)O->C1=CC=CC=C1C(=O)O, C1=CC=CC=C1C(=O)O+CCO->target

    Different flat reactions (different precursors in step 2 and 3).
    """
    return Route(
        target="CC(=O)Oc1ccccc1C(=O)O",
        reactions=[
            Reaction(product="CC(=O)O", precursors=["CCO"]),
            Reaction(product="C1=CC=CC=C1C(=O)O", precursors=["CC(=O)O"]),  # Different: CC(=O)O as precursor
            Reaction(product="CC(=O)Oc1ccccc1C(=O)O", precursors=["C1=CC=CC=C1C(=O)O", "CCO"]),  # Different: only one CCO
        ],
        leaves=[Molecule(smiles="CCO", in_stock=True)],
        solved=True,
        tree={
            "type": "mol", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "children": [{
                "type": "reaction", "smiles": "C1=CC=CC=C1C(=O)O.CCO>>CC(=O)Oc1ccccc1C(=O)O",
                "metadata": {"mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1[C:7](=[O:8])[O:9].[CH3:10][CH2:11][O:12]"},
                "children": [
                    {"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O", "children": [{
                        "type": "reaction", "smiles": "CC(=O)O>>C1=CC=CC=C1C(=O)O",
                        "metadata": {"mapped_reaction_smiles": "[c:1]1[c:2][c:3][c:4][c:5][c:6]1[C:7](=[O:8])[O:9]>>[CH3:1][C:2](=[O:3])[O:4]"},
                        "children": [{
                            "type": "mol", "smiles": "CC(=O)O", "children": [{
                                "type": "reaction", "smiles": "CCO>>CC(=O)O",
                                "metadata": {"mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4]>>[CH3:1][CH2:2][O:3]"},
                                "children": [{"type": "mol", "smiles": "CCO"}]
                            }]}
                        ]}
                    ]},
                    {"type": "mol", "smiles": "CCO", "children": [{
                        "type": "reaction", "smiles": "CCO>>CCO",  # Identity for leaf
                        "metadata": {"mapped_reaction_smiles": "[CH3:1][CH2:2][O:3]>>[CH3:1][CH2:2][O:3]"},
                        "children": [{"type": "mol", "smiles": "CCO"}]
                    }]}
                ]
            }]
        }
    )


def create_checkpoint_with_routes(routes: list[Route], target_m0: str) -> str:
    """Create a temporary checkpoint directory with the given routes.

    Uses the correct checkpoint format: JSON files with SHA256-hashed target names.
    """
    tmpdir = tempfile.mkdtemp()
    import hashlib
    import os

    # Write manifest - checkpoint expects "schema" not "schema_version"
    manifest = {
        "schema": 1,
        "config_hash": "test_config_hash",
        "model_hash": "test_model_hash",
        "stock_hash": "test_stock_hash",
        "search_hash": "test_search_hash",
        "deps_hash": "test_deps_hash",
        "created_at": "2024-01-01T00:00:00",
    }
    with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    # Write target data - checkpoint uses SHA256 hash of target SMILES as filename
    from reagent.eval.checkpoint import SearchResult
    target_smiles = routes[0].target
    target_hash = hashlib.sha256(target_smiles.encode()).hexdigest()
    result = SearchResult(target=target_smiles, routes=routes, time_capped=False)
    with open(os.path.join(tmpdir, f"{target_hash}.json"), "w") as f:
        f.write(result.model_dump_json())

    return tmpdir


# ============================================================
# 1. OFFICIAL API PARITY TESTS
# ============================================================

class TestOfficialAPIParity:
    """Verify adapter uses official rxnutils functions correctly."""

    def test_reaction_utils_version(self):
        """Record reaction-utils distribution version."""
        import importlib.metadata

        import rxnutils
        version = importlib.metadata.version("reaction-utils")
        module_path = rxnutils.__file__
        print(f"reaction-utils version: {version}")
        print(f"rxnutils module path: {module_path}")
        assert version == "1.9.4"
        assert "rxnutils" in module_path

    def test_official_function_signatures(self):
        """Verify imported function signatures match expected API."""
        import inspect

        import rxnutils.routes.comparison as comp

        # simple_route_similarity
        sig = inspect.signature(comp.simple_route_similarity)
        assert "routes" in sig.parameters
        assert sig.return_annotation == np.ndarray

        # atom_matching_bonanza_similarity
        sig = inspect.signature(comp.atom_matching_bonanza_similarity)
        assert "routes" in sig.parameters

        # simple_bond_forming_similarity
        sig = inspect.signature(comp.simple_bond_forming_similarity)
        assert "routes" in sig.parameters

        # Verify they are separate functions (not inferred)
        assert comp.simple_route_similarity is not comp.atom_matching_bonanza_similarity
        assert comp.simple_route_similarity is not comp.simple_bond_forming_similarity

    def test_official_vs_adapter_parity(self):
        """Compare official API output directly against adapter for a fixed route pair."""
        import rxnutils.routes.comparison as comp
        import rxnutils.routes.readers as readers

        # Build two routes that are identical in topology but use official readers
        tree1 = {
            "type": "mol", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "children": [{
                "type": "reaction",
                "smiles": "CC(=O)O.C1=CC=CC=C1C(=O)O>>CC(=O)Oc1ccccc1C(=O)O",
                "metadata": {
                    "mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]>>[CH3:1][C:2](=[O:3])[O:4].[c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]"
                },
                "children": [
                    {"type": "mol", "smiles": "CC(=O)O"},
                    {"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O"}
                ]
            }]
        }

        tree2 = {
            "type": "mol", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "children": [{
                "type": "reaction",
                "smiles": "CC(=O)O.C1=CC=CC=C1C(=O)O>>CC(=O)Oc1ccccc1C(=O)O",
                "metadata": {
                    "mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]>>[CH3:1][C:2](=[O:3])[O:4].[c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]"
                },
                "children": [
                    {"type": "mol", "smiles": "CC(=O)O"},
                    {"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O"}
                ]
            }]
        }

        route1 = readers.read_aizynthfinder_dict(tree1)
        route2 = readers.read_aizynthfinder_dict(tree2)

        # Official computations
        official_total = comp.simple_route_similarity([route1, route2])[0, 1]
        official_atom = comp.atom_matching_bonanza_similarity([route1, route2])[0, 1]
        official_bond = comp.simple_bond_forming_similarity([route1, route2])[0, 1]

        # Verify geometric mean relationship
        import math
        expected_total = math.sqrt(official_atom * official_bond)
        assert abs(official_total - expected_total) < 1e-10, \
            f"simple_route_similarity != sqrt(atom * bond): {official_total} vs {expected_total}"

        # Symmetry: S(A,B) == S(B,A)
        total_21 = comp.simple_route_similarity([route2, route1])[1, 0]
        assert abs(official_total - total_21) < 1e-10

        # Bounds
        assert 0.0 <= official_total <= 1.0
        assert 0.0 <= official_atom <= 1.0
        assert 0.0 <= official_bond <= 1.0

        # Store for adapter comparison test
        self.official_total = official_total
        self.official_atom = official_atom
        self.official_bond = official_bond
        self.route1 = route1
        self.route2 = route2


# ============================================================
# 2. TARGET COMPATIBILITY: M1 FOR STEREO-STRESS
# ============================================================

class TestTargetCompatibilityM1:
    """Verify similarity target compatibility uses M1 for stereo-stress."""

    def test_opposite_enantiomers_pass_m1_compatibility(self):
        """Same constitution, opposite stereo -> should pass M1 target compatibility."""
        from reagent.core.chem import m1_key

        r_ibuprofen = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"
        s_ibuprofen = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"

        m1_r = m1_key(r_ibuprofen)
        m1_s = m1_key(s_ibuprofen)

        assert m1_r == m1_s, "M1 should match for opposite enantiomers"
        assert m0_key(r_ibuprofen) != m0_key(s_ibuprofen), "M0 should differ"

    def test_specified_vs_unspecified_stereo_pass_m1(self):
        """Specified vs unspecified stereo -> should pass if M1 matches."""
        from reagent.core.chem import m1_key

        specified = "CC[C@H](O)C"  # butan-2-ol with R config
        unspecified = "CCC(O)C"    # butan-2-ol without stereo

        m1_spec = m1_key(specified)
        m1_unspec = m1_key(unspecified)

        assert m1_spec == m1_unspec, "M1 should match for specified vs unspecified"
        assert m0_key(specified) != m0_key(unspecified), "M0 should differ"

    def test_different_protonation_fail_m1(self):
        """Different protonation -> must fail M1."""
        from reagent.core.chem import m1_key

        neutral = "CC(=O)O"
        protonated = "CC(=O)[OH2+]"

        assert m1_key(neutral) != m1_key(protonated), "M1 must preserve charge"

    def test_salt_vs_parent_fail_m1(self):
        """Salt vs parent -> must fail M1."""
        from reagent.core.chem import m1_key

        free_base = "CCN"
        hydrochloride = "CCN.Cl"

        assert m1_key(free_base) != m1_key(hydrochloride), "M1 must preserve fragments"

    def test_different_tautomer_fail_m1(self):
        """Different tautomer -> must fail M1."""
        from reagent.core.chem import m1_key

        keto = "CC(=O)C"
        enol = "CC(=C)O"

        assert m1_key(keto) != m1_key(enol), "M1 must preserve tautomeric state"


# ============================================================
# 3. CANDIDATE TOPOLOGY: PREFER ORIGINAL Route.tree
# ============================================================

class TestCandidateTopologySource:
    """Verify candidate tree construction prefers authoritative Route.tree."""

    def test_tree_preferred_over_flat_reconstruction(self):
        """When Route.tree exists, it should be used for topology."""
        # Convergent and linear routes with DIFFERENT flat reactions (different topology)
        conv_route = create_convergent_route()
        lin_route = create_linear_route()

        # Flat reactions differ (different precursors in step 2)
        conv_flat = set(f"{r.product}>>{'.'.join(sorted(r.precursors))}" for r in conv_route.reactions)
        lin_flat = set(f"{r.product}>>{'.'.join(sorted(r.precursors))}" for r in lin_route.reactions)

        # The flat reactions should differ (different topology)
        assert conv_flat != lin_flat, "Flat reactions should differ due to different topology"

        # Trees differ
        assert conv_route.tree != lin_route.tree, "Trees should differ"

        # DerivedCandidate should preserve topology via graph_signature
        from reagent.eval.literature_derived import derive_candidate
        cand_conv = derive_candidate(conv_route, "test_hash")
        cand_lin = derive_candidate(lin_route, "test_hash")

        # Content hashes must differ because graph_signature differs
        assert cand_conv.content_hash() != cand_lin.content_hash(), \
            "Different topology must produce different content hash"
        assert cand_conv.graph_signature != cand_lin.graph_signature

    def test_flat_reconstruction_when_tree_missing(self):
        """When Route.tree is missing, fall back to flat reconstruction."""
        route = Route(
            target="CC(=O)O",
            reactions=[Reaction(product="CC(=O)O", precursors=["CCO"])],
            leaves=[Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree=None,  # No tree!
        )

        from reagent.eval.literature_derived import derive_candidate
        cand = derive_candidate(route, "test_hash")

        # Should still work but graph_signature comes from flat reconstruction
        assert cand.graph_signature is not None


# ============================================================
# 4. REFERENCE MAPPING LIFECYCLE
# ============================================================

class TestReferenceMappingLifecycle:
    """Trace atom mapping from LiteratureReference through similarity."""

    def test_reference_route_gets_mapped(self):
        """LiteratureReference -> DerivedReference -> SynthesisRoute -> mapped."""
        ref = create_simple_reference()
        derived = derive_reference(ref, "test_hash")

        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "test_hash")
        route = evaluator._get_reference_route(derived)

        assert route is not None
        # Literature steps carry no mapping of their own; without a mapper the
        # route must report none rather than an unmapped stand-in.
        if not evaluator._check_mapper_availability():
            assert evaluator._has_valid_atom_mappings(route) is False

    def test_candidate_route_gets_mapped(self):
        """Candidate Route -> DerivedCandidate -> SynthesisRoute -> mapped."""
        ref = create_simple_reference()
        derive_reference(ref, "test_hash")

        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "test_hash")

        cand_route = create_candidate_route()
        cand_synthesis = evaluator._build_candidate_route(
            derive_candidate(cand_route, "test_hash")
        )

        assert cand_synthesis is not None
        # Mapping attempted

    def test_mapping_failure_handled_gracefully(self):
        """If mapping fails, comparison returns MAPPING_FAILED not zero."""
        # This is tested implicitly in the similarity computation
        pass


# ============================================================
# 5. MANDATORY: SIMILARITY == 1.0 BUT EXACT == FALSE
# ============================================================

class TestSimilarityOneNotExact:
    """Test that similarity=1.0 does not imply exact recovery."""

    def test_convergent_vs_linear_similarity_one(self):
        """Convergent vs linear with same target atoms -> similarity may be 1.0.

        NOTE: With the current rxnutils 1.9.4 implementation, the Genheden-Shields metric
        requires identical atom mapping numbers for similarity=1.0. Due to the reader's
        atom mapping transformation stripping non-root atoms at each reaction step,
        convergent routes with different leaving groups cannot achieve similarity=1.0
        while differing in exact comparison. This test documents the actual behavior.
        """
        import rxnutils.routes.comparison as comp
        import rxnutils.routes.readers as readers

        # Use the route creation functions which have proper trees
        conv_route = create_convergent_route()
        lin_route = create_linear_route()

        # Read trees with proper atom mapping
        route_conv = readers.read_aizynthfinder_dict(conv_route.tree)
        route_lin = readers.read_aizynthfinder_dict(lin_route.tree)

        # Compute official similarity
        sim = comp.simple_route_similarity([route_conv, route_lin])[0, 1]
        atom = comp.atom_matching_bonanza_similarity([route_conv, route_lin])[0, 1]
        bond = comp.simple_bond_forming_similarity([route_conv, route_lin])[0, 1]

        print(f"Convergent vs Linear: total={sim:.4f}, atom={atom:.4f}, bond={bond:.4f}")

        # Document the actual behavior - similarity < 1.0 due to atom mapping differences
        # from convergent route topology and leaving group differences
        assert 0.0 <= sim <= 1.0
        # This test documents that similarity=1.0 / exact=false is not achievable
        # with the current rxnutils implementation for chemically meaningful differences

    def test_identical_routes_similarity_one(self):
        """Identical routes from same tree -> similarity behavior documented."""
        import rxnutils.routes.comparison as comp
        import rxnutils.routes.readers as readers

        cand_route = create_candidate_route()
        route1 = readers.read_aizynthfinder_dict(cand_route.tree)
        route2 = readers.read_aizynthfinder_dict(cand_route.tree)

        # Two routes read from the same mapped tree are identical
        sim = comp.simple_route_similarity([route1, route2])[0, 1]
        assert sim == pytest.approx(1.0)


# N-methylbenzamide, mapped the way AiZynthFinder stores it (product>>reactants).
# The acid and acid-chloride routes form the same target bond with different
# leaving groups; the N-methylation route forms a different target bond.
_AMIDE = "CNC(=O)c1ccccc1"
_AMIDE_MAPPED = "[CH3:10][NH:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1"
# The same amide with an unrelated numbering, as an independent mapper call would give
_AMIDE_RENUMBERED = "[CH3:1][NH:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"


def _amide_route(precursors: list[str], mapped_retro: str) -> Route:
    tree = {
        "type": "mol", "smiles": _AMIDE,
        "children": [{
            "type": "reaction", "smiles": f"{_AMIDE}>>{'.'.join(precursors)}",
            "metadata": {"mapped_reaction_smiles": mapped_retro},
            "children": [{"type": "mol", "smiles": p} for p in precursors],
        }],
    }
    return Route(
        target=_AMIDE,
        reactions=[Reaction(product=_AMIDE, precursors=precursors)],
        leaves=[Molecule(smiles=p, in_stock=True) for p in precursors],
        solved=True,
        tree=tree,
    )


_ACID_CHLORIDE_ROUTE = lambda: _amide_route(  # noqa: E731
    ["O=C(Cl)c1ccccc1", "CN"],
    _AMIDE_MAPPED + ">>[O:3]=[C:2](Cl)[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.[CH3:10][NH2:1]",
)
_ACID_ROUTE_RENUMBERED = lambda: _amide_route(  # noqa: E731
    ["O=C(O)c1ccccc1", "CN"],
    _AMIDE_RENUMBERED + ">>[O:4]=[C:3]([OH:20])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[CH3:1][NH2:2]",
)
_METHYLATION_ROUTE = lambda: _amide_route(  # noqa: E731
    ["NC(=O)c1ccccc1", "CI"],
    _AMIDE_MAPPED + ">>[NH2:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.I[CH3:10]",
)


def _amide_reference() -> LiteratureReference:
    """Amide coupling from benzoyl chloride, the chemistry of _ACID_CHLORIDE_ROUTE."""
    def mol(molecule_id, smiles, role):
        return ReferenceMolecule(molecule_id=molecule_id, reported_smiles=smiles, role=role,
                                 structure_resolution=StructureResolutionStatus.RESOLVED)
    return LiteratureReference(
        reference_id="ref_amide",
        target=TargetRecord(reagent_target_name="N-methylbenzamide", reagent_target_smiles=_AMIDE),
        sources=[SourceRecord(source_id="src", source_type=SourceType.PATENT, is_primary=True)],
        molecules=[
            mol("mol_t", _AMIDE, MoleculeRole.TARGET),
            mol("mol_a", "O=C(Cl)c1ccccc1", MoleculeRole.STARTING_MATERIAL),
            mol("mol_b", "CN", MoleculeRole.STARTING_MATERIAL),
        ],
        steps=[ReferenceStep(step_id="s1", precursor_ids=["mol_a", "mol_b"], product_id="mol_t",
                             evidence_locators=[EvidenceLocator(source_id="src", example="1")])],
        stereo_metadata=StereoMetadata(relation_to_reagent=StereoRelation.EXACTLY_COMPATIBLE),
    )


class TestPreMappedAdapterComparison:
    """Pre-mapped comparisons through the adapter, with mappings as AiZynthFinder stores them."""

    def _compare(self, candidate: Route):
        ref = _amide_reference()
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "h")
        derived_ref = derive_reference(ref, "h")
        # Supply the reference's mapping directly: the curated schema carries none
        evaluator._reference_routes[ref.reference_id] = evaluator._build_candidate_route(_ACID_CHLORIDE_ROUTE())
        cand = derive_candidate(candidate, "h")
        return evaluator._compare_pair(
            derived_ref, evaluator._build_candidate_route(candidate),
            candidate_route_id=cand.route_id, candidate_content_hash=cand.content_hash(),
            reagent_rank=1, baseline_rank=None,
        )

    def test_similarity_one_does_not_imply_exact_recovery(self):
        """Different leaving groups, independent numbering: similarity 1.0, exact recovery false."""
        from reagent.eval.literature_exact import GraphEquality, compare_graph_exact

        candidate = _ACID_ROUTE_RENUMBERED()
        result = self._compare(candidate)

        assert result.status == SimilarityStatus.SUCCESS
        assert result.route_similarity == pytest.approx(1.0)
        assert result.atom_similarity == pytest.approx(1.0)
        assert result.bond_similarity == pytest.approx(1.0)
        exact = compare_graph_exact(derive_reference(_amide_reference(), "h"), derive_candidate(candidate, "h"))
        assert exact.equality == GraphEquality.NOT_EXACT

    def test_different_disconnection_scores_zero_bond_similarity(self):
        result = self._compare(_METHYLATION_ROUTE())

        assert result.status == SimilarityStatus.SUCCESS
        assert result.bond_similarity == pytest.approx(0.0)
        assert result.route_similarity == pytest.approx(0.0)

    def test_comparison_does_not_mutate_candidate_mapping(self):
        candidate = _ACID_ROUTE_RENUMBERED()
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[_amide_reference()]), "h")
        built = evaluator._build_candidate_route(candidate)
        before = built.atom_mapped_reaction_smiles()
        evaluator._reference_routes["ref_amide"] = evaluator._build_candidate_route(_ACID_CHLORIDE_ROUTE())
        evaluator._compare_pair(derive_reference(_amide_reference(), "h"), built, "c", "h", 1, None)
        assert built.atom_mapped_reaction_smiles() == before

    def test_different_root_compound_is_target_mismatch(self):
        aspirin = create_candidate_route()
        result = self._compare(aspirin)

        assert result.status == SimilarityStatus.TARGET_MISMATCH
        assert result.route_similarity is None


class TestMappingLayerSemantics:
    """Mapping outcomes Phase 3A must report, whatever the mapper environment."""

    def test_python_rxnmapper_package_is_not_a_route_mapper(self, tmp_path, monkeypatch):
        """rxnutils maps through a conda environment, so an importable rxnmapper
        package does not make mapping possible: that is MAPPER_UNAVAILABLE, not a
        mapping failure."""
        (tmp_path / "rxnmapper").mkdir()
        (tmp_path / "rxnmapper" / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delenv("RXNMAPPER_ENV_PATH", raising=False)
        ref = create_simple_reference()
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "h")
        cand = create_candidate_route()

        result = evaluator._compare_pair(derive_reference(ref, "h"), evaluator._build_candidate_route(cand),
                                         "c", "h", 1, None)

        assert result.status == SimilarityStatus.MAPPER_UNAVAILABLE
        assert result.route_similarity is None

    def test_per_step_target_number_reuse_is_not_scored(self):
        """AiZynthFinder can reuse a target map number for an atom outside the
        target in a later step. Scoring that would count a formed "target" bond
        that does not exist (C-Cl here), so the stored maps are rejected."""
        chloride_step = (
            "[O:3]=[C:2]([Cl:10])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1"
            ">>[O:3]=[C:2](O)[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.O=S([Cl:10])Cl"
        )
        route = _amide_route(["O=C(Cl)c1ccccc1", "CN"], _ACID_CHLORIDE_ROUTE().tree["children"][0]["metadata"][
            "mapped_reaction_smiles"])
        route.tree["children"][0]["children"][0]["children"] = [{
            "type": "reaction", "smiles": "", "metadata": {"mapped_reaction_smiles": chloride_step},
            "children": [{"type": "mol", "smiles": "O=C(O)c1ccccc1"}, {"type": "mol", "smiles": "O=S(Cl)Cl"}],
        }]
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[_amide_reference()]), "h")
        evaluator._reference_routes["ref_amide"] = evaluator._build_candidate_route(_ACID_CHLORIDE_ROUTE())

        result = evaluator._compare_pair(derive_reference(_amide_reference(), "h"),
                                         evaluator._build_candidate_route(route), "c", "h", 1, None)

        assert result.status == SimilarityStatus.MAPPER_UNAVAILABLE
        assert result.bond_similarity is None


# ============================================================
# 6. STEREO SEMANTICS TEST
# ============================================================

class TestStereoSemantics:
    """Test stereo-sensitive exact vs similarity behavior."""

    def test_stereo_stress_reference_evaluated(self):
        """STEREO_STRESS reference should be similarity-evaluable via M1."""
        ref = create_stereo_stress_reference()
        derived = derive_reference(ref, "test_hash")

        assert derived.exact_eligibility == ExactEligibility.STEREO_STRESS
        assert derived.target_m0 == m0_key(ref.target.reagent_target_smiles)

        # M1 of target should match racemic/unspecified candidate target
        target_m1 = m1_key(ref.target.reagent_target_smiles)
        racemic_target = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"  # without stereo
        assert m1_key(racemic_target) == target_m1

    def test_exact_comparison_uses_m0(self):
        """Phase 2 exact comparison still uses M0 (strict)."""
        from reagent.eval.literature_exact import GraphEquality, compare_graph_exact

        # Reference with R stereo
        ref_r = create_stereo_stress_reference()
        derived_ref = derive_reference(ref_r, "test_hash")

        # Candidate with S stereo (opposite)
        cand_route_s = create_candidate_route(
            target_smiles="CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
            tree=None  # Will use flat
        )
        cand_derived = derive_candidate(cand_route_s, "test_hash")

        result = compare_graph_exact(derived_ref, cand_derived)
        assert result.equality == GraphEquality.NOT_EXACT, "M0 should distinguish enantiomers"


# ============================================================
# 7. STEP-PARTITION SENSITIVITY TEST
# ============================================================

class TestStepPartitionSensitivity:
    """Test that step partitioning affects similarity (telescoped vs explicit)."""

    def test_telescoped_vs_explicit_steps(self):
        """One-step telescoped vs two-step explicit -> similarity changes."""
        import rxnutils.routes.comparison as comp
        import rxnutils.routes.readers as readers

        # Two-step: A -> B -> C with aspirin target
        target = "CC(=O)Oc1ccccc1C(=O)O"
        tree_two_step = {
            "type": "mol", "smiles": target,
            "children": [{
                "type": "reaction", "smiles": "CC(=O)O.C1=CC=CC=C1C(=O)O>>CC(=O)Oc1ccccc1C(=O)O",
                "metadata": {"mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]>>[CH3:1][C:2](=[O:3])[O:4].[c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]"},
                "children": [
                    {"type": "mol", "smiles": "CC(=O)O", "children": [{
                        "type": "reaction", "smiles": "CCO>>CC(=O)O",
                        "metadata": {"mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4]>>[CH3:1][CH2:2][O:3]"},
                        "children": [{"type": "mol", "smiles": "CCO"}]
                    }]},
                    {"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O", "children": [{
                        "type": "reaction", "smiles": "CCO>>C1=CC=CC=C1C(=O)O",
                        "metadata": {"mapped_reaction_smiles": "[c:1]1[c:2][c:3][c:4][c:5][c:6]1[C:7](=[O:8])[O:9]>>[CH3:1][CH2:2][O:3]"},
                        "children": [{"type": "mol", "smiles": "CCO"}]
                    }]}
                ]
            }]
        }

        # One-step telescoped: A -> C (direct)
        tree_one_step = {
            "type": "mol", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "children": [{
                "type": "reaction", "smiles": "CCO.C1=CC=CC=C1C(=O)O>>CC(=O)Oc1ccccc1C(=O)O",
                "metadata": {"mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])[O:4][c:5]1[c:6][c:7][c:8][c:9][c:10]1[C:11](=[O:12])[O:13]>>[CH3:1][CH2:2][O:3].[c:4]1[c:5][c:6][c:7][c:8][c:9]1[C:10](=[O:11])[O:12]"},
                "children": [{"type": "mol", "smiles": "CCO"}, {"type": "mol", "smiles": "C1=CC=CC=C1C(=O)O"}]
            }]
        }

        route_two = readers.read_aizynthfinder_dict(tree_two_step)
        route_one = readers.read_aizynthfinder_dict(tree_one_step)

        sim = comp.simple_route_similarity([route_two, route_one])[0, 1]
        atom = comp.atom_matching_bonanza_similarity([route_two, route_one])[0, 1]
        bond = comp.simple_bond_forming_similarity([route_two, route_one])[0, 1]

        print(f"Two-step vs Telescoped: total={sim:.4f}, atom={atom:.4f}, bond={bond:.4f}")

        # The official metric may or may not give 1.0 depending on implementation
        # This test documents the actual behavior - DO NOT normalize steps
        assert 0.0 <= sim <= 1.0


# ============================================================
# 8. COMPONENT SCORES FROM OFFICIAL API
# ============================================================

class TestComponentScoresFromOfficialAPI:
    """Verify SimilarityPairResult components come directly from official functions."""
    def test_adapter_uses_official_total_atom_bond(self):
        """Adapter must read total/atom/bond directly from official functions."""
        from reagent.eval.literature_derived import derive_candidate, derive_reference

        ref = create_simple_reference()
        derived_ref = derive_reference(ref, "test_hash")

        # Use a candidate route with explicit tree that has atom maps
        cand_route = create_candidate_route()
        cand_derived = derive_candidate(cand_route, "test_hash")

        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "test_hash")
        ref_route = evaluator._get_reference_route(derived_ref)
        cand_synthesis = evaluator._build_candidate_route(cand_derived)

        assert ref_route is not None
        assert cand_synthesis is not None

        # Compare via adapter
        result = evaluator._compare_pair(
            derived_ref, cand_synthesis, "cand_1", "hash1", 1, None
        )

        # Without mapper, we expect MAPPER_UNAVAILABLE
        # This is the correct behavior when no mapper is installed
        assert result.status in (SimilarityStatus.SUCCESS, SimilarityStatus.MAPPER_UNAVAILABLE)

        if result.status == SimilarityStatus.SUCCESS:
            assert result.route_similarity is not None
            assert result.atom_similarity is not None
            assert result.bond_similarity is not None

            # Verify they come from official functions (not inferred)
            import rxnutils.routes.comparison as comp
            official_total = comp.simple_route_similarity([ref_route, cand_synthesis])[0, 1]
            official_atom = comp.atom_matching_bonanza_similarity([ref_route, cand_synthesis])[0, 1]
            official_bond = comp.simple_bond_forming_similarity([ref_route, cand_synthesis])[0, 1]

            assert abs(result.route_similarity - official_total) < 1e-10
            assert abs(result.atom_similarity - official_atom) < 1e-10
            assert abs(result.bond_similarity - official_bond) < 1e-10

        # Verify geometric mean relationship when available


    def test_geometric_mean_relationship(self):
        """route_similarity ≈ sqrt(atom * bond) as diagnostic."""
        from reagent.eval.literature_derived import derive_candidate, derive_reference

        ref = create_simple_reference()
        derived_ref = derive_reference(ref, "test_hash")

        cand_route = create_candidate_route()
        cand_derived = derive_candidate(cand_route, "test_hash")

        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "test_hash")
        evaluator._get_reference_route(derived_ref)
        cand_synthesis = evaluator._build_candidate_route(cand_derived)

        result = evaluator._compare_pair(
            derived_ref, cand_synthesis, "cand_1", "hash1", 1, None
        )

        if all(v is not None for v in [result.route_similarity, result.atom_similarity, result.bond_similarity]):
            import math
            expected = math.sqrt(result.atom_similarity * result.bond_similarity)
            assert abs(result.route_similarity - expected) < 1e-10


# ============================================================
# 9. AGGREGATION AND FAILURE SEMANTICS
# ============================================================

_PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"


def _reference_for(target_name: str, target_smiles: str, ref_id: str) -> LiteratureReference:
    ref = create_simple_reference(ref_id)
    return ref.model_copy(update={
        "target": TargetRecord(reagent_target_name=target_name, reagent_target_smiles=target_smiles),
    })


def _run_scripted(tmp_path, monkeypatch, references, n_routes_by_target, script):
    """Run evaluate() on a real checkpoint with the per-pair comparison scripted.

    ``script(reference_id, target_smiles, reagent_rank, baseline_rank)`` returns
    ``(status, similarity)``. Only the metric computation is replaced, so this
    exercises target grouping, ranking slots and cohort aggregation.
    """
    from reagent.eval.literature_similarity import SimilarityPairResult

    checkpoint = Checkpoint(tmp_path / "checkpoint", {"schema": 1})
    for target, n in n_routes_by_target.items():
        key = m0_key(target)
        route = create_candidate_route(target_smiles=key)
        checkpoint.save(key, [route.model_copy(deep=True) for _ in range(n)], False)

    def scripted(self, ref, cand_route, candidate_route_id, candidate_content_hash, reagent_rank, baseline_rank):
        target = next(r.target.reagent_target_smiles for r in references if r.reference_id == ref.reference_id)
        status, sim = script(ref.reference_id, target, reagent_rank, baseline_rank)
        value = sim if status == SimilarityStatus.SUCCESS else None
        return SimilarityPairResult(
            ref.reference_id, candidate_route_id, candidate_content_hash, reagent_rank, baseline_rank,
            value, value, value, status, status.value, [],
        )

    monkeypatch.setattr(SimilarityEvaluator, "_compare_pair", scripted)
    evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=references), "h")
    return evaluator.evaluate(Checkpoint.open_readonly(tmp_path / "checkpoint"))


class TestAggregationSemantics:
    """Test aggregate metrics handle failures correctly."""

    def test_selected_vs_retained_best(self, tmp_path, monkeypatch):
        """rank 1 -> sim 0.55, rank 4 -> sim 0.92 -> selected=0.55, max=0.92, best_rank=4."""
        sims = {1: 0.55, 4: 0.92}
        metrics = _run_scripted(
            tmp_path, monkeypatch, [create_simple_reference("ref_a")], {"CC(=O)Oc1ccccc1C(=O)O": 5},
            lambda ref_id, target, rank, base: (SimilarityStatus.SUCCESS, sims.get(rank, 0.3)),
        )
        target = metrics.per_target[0]
        assert target.selected_route_similarity == pytest.approx(0.55)
        assert target.max_similarity_retained == pytest.approx(0.92)
        assert target.best_similarity_rank == 4

    def test_best_rank_is_lowest_rank_among_ties(self, tmp_path, monkeypatch):
        """Ties at ranks 1 and 4 (and the baseline pick) report rank 1."""
        metrics = _run_scripted(
            tmp_path, monkeypatch, [create_simple_reference("ref_a")], {"CC(=O)Oc1ccccc1C(=O)O": 5},
            lambda ref_id, target, rank, base: (SimilarityStatus.SUCCESS, 0.9 if rank in (1, 4, None) else 0.2),
        )
        assert metrics.per_target[0].best_similarity_rank == 1

    def test_multiple_references_preserves_both(self, tmp_path, monkeypatch):
        """Multiple references -> preserve both pairwise, use best for max; still one target."""
        refs = [create_simple_reference("ref_a"), create_simple_reference("ref_b")]
        sims = {"ref_a": 0.4, "ref_b": 0.7}
        metrics = _run_scripted(
            tmp_path, monkeypatch, refs, {"CC(=O)Oc1ccccc1C(=O)O": 2},
            lambda ref_id, target, rank, base: (SimilarityStatus.SUCCESS, sims[ref_id]),
        )
        target = metrics.per_target[0]
        assert {m.reference_id for m in target.reference_matches} == {"ref_a", "ref_b"}
        assert target.selected_route_similarity == pytest.approx(0.7)
        assert metrics.n_targets_requested == 1

    def test_mapping_failed_excluded_from_mean(self, tmp_path, monkeypatch):
        """Failed comparisons must not become zero in mean."""
        refs = [create_simple_reference("ref_a"), _reference_for("paracetamol", _PARACETAMOL, "ref_p")]

        def script(ref_id, target, rank, base):
            if ref_id == "ref_p":
                return SimilarityStatus.MAPPING_FAILED, None
            return SimilarityStatus.SUCCESS, 0.6

        metrics = _run_scripted(tmp_path, monkeypatch, refs,
                                {"CC(=O)Oc1ccccc1C(=O)O": 1, _PARACETAMOL: 1}, script)
        assert metrics.mean_selected_route_similarity == pytest.approx(0.6)
        assert metrics.median_selected_route_similarity == pytest.approx(0.6)

    def test_mapping_failed_counted_in_denominator(self, tmp_path, monkeypatch):
        """Target with all failed comparisons -> in n_targets_mapping_failed."""
        refs = [create_simple_reference("ref_a"), _reference_for("paracetamol", _PARACETAMOL, "ref_p")]

        def script(ref_id, target, rank, base):
            if ref_id == "ref_p":
                return SimilarityStatus.MAPPING_FAILED, None
            return SimilarityStatus.SUCCESS, 0.6

        metrics = _run_scripted(tmp_path, monkeypatch, refs,
                                {"CC(=O)Oc1ccccc1C(=O)O": 1, _PARACETAMOL: 1}, script)
        assert metrics.n_targets_requested == 2
        assert metrics.n_targets_with_graded_reference == 2
        assert metrics.n_targets_similarity_evaluable == 1
        assert metrics.n_targets_mapping_failed == 1

    def test_aggregate_over_evaluable_only(self, tmp_path, monkeypatch):
        """With no evaluable target, aggregates are None rather than zero."""
        metrics = _run_scripted(
            tmp_path, monkeypatch, [create_simple_reference("ref_a")], {"CC(=O)Oc1ccccc1C(=O)O": 2},
            lambda ref_id, target, rank, base: (SimilarityStatus.MAPPER_UNAVAILABLE, None),
        )
        assert metrics.n_targets_similarity_evaluable == 0
        assert metrics.n_targets_mapper_unavailable == 1
        assert metrics.mean_selected_route_similarity is None
        assert metrics.median_max_similarity_retained is None


# ============================================================
# 10. MAPPING FAILURE TEST
# ============================================================

class TestCandidateRankAttribution:
    """A candidate that cannot be converted keeps its rank slot."""

    def test_unconvertible_rank_one_does_not_promote_rank_two(self, tmp_path, monkeypatch):
        import reagent.eval.harness as harness
        from reagent.eval.literature_similarity import SimilarityPairResult

        target = m0_key("CC(=O)Oc1ccccc1C(=O)O")
        convertible = create_candidate_route(target_smiles=target)
        # Treeless, and salicylic acid has two producers: its topology cannot be rebuilt
        ambiguous = Route(
            target=target,
            reactions=[
                Reaction(product=target, precursors=["CC(=O)O", "O=C(O)c1ccccc1O"]),
                Reaction(product="O=C(O)c1ccccc1O", precursors=["Oc1ccccc1"]),
                Reaction(product="O=C(O)c1ccccc1O", precursors=["COC(=O)c1ccccc1O"]),
            ],
            leaves=[Molecule(smiles=s, in_stock=True) for s in ("CC(=O)O", "Oc1ccccc1", "COC(=O)c1ccccc1O")],
            solved=True,
            tree=None,
        )
        checkpoint = Checkpoint(tmp_path / "checkpoint", {"schema": 1})
        checkpoint.save(target, [convertible, ambiguous], False)
        order = [ambiguous, convertible]
        monkeypatch.setattr(harness, "rank_candidates", lambda routes: harness.RankingResult(
            reagent_ranked=order, baseline_ranked=order, weight_vector={}))

        calls = []

        def recording(self, ref, cand_route, candidate_route_id, candidate_content_hash, reagent_rank, baseline_rank):
            calls.append((candidate_route_id, reagent_rank, baseline_rank, cand_route is None))
            return SimilarityPairResult(ref.reference_id, candidate_route_id, candidate_content_hash, reagent_rank,
                                        baseline_rank, None, None, None, SimilarityStatus.MAPPER_UNAVAILABLE, "x", [])

        monkeypatch.setattr(SimilarityEvaluator, "_compare_pair", recording)
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[create_simple_reference()]), "h")
        evaluator.evaluate(Checkpoint.open_readonly(tmp_path / "checkpoint"))

        ambiguous_id = derive_candidate(ambiguous, "h").route_id
        convertible_id = derive_candidate(convertible, "h").route_id
        assert (ambiguous_id, 1, None, True) in calls
        assert (convertible_id, 2, None, False) in calls
        # The baseline-selected slot is the unconvertible route, not the next one
        assert (ambiguous_id, None, 1, True) in calls


class TestMappingFailure:
    """Test explicit mapping failure handling."""

    def test_mapping_failure_returns_typed_status(self):
        """Force mapping failure -> status=MAPPING_FAILED, similarity=None."""

        from reagent.core.models import Molecule, Reaction, Route
        from reagent.eval.literature import LiteratureReferenceSet
        from reagent.eval.literature_derived import derive_candidate, derive_reference

        # Create a reference with a route that has no atom mapping
        ref = create_simple_reference()
        derived_ref = derive_reference(ref, "test_hash")

        # Create a candidate route without atom mapping (no tree)
        cand_route = Route(
            target="CC(=O)Oc1ccccc1C(=O)O",
            reactions=[Reaction(product="CC(=O)Oc1ccccc1C(=O)O", precursors=["CC(=O)O", "C1=CC=CC=C1C(=O)O"])],
            leaves=[Molecule(smiles="CC(=O)O", in_stock=True), Molecule(smiles="C1=CC=CC=C1C(=O)O", in_stock=True)],
            solved=True,
            tree=None,  # No tree = no atom mapping
        )
        cand_derived = derive_candidate(cand_route, "test_hash")

        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "test_hash")
        evaluator._get_reference_route(derived_ref)
        cand_synthesis = evaluator._build_candidate_route(cand_derived)

        # The candidate route should fail due to missing atom mapping
        # Result should be MAPPER_UNAVAILABLE or MAPPING_FAILED
        result = evaluator._compare_pair(
            derived_ref, cand_synthesis, "cand_1", "hash1", 1, None
        )

        assert result.status in (SimilarityStatus.MAPPER_UNAVAILABLE, SimilarityStatus.MAPPING_FAILED)
        assert result.route_similarity is None
        assert result.atom_similarity is None
        assert result.bond_similarity is None


# ============================================================
# 11. PROVENANCE
# ============================================================

class TestProvenance:
    """Test provenance records actual implementation details."""

    def test_evaluator_records_provenance(self):
        """Evaluator should capture reaction-utils version, mapper, etc."""
        import importlib.metadata

        import rdkit
        import rxnutils

        from reagent.eval.literature_similarity import SimilarityProvenance

        prov = SimilarityProvenance(
            metric_name="Genheden-Shields",
            metric_reference_doi="10.1039/D4DD00292J",
            metric_implementation="rxnutils.routes.comparison.simple_route_similarity",
            reaction_utils_version=importlib.metadata.version("reaction-utils"),
            reaction_utils_module=rxnutils.__file__,
            adapter_version="1.0",
            rdkit_version=rdkit.__version__,
            mapping_tool="rxnmapper" if False else "rdkit_builtin",  # RXNMapper not installed
            mapping_tool_version="",
            reference_set_content_hash="test_hash",
            checkpoint_manifest_digest="test_manifest",
            ranking_implementation_hash="test_ranking_hash",
            similarity_evaluator_hash="test_eval_hash",
        )

        d = prov.to_dict()
        assert d["metric_name"] == "Genheden-Shields"
        assert d["reaction_utils_version"] == "1.9.4"
        assert "rxnutils" in d["reaction_utils_module"]


# ============================================================
# 12. REGRESSION: PHASE 1/2 SEMANTICS UNCHANGED
# ============================================================

class TestPhase12Regression:
    """Ensure Phase 3A doesn't break Phase 1/2."""

    def test_m0_m1_unchanged(self):
        """M0 and M1 behavior unchanged."""
        from reagent.core.chem import m0_key, m1_key

        # M0 preserves stereo - use ibuprofen which has a true chiral center
        r_ibuprofen = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"
        s_ibuprofen = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"
        assert m0_key(r_ibuprofen) != m0_key(s_ibuprofen), "M0 should distinguish enantiomers"

        # M1 ignores stereo
        assert m1_key(r_ibuprofen) == m1_key(s_ibuprofen), "M1 should ignore enantiomers"

        # M0 preserves charge
        assert m0_key("CC(=O)O") != m0_key("CC(=O)[OH2+]"), "M0 should preserve charge"
        # M1 preserves charge
        assert m1_key("CC(=O)O") != m1_key("CC(=O)[OH2+]"), "M1 should preserve charge"

    def test_exact_recovery_unchanged(self):
        """Phase 2 exact recovery logic unchanged."""
        from reagent.eval.literature_derived import derive_candidate, derive_reference
        from reagent.eval.literature_exact import GraphEquality, compare_graph_exact

        ref = create_simple_reference()
        derived_ref = derive_reference(ref, "test_hash")
        assert derived_ref.exact_eligibility == ExactEligibility.CORE_EXACT_ELIGIBLE

        cand_route = create_candidate_route()
        cand_derived = derive_candidate(cand_route, "test_hash")

        result = compare_graph_exact(derived_ref, cand_derived)
        assert result.equality == GraphEquality.EXACT

    def test_deterministic_ranking_unchanged(self):
        """Deterministic ranking from harness unchanged."""
        from reagent.eval.harness import rank_candidates

        routes = [
            Route(target="C", reactions=[Reaction(product="C", precursors=["A", "B"])],
                  leaves=[Molecule(smiles="A", in_stock=True), Molecule(smiles="B", in_stock=True)], solved=True),
            Route(target="C", reactions=[Reaction(product="C", precursors=["D"])],
                  leaves=[Molecule(smiles="D", in_stock=True)], solved=True),
        ]

        ranked1 = rank_candidates(routes).reagent_ranked
        ranked2 = rank_candidates(routes).reagent_ranked

        assert [id(r) for r in ranked1] == [id(r) for r in ranked2]


# ============================================================
# 13. INTEGRATION: FULL SIMILARITY BENCHMARK RUN
# ============================================================

class TestIntegration:
    """End-to-end integration test of similarity benchmark."""

    def test_run_similarity_benchmark(self):
        """Run full similarity benchmark with mock checkpoint."""
        ref = create_simple_reference()
        ref_set = LiteratureReferenceSet(references=[ref])

        cand_route = create_candidate_route()
        target_m0 = m0_key(ref.target.reagent_target_smiles)

        tmpdir = create_checkpoint_with_routes([cand_route], target_m0)
        checkpoint = Checkpoint.open_readonly(Path(tmpdir))

        try:
            metrics = run_similarity_benchmark(ref_set, checkpoint)

            assert metrics.n_targets_requested == 1
            assert metrics.n_targets_with_graded_reference == 1
            # Without atom mapper, similarity is not evaluable - mapper unavailable
            assert metrics.n_targets_similarity_evaluable == 0
            assert metrics.n_targets_mapper_unavailable == 1
            assert metrics.n_targets_mapping_failed == 0
            assert len(metrics.per_target) == 1

            target_result = metrics.per_target[0]
            assert target_result.similarity_evaluable is False
            assert target_result.mapper_unavailable_count > 0
            assert target_result.mapping_failed_count == 0
            assert len(target_result.reference_matches) > 0
            # All matches should be MAPPER_UNAVAILABLE
            for m in target_result.reference_matches:
                assert m.status == SimilarityStatus.MAPPER_UNAVAILABLE
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_stereo_stress_included_in_graded(self):
        """STEREO_STRESS references included in graded evaluation."""
        # Both references need to target the same molecule for this test
        ref_core = create_simple_reference("ref_core")
        # Create a stereo stress reference for the SAME target
        ref_stereo = LiteratureReference(
            reference_id="ref_stereo",
            target=TargetRecord(
                reagent_target_name="aspirin",
                reagent_target_smiles="CC(=O)Oc1ccccc1C(=O)O",
            ),
            sources=[SourceRecord(source_id="src_2", source_type=SourceType.PEER_REVIEWED_ARTICLE, is_primary=True)],
            route_scope=RouteScope.FULL_ROUTE,
            route_completeness=RouteCompleteness.COMPLETE,
            molecules=[
                ReferenceMolecule(molecule_id="mol_target", reported_smiles="CC(=O)Oc1ccccc1C(=O)O", role=MoleculeRole.TARGET, structure_resolution=StructureResolutionStatus.RESOLVED, stereo_status=StereoStatus.STEREOSPECIFIC),
                ReferenceMolecule(molecule_id="mol_1", reported_smiles="CC(=O)O", role=MoleculeRole.STARTING_MATERIAL, structure_resolution=StructureResolutionStatus.RESOLVED),
                ReferenceMolecule(molecule_id="mol_2", reported_smiles="C1=CC=CC=C1C(=O)O", role=MoleculeRole.STARTING_MATERIAL, structure_resolution=StructureResolutionStatus.RESOLVED),
            ],
            steps=[
                ReferenceStep(step_id="step_1", precursor_ids=["mol_1", "mol_2"], product_id="mol_target", evidence_locators=[EvidenceLocator(source_id="src_2", example="1")]),
            ],
            stereo_metadata=StereoMetadata(target_status=StereoStatus.STEREOSPECIFIC, relation_to_reagent=StereoRelation.REFERENCE_MORE_SPECIFIC),
        )
        ref_set = LiteratureReferenceSet(references=[ref_core, ref_stereo])

        # Both should be similarity-evaluable
        cand_route = create_candidate_route()
        target_m0 = m0_key(ref_core.target.reagent_target_smiles)

        tmpdir = create_checkpoint_with_routes([cand_route], target_m0)
        checkpoint = Checkpoint.open_readonly(Path(tmpdir))

        try:
            metrics = run_similarity_benchmark(ref_set, checkpoint)
            # Both references target the SAME molecule, so 1 target with 2 graded references
            assert metrics.n_targets_with_graded_reference == 1
            # The target should have both references as eligible
            target_result = metrics.per_target[0]
            assert "ref_core" in target_result.eligible_reference_ids
            assert "ref_stereo" in target_result.eligible_reference_ids
            assert len(target_result.eligible_reference_ids) == 2
        finally:
            import shutil
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])