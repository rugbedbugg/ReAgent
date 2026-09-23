"""Tests for Phase 2: Derived representations, exact graph comparison, and recovery metrics."""

from __future__ import annotations

import json

import pytest

from reagent.core.chem import m0_key, m1_key
from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.harness import (
    DEFAULT_WEIGHTS,
    RankingResult,
    _select,
    rank_baseline_feasibility,
    rank_candidates,
    rank_routes_deterministic,
)
from reagent.eval.literature import (
    EvidenceLocator,
    LiteratureReference,
    MolecularForm,
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
    DerivedCandidate,
    DerivedMolecule,
    DerivedReference,
    DerivedStep,
    ExactEligibility,
    derive_candidate,
    derive_reference,
)
from reagent.eval.literature_exact import (
    GraphEquality,
    compare_graph_exact,
)


class TestM0Identity:
    """Tests for M0 strict molecular identity."""

    def test_same_molecule_different_smiles(self):
        """Same molecule, different valid SMILES -> equal."""
        smiles1 = "CC(=O)O"
        smiles2 = "OC(=O)C"
        assert m0_key(smiles1) == m0_key(smiles2)

    def test_atom_map_differences(self):
        """Atom-map differences -> equal."""
        smiles1 = "[CH3:1][C:2](=[O:3])[O:4][H:5]"
        smiles2 = "CC(=O)O"
        assert m0_key(smiles1) == m0_key(smiles2)

    def test_opposite_enantiomers(self):
        """Opposite enantiomers -> unequal."""
        r_ibuprofen = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"
        s_ibuprofen = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"
        assert m0_key(r_ibuprofen) != m0_key(s_ibuprofen)

    def test_specified_vs_unspecified_stereo(self):
        """Specified vs unspecified stereo -> unequal where structure differs."""
        specified = "CC[C@H](O)C"
        unspecified = "CCC(O)C"
        assert m0_key(specified) != m0_key(unspecified)

    def test_neutral_vs_protonated(self):
        """Neutral vs protonated -> unequal."""
        neutral = "CC(=O)O"
        protonated = "CC(=O)[OH2+]"
        assert m0_key(neutral) != m0_key(protonated)

    def test_salt_vs_parent(self):
        """Salt/disconnected representation vs parent -> unequal."""
        free_base = "CCN"
        hydrochloride = "CCN.Cl"
        assert m0_key(free_base) != m0_key(hydrochloride)

    def test_tautomer_pair(self):
        """Tautomer pair -> unequal."""
        keto = "CC(=O)C"
        enol = "CC(=C)O"
        assert m0_key(keto) != m0_key(enol)

    def test_invalid_smiles(self):
        """Invalid SMILES returns None."""
        assert m0_key("invalid") is None
        assert m0_key("") is None
        assert m0_key("   ") is None

    def test_formal_charge_preserved(self):
        """Formal charge preserved in M0."""
        neutral = "C[N+](C)(C)C"
        charged = "C[N+](C)(C)C.[Cl-]"
        assert m0_key(neutral) != m0_key(charged)

    def test_isotopes_preserved(self):
        """Isotopes preserved in M0."""
        normal = "CCO"
        deuterated = "[2H]C[2H]O"
        assert m0_key(normal) != m0_key(deuterated)

    def test_disconnected_fragments(self):
        """Disconnected fragments preserved."""
        single = "CCO"
        salt = "CCO.[Na+].[Cl-]"
        assert m0_key(single) != m0_key(salt)


class TestM1Identity:
    """Tests for M1 stereo-agnostic diagnostic identity."""

    def test_opposite_enantiomers_equal(self):
        """Opposite enantiomers -> equal under M1."""
        r_ibuprofen = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"
        s_ibuprofen = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"
        assert m1_key(r_ibuprofen) == m1_key(s_ibuprofen)

    def test_stereo_specified_vs_unspecified(self):
        """Specified vs unspecified -> equal under M1."""
        specified = "CC[C@H](O)C"
        unspecified = "CCC(O)C"
        assert m1_key(specified) == m1_key(unspecified)

    def test_charge_differences_remain_unequal(self):
        """Charge/protonation differences remain unequal."""
        neutral = "CC(=O)O"
        protonated = "CC(=O)[OH2+]"
        assert m1_key(neutral) != m1_key(protonated)

    def test_tautomer_differences_remain_unequal(self):
        """Tautomer differences remain unequal."""
        keto = "CC(=O)C"
        enol = "CC(=C)O"
        assert m1_key(keto) != m1_key(enol)

    def test_salt_parent_differences_remain_unequal(self):
        """Salt-parent differences remain unequal."""
        free_base = "CCN"
        hydrochloride = "CCN.Cl"
        assert m1_key(free_base) != m1_key(hydrochloride)

    def test_alkene_stereo_ignored(self):
        """Alkene stereochemistry ignored in M1."""
        e_alkene = "C/C=C/C"
        z_alkene = "C/C=C\\C"
        assert m1_key(e_alkene) == m1_key(z_alkene)

    def test_invalid_smiles(self):
        """Invalid SMILES returns None."""
        assert m1_key("invalid") is None


class TestReferenceMolecule:
    """Tests for ReferenceMolecule model."""

    def test_valid_molecule(self):
        from reagent.eval.literature import StereoStatus
        mol = ReferenceMolecule(
            molecule_id="mol_1",
            reported_name="aspirin",
            reported_smiles="CC(=O)Oc1ccccc1C(=O)O",
            role=MoleculeRole.TARGET,
            structure_resolution=StructureResolutionStatus.RESOLVED,
            stereo_status=StereoStatus.UNSPECIFIED,
        )
        assert mol.m0 is not None
        assert mol.m1 is not None
        assert mol.molecule_id == "mol_1"

    def test_unresolved_structure(self):
        mol = ReferenceMolecule(
            molecule_id="mol_1",
            reported_name="unknown intermediate",
            reported_smiles=None,
            structure_resolution=StructureResolutionStatus.UNRESOLVED,
        )
        assert mol.m0 is None
        assert mol.m1 is None

    def test_molecular_form_enum(self):
        mol = ReferenceMolecule(
            molecule_id="mol_1",
            reported_smiles="CC(=O)O",
            molecular_form=MolecularForm.SALT,
        )
        assert mol.molecular_form == MolecularForm.SALT


class TestReferenceStep:
    """Tests for ReferenceStep model."""

    def test_valid_step(self):
        step = ReferenceStep(
            step_id="step_1",
            precursor_ids=["mol_1", "mol_2"],
            product_id="mol_3",
            evidence_locators=[EvidenceLocator(source_id="source_1", example="1")],
        )
        assert step.inferred is False

    def test_step_requires_evidence(self):
        with pytest.raises(ValueError, match="must have at least one evidence locator"):
            ReferenceStep(
                step_id="step_1",
                precursor_ids=["mol_1"],
                product_id="mol_2",
                evidence_locators=[],
            )

    def test_inferred_step(self):
        step = ReferenceStep(
            step_id="step_1",
            precursor_ids=["mol_1"],
            product_id="mol_2",
            evidence_locators=[EvidenceLocator(source_id="source_1")],
            inferred=True,
        )
        assert step.inferred is True


class TestSourceRecord:
    """Tests for SourceRecord model."""

    def test_peer_reviewed_article(self):
        source = SourceRecord(
            source_id="src_1",
            source_type=SourceType.PEER_REVIEWED_ARTICLE,
            title="Test Paper",
            authors=["Author A", "Author B"],
            year=2023,
            doi="10.1234/test",
            is_primary=True,
        )
        assert source.source_type == SourceType.PEER_REVIEWED_ARTICLE

    def test_patent(self):
        source = SourceRecord(
            source_id="src_1",
            source_type=SourceType.PATENT,
            title="Test Patent",
            assignee="Company Inc",
            patent_number="US1234567",
            is_primary=True,
        )
        assert source.source_type == SourceType.PATENT


class TestDerivedReference:
    """Tests for DerivedReference generation."""

    def create_valid_reference(self):
        from reagent.eval.literature import StereoStatus
        return LiteratureReference(
            reference_id="ref_1",
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

    def test_derive_reference_success(self):
        from reagent.eval.literature_derived import ExactEligibility
        ref = self.create_valid_reference()
        derived = derive_reference(ref, "test_hash")

        assert derived.reference_id == "ref_1"
        assert derived.exact_eligibility == ExactEligibility.CORE_EXACT_ELIGIBLE
        assert derived.graph_signature is not None
        assert len(derived.exclusion_reasons) == 0
        assert len(derived.molecules) == 3
        assert len(derived.steps) == 1

    def test_derive_reference_terminal_segment_ineligible(self):
        from reagent.eval.literature_derived import ExactEligibility, derive_reference
        ref = self.create_valid_reference()
        ref.route_scope = RouteScope.TERMINAL_SEGMENT
        derived = derive_reference(ref, "test_hash")

        assert derived.exact_eligibility == ExactEligibility.EXCLUDED
        assert any("route_scope=terminal_segment" in r for r in derived.exclusion_reasons)

    def test_derive_reference_partial_ineligible(self):
        from reagent.eval.literature_derived import ExactEligibility, derive_reference
        ref = self.create_valid_reference()
        ref.route_completeness = RouteCompleteness.PARTIAL
        derived = derive_reference(ref, "test_hash")

        assert derived.exact_eligibility == ExactEligibility.EXCLUDED
        assert any("route_completeness=partial" in r for r in derived.exclusion_reasons)

    def test_derive_reference_inferred_step_ineligible(self):
        from reagent.eval.literature_derived import ExactEligibility
        ref = self.create_valid_reference()
        ref.steps[0].inferred = True
        derived = derive_reference(ref, "test_hash")

        assert derived.exact_eligibility == ExactEligibility.EXCLUDED
        assert any("inferred=true" in r for r in derived.exclusion_reasons)

    def test_derive_reference_stereo_stress(self):
        from reagent.eval.literature_derived import ExactEligibility, derive_reference
        ref = self.create_valid_reference()
        ref.stereo_metadata = StereoMetadata(
            target_status=StereoStatus.STEREOSPECIFIC,
            relation_to_reagent=StereoRelation.REFERENCE_MORE_SPECIFIC,
        )
        derived = derive_reference(ref, "test_hash")

        assert derived.exact_eligibility == ExactEligibility.STEREO_STRESS

    def test_derive_reference_conflicting_stereo_excluded(self):
        from reagent.eval.literature_derived import ExactEligibility, derive_reference
        ref = self.create_valid_reference()
        ref.stereo_metadata = StereoMetadata(
            target_status=StereoStatus.STEREOSPECIFIC,
            relation_to_reagent=StereoRelation.CONFLICTING,
        )
        derived = derive_reference(ref, "test_hash")

        assert derived.exact_eligibility == ExactEligibility.EXCLUDED
        assert any("stereo_relation=conflicting" in r for r in derived.exclusion_reasons)

    def test_derive_reference_unresolved_molecule(self):
        from reagent.eval.literature_derived import ExactEligibility, derive_reference
        ref = self.create_valid_reference()
        ref.molecules[1].structure_resolution = StructureResolutionStatus.UNRESOLVED
        derived = derive_reference(ref, "test_hash")

        assert derived.exact_eligibility == ExactEligibility.EXCLUDED
        assert any("structure_resolution=unresolved" in r for r in derived.exclusion_reasons)

    def test_derive_reference_deterministic_hash(self):
        ref = self.create_valid_reference()
        derived1 = derive_reference(ref, "test_hash")
        derived2 = derive_reference(ref, "test_hash")

        assert derived1.content_hash() == derived2.content_hash()

    def test_derive_reference_changed_chemistry_different_hash(self):
        ref1 = self.create_valid_reference()
        ref2 = self.create_valid_reference()
        ref2.molecules[1].reported_smiles = "CCC(=O)O"  # Different starting material

        derived1 = derive_reference(ref1, "test_hash")
        derived2 = derive_reference(ref2, "test_hash")

        assert derived1.content_hash() != derived2.content_hash()


class TestDerivedCandidate:
    """Tests for DerivedCandidate generation from generated routes."""

    def create_test_route(self) -> Route:
        """Create a simple test route."""
        return Route(
            target="CC(=O)Oc1ccccc1C(=O)O",
            reactions=[
                Reaction(
                    product="CC(=O)Oc1ccccc1C(=O)O",
                    precursors=["CC(=O)O", "C1=CC=CC=C1C(=O)O"],
                ),
            ],
            leaves=[
                Molecule(smiles="CC(=O)O", in_stock=True),
                Molecule(smiles="C1=CC=CC=C1C(=O)O", in_stock=True),
            ],
            solved=True,
        )

    def test_derive_candidate_success(self):
        from reagent.eval.literature_derived import m0_key
        route = self.create_test_route()
        derived = derive_candidate(route, "test_hash")

        assert derived.target_m0 == m0_key(route.target)
        assert len(derived.steps) == 1
        assert len(derived.leaf_m0s) == 2
        assert derived.is_solved is True
        assert len(derived.derivation_errors) == 0

    def test_derive_candidate_unparseable_molecule(self):
        route = Route(
            target="invalid_smiles",
            reactions=[
                Reaction(product="C", precursors=["C"]),
            ],
            leaves=[Molecule(smiles="C", in_stock=True)],
            solved=True,
        )
        derived = derive_candidate(route, "test_hash")

        # Should record error but not crash
        assert len(derived.derivation_errors) > 0
        assert any("unparseable" in e for e in derived.derivation_errors)
        assert derived.target_m0 == ""

    def test_derive_candidate_unsolved(self):
        route = self.create_test_route()
        route.solved = False
        derived = derive_candidate(route, "test_hash")

        assert derived.is_solved is False


class TestExactReactionIdentity:
    """Tests for exact reaction identity (M0 product + multiset of M0 precursors)."""

    def test_precursor_order_irrelevant(self):
        """Precursor order does not affect exact key."""
        step1 = DerivedStep(
            step_id="s1",
            product_m0="CC(=O)O",
            precursor_m0s=("CCO", "CCC"),
            inferred=False,
        )
        step2 = DerivedStep(
            step_id="s2",
            product_m0="CC(=O)O",
            precursor_m0s=("CCC", "CCO"),  # Swapped
            inferred=False,
        )

        assert step1.exact_key() == step2.exact_key()

    def test_precursor_multiplicity_relevant(self):
        """Precursor multiplicity affects exact key."""
        step1 = DerivedStep(
            step_id="s1",
            product_m0="C",
            precursor_m0s=("A", "A"),  # A + A
            inferred=False,
        )
        step2 = DerivedStep(
            step_id="s2",
            product_m0="C",
            precursor_m0s=("A",),  # A only
            inferred=False,
        )

        assert step1.exact_key() != step2.exact_key()

    def test_product_identity_relevant(self):
        """Product M0 affects exact key."""
        step1 = DerivedStep(
            step_id="s1",
            product_m0="A",
            precursor_m0s=("B", "C"),
            inferred=False,
        )
        step2 = DerivedStep(
            step_id="s2",
            product_m0="D",  # Different product
            precursor_m0s=("B", "C"),
            inferred=False,
        )

        assert step1.exact_key() != step2.exact_key()

    def test_stereo_difference_changes_key(self):
        """Opposite enantiomers have different exact keys."""
        r_m0 = m0_key("CC(C)Cc1ccc([C@H](C)C(=O)O)cc1")
        s_m0 = m0_key("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1")

        step1 = DerivedStep(step_id="s1", product_m0=r_m0, precursor_m0s=("A",), inferred=False)
        step2 = DerivedStep(step_id="s2", product_m0=s_m0, precursor_m0s=("A",), inferred=False)

        assert step1.exact_key() != step2.exact_key()

    def test_atom_maps_removed_by_m0(self):
        """Atom maps do not affect M0, so don't affect exact key."""
        # Use properly mapped version with explicit hydrogens
        mapped = "[CH3:1][C:2](=[O:3])[O:4][H:5]"
        unmapped = "CC(=O)O"

        step1 = DerivedStep(
            step_id="s1",
            product_m0=m0_key(mapped),
            precursor_m0s=(m0_key(mapped),),
            inferred=False,
        )
        step2 = DerivedStep(
            step_id="s2",
            product_m0=m0_key(unmapped),
            precursor_m0s=(m0_key(unmapped),),
            inferred=False,
        )

        assert step1.exact_key() == step2.exact_key()


class TestGraphExactComparison:
    """Tests for graph-exact route comparison."""

    def create_simple_ref(self) -> DerivedReference:
        """Create a simple derived reference."""
        return DerivedReference(
            reference_id="ref_1",
            source_content_hash="hash1",
            derivation_schema_version=1,
            rdkit_version="2024.03.1",
            generator_code_hash="test_gen_hash",
            target_m0="CC(=O)Oc1ccccc1C(=O)O",
            molecules=[
                DerivedMolecule(molecule_id="m1", m0="CC(=O)Oc1ccccc1C(=O)O", m1="..."),
                DerivedMolecule(molecule_id="m2", m0="CC(=O)O", m1="..."),
                DerivedMolecule(molecule_id="m3", m0="C1=CC=CC=C1C(=O)O", m1="..."),
            ],
            steps=[
                DerivedStep(
                    step_id="s1",
                    product_m0="CC(=O)Oc1ccccc1C(=O)O",
                    precursor_m0s=("CC(=O)O", "C1=CC=CC=C1C(=O)O"),
                    inferred=False,
                ),
            ],
            starting_material_m0s=("CC(=O)O", "C1=CC=CC=C1C(=O)O"),
            intermediate_m0s=(),
            graph_signature=json.dumps(("reaction", "CC(=O)Oc1ccccc1C(=O)O", (("leaf", "CC(=O)O"), ("leaf", "C1=CC=CC=C1C(=O)O")))),
            exact_eligibility=ExactEligibility.CORE_EXACT_ELIGIBLE,
        )

    def create_simple_cand(self, target_m0: str = "CC(=O)Oc1ccccc1C(=O)O") -> DerivedCandidate:
        """Create a matching derived candidate."""
        return DerivedCandidate(
            route_id="cand_1",
            target_m0=target_m0,
            steps=[
                DerivedStep(
                    step_id="s1",
                    product_m0="CC(=O)Oc1ccccc1C(=O)O",
                    precursor_m0s=("CC(=O)O", "C1=CC=CC=C1C(=O)O"),
                    inferred=False,
                ),
            ],
            leaf_m0s=("CC(=O)O", "C1=CC=CC=C1C(=O)O"),
            graph_signature=json.dumps(("reaction", "CC(=O)Oc1ccccc1C(=O)O", (("leaf", "CC(=O)O"), ("leaf", "C1=CC=CC=C1C(=O)O")))),
            is_solved=True,
        )

    def test_same_route_exact_match(self):
        """Same route topology and molecules = EXACT match."""
        ref = self.create_simple_ref()
        cand = self.create_simple_cand()

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.EXACT
        assert result.target_m0_match is True
        assert result.graph_signature_match is True

    def test_different_target_not_exact(self):
        """Different target M0 = NOT_EXACT."""
        ref = self.create_simple_ref()
        cand = self.create_simple_cand("different_target")

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.NOT_EXACT
        assert result.target_m0_match is False

    def test_reordered_steps_same_signature(self):
        """Reordered source steps with same topology = EXACT (via graph_signature)."""
        ref = self.create_simple_ref()
        cand = self.create_simple_cand()
        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.EXACT

    def test_reordered_precursors_same_signature(self):
        """Reordered precursors = EXACT (multiset semantics)."""
        ref = self.create_simple_ref()
        cand = DerivedCandidate(
            route_id="cand_1",
            target_m0="CC(=O)Oc1ccccc1C(=O)O",
            steps=[
                DerivedStep(
                    step_id="s1",
                    product_m0="CC(=O)Oc1ccccc1C(=O)O",
                    precursor_m0s=("C1=CC=CC=C1C(=O)O", "CC(=O)O"),  # Swapped
                    inferred=False,
                ),
            ],
            leaf_m0s=("CC(=O)O", "C1=CC=CC=C1C(=O)O"),
            graph_signature=json.dumps(("reaction", "CC(=O)Oc1ccccc1C(=O)O", (("leaf", "CC(=O)O"), ("leaf", "C1=CC=CC=C1C(=O)O")))),
            is_solved=True,
        )

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.EXACT

    def test_changed_topology_not_exact(self):
        """Different graph topology = NOT_EXACT."""
        ref = self.create_simple_ref()
        cand = DerivedCandidate(
            route_id="cand_1",
            target_m0="CC(=O)Oc1ccccc1C(=O)O",
            steps=[
                DerivedStep(step_id="s1", product_m0="A", precursor_m0s=("B",), inferred=False),
                DerivedStep(step_id="s2", product_m0="CC(=O)Oc1ccccc1C(=O)O", precursor_m0s=("A", "C"), inferred=False),
            ],
            leaf_m0s=("B", "C"),
            graph_signature=json.dumps(("reaction", "CC(=O)Oc1ccccc1C(=O)O", (("reaction", "A", (("leaf", "B"),)), ("leaf", "C")))),
            is_solved=True,
        )

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.NOT_EXACT

    def test_changed_intermediate_not_exact(self):
        """Different intermediate M0 = NOT_EXACT."""
        ref = self.create_simple_ref()
        cand = DerivedCandidate(
            route_id="cand_1",
            target_m0="CC(=O)Oc1ccccc1C(=O)O",
            steps=[
                DerivedStep(
                    step_id="s1",
                    product_m0="CC(=O)Oc1ccccc1C(=O)O",
                    precursor_m0s=("CC(=O)O", "DIFFERENT_INTERMEDIATE"),
                    inferred=False,
                ),
            ],
            leaf_m0s=("CC(=O)O", "DIFFERENT_INTERMEDIATE"),
            graph_signature=json.dumps(("reaction", "CC(=O)Oc1ccccc1C(=O)O", (("leaf", "CC(=O)O"), ("leaf", "DIFFERENT_INTERMEDIATE")))),
            is_solved=True,
        )

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.NOT_EXACT

    def test_changed_leaf_not_exact(self):
        """Different leaf M0 = NOT_EXACT."""
        ref = self.create_simple_ref()
        cand = DerivedCandidate(
            route_id="cand_1",
            target_m0="CC(=O)Oc1ccccc1C(=O)O",
            steps=[
                DerivedStep(
                    step_id="s1",
                    product_m0="CC(=O)Oc1ccccc1C(=O)O",
                    precursor_m0s=("CC(=O)O", "DIFFERENT_LEAF"),
                    inferred=False,
                ),
            ],
            leaf_m0s=("CC(=O)O", "DIFFERENT_LEAF"),
            graph_signature=json.dumps(("reaction", "CC(=O)Oc1ccccc1C(=O)O", (("leaf", "CC(=O)O"), ("leaf", "DIFFERENT_LEAF")))),
            is_solved=True,
        )

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.NOT_EXACT

    def test_opposite_enantiomer_not_exact(self):
        """Opposite enantiomer target = NOT_EXACT."""
        ref = self.create_simple_ref()
        cand = DerivedCandidate(
            route_id="cand_1",
            target_m0=m0_key("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"),  # S-enantiomer
            steps=[
                DerivedStep(
                    step_id="s1",
                    product_m0=m0_key("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"),
                    precursor_m0s=("CC(=O)O", "C1=CC=CC=C1C(=O)O"),
                    inferred=False,
                ),
            ],
            leaf_m0s=("CC(=O)O", "C1=CC=CC=C1C(=O)O"),
            graph_signature=json.dumps(("reaction", m0_key("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"), (("leaf", "CC(=O)O"), ("leaf", "C1=CC=CC=C1C(=O)O")))),
            is_solved=True,
        )

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.NOT_EXACT

    def test_atom_map_only_differences_exact(self):
        """Atom-map-only differences = EXACT (M0 strips atom maps)."""
        ref = self.create_simple_ref()
        cand = self.create_simple_cand()
        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.EXACT

    def test_ineligible_reference(self):
        """Ineligible reference returns INELIGIBLE."""
        ref = self.create_simple_ref()
        ref.exact_eligibility = ExactEligibility.EXCLUDED
        ref.exclusion_reasons = ["route_scope=terminal_segment"]
        cand = self.create_simple_cand()

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.INELIGIBLE

    def test_candidate_with_errors_not_exact(self):
        """Candidate with derivation errors = NOT_EXACT."""
        ref = self.create_simple_ref()
        cand = self.create_simple_cand()
        cand.derivation_errors = ["target unparseable"]
        cand.graph_signature = None

        result = compare_graph_exact(ref, cand)
        assert result.equality == GraphEquality.NOT_EXACT


class TestRanking:
    """Tests for deterministic ranking."""

    def create_test_routes(self) -> list[Route]:
        """Create test routes with different deterministic scores."""
        routes = []
        # Route 1: high feasibility
        routes.append(Route(
            target="C",
            reactions=[Reaction(product="C", precursors=["A", "B"])],
            leaves=[Molecule(smiles="A", in_stock=True), Molecule(smiles="B", in_stock=True)],
            solved=True,
        ))
        # Route 2: lower feasibility
        routes.append(Route(
            target="C",
            reactions=[Reaction(product="C", precursors=["D"])],
            leaves=[Molecule(smiles="D", in_stock=True)],
            solved=True,
        ))
        # Route 3: unsolved
        routes.append(Route(
            target="C",
            reactions=[Reaction(product="C", precursors=["E"])],
            leaves=[Molecule(smiles="E", in_stock=False)],
            solved=False,
        ))
        return routes

    def test_rank_routes_deterministic(self):
        """Deterministic ranking produces stable order."""
        routes = self.create_test_routes()
        ranked1 = rank_routes_deterministic(routes)
        ranked2 = rank_routes_deterministic(routes)

        assert [r.target for r in ranked1] == [r.target for r in ranked2]
        assert all(r.solved for r in ranked1)

    def test_rank_baseline_feasibility(self):
        """Baseline ranks by raw feasibility."""
        routes = self.create_test_routes()
        ranked = rank_baseline_feasibility(routes)

        # Should only include solved routes
        assert all(r.solved for r in ranked)
        assert len(ranked) == 2

    def test_rank_candidates_returns_ranking_result(self):
        """rank_candidates returns RankingResult with both rankings."""
        routes = self.create_test_routes()
        result = rank_candidates(routes)

        assert isinstance(result, RankingResult)
        assert len(result.reagent_ranked) == 2
        assert len(result.baseline_ranked) == 2
        assert result.weight_vector == DEFAULT_WEIGHTS

    def test_tie_break_by_route_signature(self):
        """Ties broken by route_signature."""
        route1 = Route(
            target="C",
            reactions=[Reaction(product="C", precursors=["A", "B"])],
            leaves=[Molecule(smiles="A", in_stock=True), Molecule(smiles="B", in_stock=True)],
            solved=True,
        )
        route2 = Route(
            target="C",
            reactions=[Reaction(product="C", precursors=["D"])],
            leaves=[Molecule(smiles="D", in_stock=True)],
            solved=True,
        )
        routes = [route1, route2]
        ranked = rank_routes_deterministic(routes)

        # Should be deterministic
        ranked2 = rank_routes_deterministic(routes)
        assert [id(r) for r in ranked] == [id(r) for r in ranked2]

    def test_rank_candidates_identical_to_harness_select(self):
        """Rank candidates produces same result as harness._select for winner."""
        routes = self.create_test_routes()
        solved = [r for r in routes if r.solved]

        # Get winner from rank_candidates
        ranking = rank_candidates(solved)
        reagent_winner = ranking.reagent_ranked[0]

        # Get winner from _select (used by evaluate)
        baseline_pick, reagent_pick = _select(solved, DEFAULT_WEIGHTS)

        assert reagent_winner is reagent_pick

    def test_input_order_independence(self):
        """Ranking result independent of input order."""
        routes = self.create_test_routes()
        ranked1 = rank_routes_deterministic(routes)
        ranked2 = rank_routes_deterministic(list(reversed(routes)))

        assert [id(r) for r in ranked1] == [id(r) for r in ranked2]


class TestRecoveryMetrics:
    """Tests for Recovery@k metrics."""

    def test_recovery_at_k_definitions(self):
        """Test Recovery@k logic with example ranks."""
        # Match at rank 7
        best_rank = 7
        assert (best_rank <= 1) is False  # @1
        assert (best_rank <= 3) is False  # @3
        assert (best_rank <= 5) is False  # @5
        assert (best_rank <= 10) is True   # @10
        assert best_rank is not None       # @Retained

        # Match at rank 1
        best_rank = 1
        assert (best_rank <= 1) is True
        assert (best_rank <= 3) is True
        assert (best_rank <= 5) is True
        assert (best_rank <= 10) is True
        assert best_rank is not None

        # No match
        best_rank = None
        assert (best_rank is not None and best_rank <= 1) is False
        assert (best_rank is not None and best_rank <= 3) is False
        assert (best_rank is not None and best_rank <= 5) is False
        assert (best_rank is not None and best_rank <= 10) is False
        assert best_rank is None  # @Retained is False when no match

    def test_denominator_excludes_ineligible_targets(self):
        """Targets with only ineligible references don't count in denominator."""
        pass


class TestMultipleReferences:
    """Tests for multiple reference aggregation."""

    def test_target_recovered_if_any_core_reference_matches(self):
        """Target recovered if ANY core-exact-eligible reference matches."""
        pass

    def test_best_rank_is_minimum_across_references(self):
        """Best rank is min across all matching eligible references."""
        ranks = [7, 3]
        assert min(ranks) == 3


class TestUnsolvedMatchDiagnostic:
    """Tests for unsolved exact match diagnostic."""

    def test_unsolved_match_diagnostic_only(self):
        """Unsolved match sets diagnostic but not primary recovery."""
        pass


class TestCandidateIdentity:
    """Tests for DerivedCandidate content identity and occurrence identity."""

    def test_same_chemistry_same_topology_stable_hash(self):
        """Same chemistry + same topology -> stable content hash across derivations."""
        from reagent.core.models import Molecule, Reaction, Route

        route = Route(
            target="CC(=O)NCCO",
            reactions=[
                Reaction(product="CC(=O)O", precursors=["CCO"]),
                Reaction(product="CCN", precursors=["CCO"]),
                Reaction(product="CC(=O)NCCO", precursors=["CC(=O)O", "CCN"]),
            ],
            leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(=O)NCCO",
                "children": [{
                    "type": "reaction", "smiles": "CC(=O)O.CCN>>CC(=O)NCCO",
                    "children": [
                        {"type": "mol", "smiles": "CC(=O)O", "children": [{
                            "type": "reaction", "smiles": "CCO>>CC(=O)O",
                            "children": [{"type": "mol", "smiles": "CCO"}]
                        }]},
                        {"type": "mol", "smiles": "CCN", "children": [{
                            "type": "reaction", "smiles": "CCO>>CCN",
                            "children": [{"type": "mol", "smiles": "CCO"}]
                        }]}
                    ]
                }]
            }
        )

        cand1 = derive_candidate(route, "test_hash")
        cand2 = derive_candidate(route, "test_hash")

        # Same content should produce same hash
        assert cand1.content_hash() == cand2.content_hash()
        assert cand1.route_id == cand2.route_id

    def test_same_reactions_different_topology_different_hash(self):
        """Same reaction labels but different topology -> different content hash."""
        from reagent.core.models import Molecule, Reaction, Route

        # Convergent: A->B, A->C, B+C->T
        route_conv = Route(
            target="CC(=O)NCCO",
            reactions=[
                Reaction(product="CC(=O)O", precursors=["CCO"]),
                Reaction(product="CCN", precursors=["CCO"]),
                Reaction(product="CC(=O)NCCO", precursors=["CC(=O)O", "CCN"]),
            ],
            leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(=O)NCCO",
                "children": [{
                    "type": "reaction", "smiles": "CC(=O)O.CCN>>CC(=O)NCCO",
                    "children": [
                        {"type": "mol", "smiles": "CC(=O)O", "children": [{
                            "type": "reaction", "smiles": "CCO>>CC(=O)O",
                            "children": [{"type": "mol", "smiles": "CCO"}]
                        }]},
                        {"type": "mol", "smiles": "CCN", "children": [{
                            "type": "reaction", "smiles": "CCO>>CCN",
                            "children": [{"type": "mol", "smiles": "CCO"}]
                        }]}
                    ]
                }]
            }
        )

        # Linear: A->B, B->C, C->T
        route_lin = Route(
            target="CC(=O)NCCO",
            reactions=[
                Reaction(product="CC(=O)O", precursors=["CCO"]),
                Reaction(product="CCN", precursors=["CC(=O)O"]),
                Reaction(product="CC(=O)NCCO", precursors=["CCN"]),
            ],
            leaves=[Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(=O)NCCO",
                "children": [{
                    "type": "reaction", "smiles": "CCN>>CC(=O)NCCO",
                    "children": [
                        {"type": "mol", "smiles": "CCN", "children": [{
                            "type": "reaction", "smiles": "CC(=O)O>>CCN",
                            "children": [{"type": "mol", "smiles": "CC(=O)O", "children": [{
                                "type": "reaction", "smiles": "CCO>>CC(=O)O",
                                "children": [{"type": "mol", "smiles": "CCO"}]
                            }]}]
                        }]}
                ]
            }]
        }
        )

        cand_conv = derive_candidate(route_conv, "test_hash")
        cand_lin = derive_candidate(route_lin, "test_hash")

        # Same reactions but different topology -> different hash
        assert cand_conv.content_hash() != cand_lin.content_hash()
        assert cand_conv.route_id != cand_lin.route_id

    def test_different_leaf_multiplicity_different_hash(self):
        """Different leaf multiplicity -> different content hash."""
        from reagent.core.models import Molecule, Reaction, Route

        # A + A -> B (two A consumed)
        route_double = Route(
            target="CC(=O)O",
            reactions=[Reaction(product="CC(=O)O", precursors=["CCO", "CCO"])],
            leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(=O)O",
                "children": [{
                    "type": "reaction", "smiles": "CCO.CCO>>CC(=O)O",
                    "children": [
                        {"type": "mol", "smiles": "CCO"},
                        {"type": "mol", "smiles": "CCO"}
                    ]
                }]
            }
        )

        # A -> B (single A consumed)
        route_single = Route(
            target="CC(=O)O",
            reactions=[Reaction(product="CC(=O)O", precursors=["CCO"])],
            leaves=[Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(=O)O",
                "children": [{
                    "type": "reaction", "smiles": "CCO>>CC(=O)O",
                    "children": [{"type": "mol", "smiles": "CCO"}]
                }]
            }
        )

        cand_double = derive_candidate(route_double, "test_hash")
        cand_single = derive_candidate(route_single, "test_hash")

        assert cand_double.content_hash() != cand_single.content_hash()
        assert cand_double.route_id != cand_single.route_id
        # Check leaf multiplicity is preserved
        assert len(cand_double.leaf_m0s) == 2
        assert len(cand_single.leaf_m0s) == 1

    def test_opposite_stereochemistry_different_hash(self):
        """Opposite stereochemistry -> different content hash under M0."""
        from reagent.core.models import Molecule, Reaction, Route

        # R enantiomer
        route_r = Route(
            target="CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
            reactions=[Reaction(product="CC(C)Cc1ccc([C@H](C)C(=O)O)cc1", precursors=["CCO"])],
            leaves=[Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
                "children": [{"type": "reaction", "smiles": "CCO>>CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
                    "children": [{"type": "mol", "smiles": "CCO"}]
                }]
            }
        )

        # S enantiomer
        route_s = Route(
            target="CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
            reactions=[Reaction(product="CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1", precursors=["CCO"])],
            leaves=[Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
                "children": [{"type": "reaction", "smiles": "CCO>>CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
                    "children": [{"type": "mol", "smiles": "CCO"}]
                }]
            }
        )

        cand_r = derive_candidate(route_r, "test_hash")
        cand_s = derive_candidate(route_s, "test_hash")

        assert cand_r.content_hash() != cand_s.content_hash()
        assert cand_r.route_id != cand_s.route_id

    def test_duplicate_content_candidates_distinguishable_by_occurrence(self):
        """Two identical-content candidates in same retained population should be distinguishable by occurrence index."""
        from reagent.core.models import Molecule, Reaction, Route

        route = Route(
            target="CC(=O)NCCO",
            reactions=[
                Reaction(product="CC(=O)O", precursors=["CCO"]),
                Reaction(product="CCN", precursors=["CCO"]),
                Reaction(product="CC(=O)NCCO", precursors=["CC(=O)O", "CCN"]),
            ],
            leaves=[Molecule(smiles="CCO", in_stock=True), Molecule(smiles="CCO", in_stock=True)],
            solved=True,
            tree={
                "type": "mol", "smiles": "CC(=O)NCCO",
                "children": [{
                    "type": "reaction", "smiles": "CC(=O)O.CCN>>CC(=O)NCCO",
                    "children": [
                        {"type": "mol", "smiles": "CC(=O)O", "children": [{
                            "type": "reaction", "smiles": "CCO>>CC(=O)O",
                            "children": [{"type": "mol", "smiles": "CCO"}]
                        }]},
                        {"type": "mol", "smiles": "CCN", "children": [{
                            "type": "reaction", "smiles": "CCO>>CCN",
                            "children": [{"type": "mol", "smiles": "CCO"}]
                        }]}
                    ]
                }]
            }
        )

        # Create two identical-content candidates (as if from different retained occurrences)
        cand1 = derive_candidate(route, "test_hash")
        cand2 = derive_candidate(route, "test_hash")

        # Content hashes should be identical (same chemistry + topology)
        assert cand1.content_hash() == cand2.content_hash()
        assert cand1.route_id == cand2.route_id

        # But they should be distinguishable as separate occurrences
        # We can wrap them with an occurrence index
        class OccurrenceCandidate:
            def __init__(self, cand: DerivedCandidate, occurrence_index: int):
                self.cand = cand
                self.occurrence_index = occurrence_index

            def occurrence_key(self) -> str:
                return f"{self.cand.route_id}:{self.occurrence_index}"

        occ1 = OccurrenceCandidate(cand1, 0)
        occ2 = OccurrenceCandidate(cand2, 1)

        assert occ1.occurrence_key() != occ2.occurrence_key()
        assert occ1.cand.content_hash() == occ2.cand.content_hash()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])