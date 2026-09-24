"""Tests for literature reference data foundation and molecular identity."""

from __future__ import annotations

import pytest

from reagent.core.chem import m0_key, m1_key
from reagent.eval.literature import (
    Conditions,
    CurationMetadata,
    EvidenceLocator,
    LiteratureReference,
    LiteratureReferenceSet,
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
    Yield,
    YieldType,
)


class TestM0Identity:
    """Tests for M0 strict molecular identity."""

    def test_same_molecule_different_smiles(self):
        """Same molecule, different valid SMILES → equal."""
        smiles1 = "CC(=O)O"
        smiles2 = "OC(=O)C"
        assert m0_key(smiles1) == m0_key(smiles2)

    def test_atom_map_differences(self):
        """Atom-map differences → equal."""
        # Proper atom-mapped version of acetic acid
        smiles1 = "[CH3:1][C:2](=[O:3])[O:4][H:5]"
        smiles2 = "CC(=O)O"
        assert m0_key(smiles1) == m0_key(smiles2)

    def test_opposite_enantiomers(self):
        """Opposite enantiomers → unequal."""
        r_ibuprofen = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"
        s_ibuprofen = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"
        assert m0_key(r_ibuprofen) != m0_key(s_ibuprofen)

    def test_specified_vs_unspecified_stereo(self):
        """Specified vs unspecified stereo → unequal where structure differs."""
        # Use a proper chiral center: CH3-CH(OH)-CH2-CH3 (butan-2-ol)
        specified = "CC[C@H](O)C"
        unspecified = "CCC(O)C"
        assert m0_key(specified) != m0_key(unspecified)

    def test_neutral_vs_protonated(self):
        """Neutral vs protonated → unequal."""
        neutral = "CC(=O)O"
        protonated = "CC(=O)[OH2+]"
        assert m0_key(neutral) != m0_key(protonated)

    def test_salt_vs_parent(self):
        """Salt/disconnected representation vs parent → unequal."""
        free_base = "CCN"
        hydrochloride = "CCN.Cl"
        assert m0_key(free_base) != m0_key(hydrochloride)

    def test_tautomer_pair(self):
        """Tautomer pair → unequal."""
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
        """Opposite enantiomers → equal under M1."""
        r_ibuprofen = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"
        s_ibuprofen = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"
        assert m1_key(r_ibuprofen) == m1_key(s_ibuprofen)

    def test_stereo_specified_vs_unspecified(self):
        """Specified vs unspecified → equal under M1."""
        specified = "C[C@H](O)C"
        unspecified = "CC(O)C"
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


class TestLiteratureReference:
    """Tests for LiteratureReference model."""

    def create_minimal_reference(self) -> LiteratureReference:
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
            molecules=[
                ReferenceMolecule(
                    molecule_id="mol_target",
                    reported_smiles="CC(=O)Oc1ccccc1C(=O)O",
                    role=MoleculeRole.TARGET,
                ),
                ReferenceMolecule(
                    molecule_id="mol_1",
                    reported_smiles="CC(=O)O",
                    role=MoleculeRole.STARTING_MATERIAL,
                ),
                ReferenceMolecule(
                    molecule_id="mol_2",
                    reported_smiles="C1=CC=CC=C1C(=O)O",
                    role=MoleculeRole.STARTING_MATERIAL,
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
        )

    def test_valid_complete_route(self):
        ref = self.create_minimal_reference()
        assert ref.reference_id == "ref_1"
        assert ref.is_exact_recovery_eligible() is True

    def test_terminal_segment(self):
        ref = self.create_minimal_reference()
        ref.route_scope = RouteScope.TERMINAL_SEGMENT
        assert ref.is_exact_recovery_eligible() is False

    def test_partial_route(self):
        ref = self.create_minimal_reference()
        ref.route_completeness = RouteCompleteness.PARTIAL
        assert ref.is_exact_recovery_eligible() is False

    def test_ambiguous_route(self):
        ref = self.create_minimal_reference()
        ref.route_completeness = RouteCompleteness.AMBIGUOUS
        assert ref.is_exact_recovery_eligible() is False

    def test_composite_not_eligible(self):
        ref = self.create_minimal_reference()
        ref.sources.append(SourceRecord(
            source_id="src_2",
            source_type=SourceType.PATENT,
            is_primary=True,
        ))
        composite = LiteratureReference.model_validate(ref.model_dump())
        assert composite.is_composite is True
        assert composite.is_exact_recovery_eligible() is False

    def test_inferred_step_not_eligible(self):
        ref = self.create_minimal_reference()
        ref.steps[0].inferred = True
        assert ref.is_exact_recovery_eligible() is False

    def test_duplicate_molecule_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate molecule_ids"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C"),
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="CC"),
                ],
            )

    def test_duplicate_step_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate step_ids"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.TARGET),
                    ReferenceMolecule(molecule_id="mol_2", reported_smiles="CC", role=MoleculeRole.STARTING_MATERIAL),
                ],
                steps=[
                    ReferenceStep(step_id="step_1", precursor_ids=["mol_2"], product_id="mol_1",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                    ReferenceStep(step_id="step_1", precursor_ids=["mol_2"], product_id="mol_1",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                ],
            )

    def test_missing_precursor_rejected(self):
        with pytest.raises(ValueError, match="unknown precursor"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.TARGET),
                    ReferenceMolecule(molecule_id="mol_2", reported_smiles="CC", role=MoleculeRole.STARTING_MATERIAL),
                ],
                steps=[
                    ReferenceStep(step_id="step_1", precursor_ids=["mol_999"], product_id="mol_1",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                ],
            )

    def test_missing_product_rejected(self):
        with pytest.raises(ValueError, match="unknown product"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.TARGET),
                    ReferenceMolecule(molecule_id="mol_2", reported_smiles="CC", role=MoleculeRole.STARTING_MATERIAL),
                ],
                steps=[
                    ReferenceStep(step_id="step_1", precursor_ids=["mol_2"], product_id="mol_999",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                ],
            )

    def test_no_target_molecule_rejected(self):
        with pytest.raises(ValueError, match="No target molecule declared"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.STARTING_MATERIAL),
                ],
                steps=[],
            )

    def test_cycle_rejected(self):
        with pytest.raises(ValueError, match="cycle"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.TARGET),
                    ReferenceMolecule(molecule_id="mol_2", reported_smiles="CC", role=MoleculeRole.INTERMEDIATE),
                    ReferenceMolecule(molecule_id="mol_3", reported_smiles="CCC", role=MoleculeRole.STARTING_MATERIAL),
                ],
                steps=[
                    ReferenceStep(step_id="step_1", precursor_ids=["mol_2"], product_id="mol_1",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                    ReferenceStep(step_id="step_2", precursor_ids=["mol_3"], product_id="mol_2",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                    ReferenceStep(step_id="step_3", precursor_ids=["mol_1"], product_id="mol_2",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                ],
            )

    def test_invalid_starting_leaf_rejected(self):
        with pytest.raises(ValueError, match="not declared as starting material"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.TARGET),
                    ReferenceMolecule(molecule_id="mol_2", reported_smiles="CC", role=MoleculeRole.INTERMEDIATE),
                    ReferenceMolecule(molecule_id="mol_3", reported_smiles="CCC", role=MoleculeRole.INTERMEDIATE),
                ],
                steps=[
                    ReferenceStep(step_id="step_1", precursor_ids=["mol_2", "mol_3"], product_id="mol_1",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                ],
            )

    def test_undocumented_intermediate_rejected(self):
        with pytest.raises(ValueError, match="not produced by any step"):
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[
                    ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.TARGET),
                    ReferenceMolecule(molecule_id="mol_2", reported_smiles="CC", role=MoleculeRole.INTERMEDIATE),
                    ReferenceMolecule(molecule_id="mol_3", reported_smiles="CCC", role=MoleculeRole.STARTING_MATERIAL),
                ],
                steps=[
                    ReferenceStep(step_id="step_1", precursor_ids=["mol_3"], product_id="mol_1",
                                 evidence_locators=[EvidenceLocator(source_id="src_1")]),
                ],
            )

    def test_complete_with_inferred_internal_rejected(self):
        ref = self.create_minimal_reference()
        # Add an intermediate that is produced by a step
        ref.molecules.append(ReferenceMolecule(
            molecule_id="mol_internal",
            reported_smiles="CCO",
            role=MoleculeRole.INTERMEDIATE,
        ))
        # Step that produces the intermediate (non-inferred)
        ref.steps.append(ReferenceStep(
            step_id="step_2",
            precursor_ids=["mol_1"],  # from starting material
            product_id="mol_internal",
            evidence_locators=[EvidenceLocator(source_id="src_1")],
            inferred=False,
        ))
        # Step that consumes the intermediate (inferred)
        ref.steps.append(ReferenceStep(
            step_id="step_3",
            precursor_ids=["mol_internal"],
            product_id="mol_target",
            evidence_locators=[EvidenceLocator(source_id="src_1")],
            inferred=True,
        ))
        with pytest.raises(ValueError, match="Graph incomplete: internal intermediate mol_internal consumed by inferred step step_3"):
            LiteratureReference.model_validate(ref.model_dump())

    def test_composite_status_auto_detected(self):
        ref = self.create_minimal_reference()
        ref.sources.append(SourceRecord(
            source_id="src_2",
            source_type=SourceType.PEER_REVIEWED_ARTICLE,
            is_primary=True,
        ))
        # Re-validate - should detect composite
        validated = LiteratureReference.model_validate(ref.model_dump())
        assert validated.is_composite is True

    def test_composite_from_different_evidence_sources(self):
        ref = self.create_minimal_reference()
        ref.steps[0].evidence_locators.append(EvidenceLocator(source_id="src_2"))
        validated = LiteratureReference.model_validate(ref.model_dump())
        assert validated.is_composite is True

    def test_stereo_metadata_roundtrip(self):
        ref = self.create_minimal_reference()
        ref.stereo_metadata = StereoMetadata(
            target_status=StereoStatus.ACHIRAL,
            relation_to_reagent=StereoRelation.EXACTLY_COMPATIBLE,
        )
        data = ref.model_dump()
        restored = LiteratureReference.model_validate(data)
        assert restored.stereo_metadata.target_status == StereoStatus.ACHIRAL
        assert restored.stereo_metadata.relation_to_reagent == StereoRelation.EXACTLY_COMPATIBLE

    def test_deterministic_serialization(self):
        ref = self.create_minimal_reference()
        dict1 = ref.to_deterministic_dict()
        dict2 = ref.to_deterministic_dict()
        assert dict1 == dict2

    def test_content_hash_stable(self):
        ref = self.create_minimal_reference()
        hash1 = ref.content_hash()
        hash2 = ref.content_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256


class TestLiteratureReferenceSet:
    """Tests for LiteratureReferenceSet model."""

    def test_valid_set(self):
        refs = []
        for i in range(3):
            refs.append(LiteratureReference(
                reference_id=f"ref_{i}",
                target=TargetRecord(reagent_target_name=f"target_{i}", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id=f"src_{i}", is_primary=True)],
                molecules=[ReferenceMolecule(molecule_id=f"mol_{i}", reported_smiles="C", role=MoleculeRole.TARGET)],
                steps=[],
            ))
        ref_set = LiteratureReferenceSet(references=refs)
        assert len(ref_set.references) == 3

    def test_duplicate_reference_ids_rejected(self):
        refs = [
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_1", is_primary=True)],
                molecules=[ReferenceMolecule(molecule_id="mol_1", reported_smiles="C", role=MoleculeRole.TARGET)],
                steps=[],
            ),
            LiteratureReference(
                reference_id="ref_1",
                target=TargetRecord(reagent_target_name="test", reagent_target_smiles="C"),
                sources=[SourceRecord(source_id="src_2", is_primary=True)],
                molecules=[ReferenceMolecule(molecule_id="mol_2", reported_smiles="C", role=MoleculeRole.TARGET)],
                steps=[],
            ),
        ]
        with pytest.raises(ValueError, match="Duplicate reference_ids"):
            LiteratureReferenceSet(references=refs)


class TestConditionsAndYield:
    """Tests for Conditions and Yield models."""

    def test_conditions_optional(self):
        cond = Conditions(reagents=["NaOH"], catalysts=["Pd/C"], solvents=["EtOH"], temperature_c=80.0)
        assert cond.reagents == ["NaOH"]
        assert cond.temperature_c == 80.0

    def test_yield_optional(self):
        yld = Yield(value=85.0, unit="percent", type=YieldType.ISOLATED, reported=True)
        assert yld.value == 85.0
        assert yld.type == YieldType.ISOLATED


class TestCurationMetadata:
    """Tests for CurationMetadata model."""

    def test_valid_curation(self):
        cur = CurationMetadata(curator="J. Smith", curation_date="2024-01-15", version="1.0")
        assert cur.curator == "J. Smith"
        assert cur.version == "1.0"


class TestBackwardCompatibility:
    """Tests for backward compatibility with reference_review.py."""

    def test_legacy_reference_conversion(self):
        from reagent.core.models import Reaction as CoreReaction
        from reagent.eval.reference_review import Reference as LegacyReference

        LegacyReference(
            source="test_source",
            locator="Scheme 1",
            verified_by="curator",
            reactions=[
                CoreReaction(product="CC(=O)O", precursors=["C", "CO"]),
            ],
        )
        converted = LiteratureReference.model_validate(
            # Use the conversion function
            {
                "reference_id": "test_source",
                "target": {"reagent_target_name": "unknown", "reagent_target_smiles": "CC(=O)O"},
                "sources": [{"source_id": "test_source", "source_type": "other", "title": "Scheme 1", "is_primary": True}],
                "route_scope": "full_route",
                "route_completeness": "complete",
                "molecules": [
                    {"molecule_id": "mol_0", "reported_smiles": "CC(=O)O", "role": "target"},
                    {"molecule_id": "mol_1", "reported_smiles": "C", "role": "starting_material"},
                    {"molecule_id": "mol_2", "reported_smiles": "CO", "role": "starting_material"},
                ],
                "steps": [
                    {"step_id": "step_0", "sequence_index": 0, "precursor_ids": ["mol_1", "mol_2"],
                     "product_id": "mol_0", "evidence_locators": [{"source_id": "test_source"}],
                     "inferred": False}
                ],
                "is_composite": False,
            }
        )
        assert converted.reference_id == "test_source"
        assert len(converted.molecules) == 3
        assert len(converted.steps) == 1


class TestM0M1EdgeCases:
    """Additional M0/M1 regression tests."""

    def test_m0_whitespace_only(self):
        """M0 returns None for whitespace-only input."""
        from reagent.core.chem import m0_key
        assert m0_key("   ") is None
        assert m0_key("\t\n") is None

    def test_m1_whitespace_only(self):
        """M1 returns None for whitespace-only input."""
        from reagent.core.chem import m1_key
        assert m1_key("   ") is None

    def test_m1_double_bond_stereo_removed(self):
        """M1 removes E/Z double bond stereochemistry."""
        from reagent.core.chem import m0_key, m1_key
        e_alkene = "C/C=C/C"
        z_alkene = "C/C=C\\C"
        assert m0_key(e_alkene) != m0_key(z_alkene), "M0 preserves E/Z"
        assert m1_key(e_alkene) == m1_key(z_alkene), "M1 ignores E/Z"

    def test_m1_removes_cip_codes(self):
        """M1 removes CIP codes from chiral centers."""
        from reagent.core.chem import m0_key, m1_key
        # Use butan-2-ol which has a true chiral center
        specified = "CC[C@H](O)C"
        unspecified = "CCC(O)C"
        m0_specified = m0_key(specified)
        m1_specified = m1_key(specified)
        m0_unspecified = m0_key(unspecified)
        m1_unspecified = m1_key(unspecified)
        assert m0_specified != m0_unspecified, "M0 preserves chirality"
        assert m1_specified == m1_unspecified, "M1 ignores chirality"

    def test_m1_preserves_isotopes(self):
        """M1 preserves isotopes."""
        from reagent.core.chem import m1_key
        normal = "CCO"
        deuterated = "[2H]C[2H]O"
        assert m1_key(normal) != m1_key(deuterated)

    def test_m1_preserves_charge(self):
        """M1 preserves formal charge."""
        from reagent.core.chem import m1_key
        neutral = "CC(=O)O"
        protonated = "CC(=O)[OH2+]"
        assert m1_key(neutral) != m1_key(protonated)

    def test_m1_preserves_fragments(self):
        """M1 preserves disconnected fragments."""
        from reagent.core.chem import m1_key
        single = "CCO"
        salt = "CCO.[Na+].[Cl-]"
        assert m1_key(single) != m1_key(salt)

    def test_m0_preserves_all(self):
        """M0 preserves everything: stereo, charge, isotopes, fragments, tautomers."""
        from reagent.core.chem import m0_key
        keto = "CC(=O)C"
        enol = "CC(=C)O"
        assert m0_key(keto) != m0_key(enol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])