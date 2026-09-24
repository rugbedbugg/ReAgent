"""Mapped derived artifacts: provenance, failure semantics, caching and alignment.

No test here needs an installed atom mapper. Automatic mapping is exercised with
test doubles that implement the RouteMapper protocol from fixed tables, and the
real rxnutils mapper is exercised only for its absence.
"""

from __future__ import annotations

import copy
import json

import pytest

import reagent.eval.literature_mapping as lm
from reagent.core.chem import m0_key
from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.literature import (
    EvidenceLocator,
    LiteratureReference,
    LiteratureReferenceSet,
    MoleculeRole,
    ReferenceMolecule,
    ReferenceStep,
    SourceRecord,
    SourceType,
    StereoMetadata,
    StereoRelation,
    StructureResolutionStatus,
    TargetRecord,
)
from reagent.eval.literature_derived import ExactEligibility, derive_candidate, derive_reference
from reagent.eval.literature_mapping import (
    MappedArtifactStore,
    MappedDerivedCandidate,
    MappedDerivedReference,
    MapperCapability,
    MapperIdentity,
    MapperUnavailableError,
    MappingStatus,
    RxnutilsRouteMapper,
    align_to_reference,
    map_candidate,
    map_reference,
    probe_rxnutils_mapper,
    resolve_reference,
    validate_route_mapping,
)
from reagent.eval.literature_similarity import SimilarityEvaluator, SimilarityStatus

# N-methylbenzamide made from benzoic acid through the acid chloride.
AMIDE = "CNC(=O)c1ccccc1"
CHLORIDE = "O=C(Cl)c1ccccc1"
ACID = "O=C(O)c1ccccc1"
THIONYL = "O=S(Cl)Cl"
AMINE = "CN"
AMIDE_MAPPED = "[CH3:10][NH:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1"
CHLORIDE_MAPPED = "[O:3]=[C:2](Cl)[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1"
# Forward form, as a curated mapping or a route mapper would give it
AMIDATION = f"{CHLORIDE_MAPPED}.[CH3:10][NH2:1]>>{AMIDE_MAPPED}"
CHLORINATION = (
    "[O:3]=[C:2](O)[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.O=S([Cl:11])Cl"
    ">>[O:3]=[C:2]([Cl:11])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1"
)
# The same product numbered unlike its parent reaction: O3 and C2 swapped
CHLORINATION_INCONSISTENT = (
    "[O:2]=[C:3](O)[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.O=S([Cl:11])Cl"
    ">>[O:2]=[C:3]([Cl:11])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1"
)
# Chlorine takes map number 10, which the target gives to its methyl carbon: a
# per-step template number reused across the route (seen in real checkpoints)
CHLORINATION_TARGET_REUSE = CHLORINATION.replace(":11]", ":10]")


def _mol(molecule_id, smiles, role):
    return ReferenceMolecule(molecule_id=molecule_id, reported_smiles=smiles, role=role,
                             structure_resolution=StructureResolutionStatus.RESOLVED)


def _reference(ref_id="ref_amide", two_step=True, composite=False, note=None) -> LiteratureReference:
    molecules = [_mol("t", AMIDE, MoleculeRole.TARGET), _mol("amine", AMINE, MoleculeRole.STARTING_MATERIAL)]
    steps = []
    if two_step:
        molecules += [_mol("chloride", CHLORIDE, MoleculeRole.INTERMEDIATE),
                      _mol("acid", ACID, MoleculeRole.STARTING_MATERIAL),
                      _mol("thionyl", THIONYL, MoleculeRole.STARTING_MATERIAL)]
        steps.append(ReferenceStep(step_id="s1", precursor_ids=["acid", "thionyl"], product_id="chloride",
                                   evidence_locators=[EvidenceLocator(source_id="src", example="1")]))
    else:
        molecules.append(_mol("chloride", CHLORIDE, MoleculeRole.STARTING_MATERIAL))
    steps.append(ReferenceStep(step_id="s2", precursor_ids=["chloride", "amine"], product_id="t",
                               evidence_locators=[EvidenceLocator(source_id="src", example="2")]))
    sources = [SourceRecord(source_id="src", source_type=SourceType.PATENT, is_primary=True)]
    if composite:
        sources.append(SourceRecord(source_id="src2", source_type=SourceType.PATENT))
    return LiteratureReference(
        reference_id=ref_id,
        target=TargetRecord(reagent_target_name="N-methylbenzamide", reagent_target_smiles=AMIDE),
        sources=sources,
        molecules=molecules,
        steps=steps,
        stereo_metadata=StereoMetadata(relation_to_reagent=StereoRelation.EXACTLY_COMPATIBLE),
        is_composite=composite,
        limitations=[note] if note else [],
    )


CURATED = MapperIdentity("curated-forward-maps", "test-1")
PREMAPPED = {"s1": CHLORINATION, "s2": AMIDATION}


def _mapped_reference(ref=None, premapped=PREMAPPED, mapper=None):
    ref = ref or _reference()
    return map_reference(ref, derive_reference(ref, "h"), mapper, premapped, CURATED)


class TableMapper:
    """RouteMapper test double: returns fixed forward mappings keyed by product M0."""

    def __init__(self, table, capability=MapperCapability.AVAILABLE, version="1", error=None):
        self.table, self._capability, self.error = table, capability, error
        self.identity = MapperIdentity("table-mapper", version)
        self.calls = 0

    def capability(self):
        return self._capability

    def map_tree(self, tree):
        self.calls += 1
        if self.error is not None:
            raise self.error
        tree = copy.deepcopy(tree)
        for mol, reaction in lm._reaction_nodes(tree):
            reaction["metadata"]["mapped_reaction_smiles"] = self.table[m0_key(mol["smiles"])]
        return tree


TABLE = {m0_key(AMIDE): AMIDATION, m0_key(CHLORIDE): CHLORINATION}


def _canon(rsmi: str) -> str:
    """Mapped reaction SMILES in RDKit canonical text, for comparing emitted forms."""
    from rdkit import Chem
    reactants, _, product = rsmi.split(">")
    return ">>".join(Chem.MolToSmiles(Chem.MolFromSmiles(part)) for part in (reactants, product))


def _retro(forward: str) -> str:
    reactants, _, product = forward.split(">")
    return f"{product}>>{reactants}"


def _candidate(maps: dict | None = None, tree: bool = True, amine=AMINE) -> Route:
    """Two-step candidate; ``maps`` gives forward mappings stored retro-form on the tree."""
    maps = maps or {}

    def reaction(product, precursors, children):
        metadata = {"mapped_reaction_smiles": _retro(maps[product])} if product in maps else {}
        return {"type": "reaction", "smiles": f"{product}>>{'.'.join(precursors)}",
                "metadata": metadata, "children": children}

    saved = {"type": "mol", "smiles": AMIDE, "children": [reaction(AMIDE, [CHLORIDE, amine], [
        {"type": "mol", "smiles": CHLORIDE, "children": [reaction(CHLORIDE, [ACID, THIONYL], [
            {"type": "mol", "smiles": ACID}, {"type": "mol", "smiles": THIONYL}])]},
        {"type": "mol", "smiles": amine},
    ])]}
    return Route(
        target=AMIDE,
        reactions=[Reaction(product=AMIDE, precursors=[CHLORIDE, amine]),
                   Reaction(product=CHLORIDE, precursors=[ACID, THIONYL])],
        leaves=[Molecule(smiles=s, in_stock=True) for s in (ACID, THIONYL, amine)],
        solved=True,
        tree=saved if tree else None,
    )


def _forward_tree(amidation=AMIDATION, chlorination=CHLORINATION) -> dict:
    return {"type": "mol", "smiles": AMIDE, "children": [{"type": "reaction", "metadata": {
        "mapped_reaction_smiles": amidation}, "children": [
        {"type": "mol", "smiles": CHLORIDE, "children": [{"type": "reaction", "metadata": (
            {"mapped_reaction_smiles": chlorination} if chlorination else {}), "children": [
            {"type": "mol", "smiles": ACID}, {"type": "mol", "smiles": THIONYL}]}]},
        {"type": "mol", "smiles": AMINE}]}]}


# ============================================================ data model


class TestArtifactModel:
    def test_successful_premapped_artifact_records_provenance(self):
        artifact = _mapped_reference()

        assert isinstance(artifact, MappedDerivedReference)
        assert artifact.mapping_status == MappingStatus.SUCCESS
        assert (artifact.mapping_tool, artifact.mapping_tool_version) == (CURATED.tool, CURATED.version)
        assert artifact.mapped_target_smiles == AMIDE_MAPPED
        assert artifact.mapped_reactions == [AMIDATION, CHLORINATION]
        assert artifact.normalization_policy == lm.NORMALIZATION_POLICY
        assert artifact.mapping_policy_version == lm.MAPPING_POLICY_VERSION
        assert artifact.source_digest == _reference().content_hash()
        assert artifact.derived_digest == derive_reference(_reference(), "h").content_hash()
        assert artifact.route_chemistry_digest and artifact.failure_reason is None
        assert artifact.artifact_digest == artifact.compute_digest()

    def test_artifact_holds_no_evaluation_state(self):
        fields = set(MappedDerivedReference.model_fields) | set(MappedDerivedCandidate.model_fields)
        for forbidden in ("rank", "score", "stock", "recovery", "search"):
            assert not any(forbidden in f for f in fields), forbidden

    def test_mapper_unavailable_artifact(self):
        mapper = RxnutilsRouteMapper(environ={"PATH": ""})
        artifact = _mapped_reference(premapped=None, mapper=mapper)

        assert artifact.mapping_status == MappingStatus.MAPPER_UNAVAILABLE
        assert artifact.mapped_tree is None and artifact.mapped_reactions == []
        assert artifact.route_tree is not None
        assert "unavailable" in artifact.failure_reason

    def test_mapping_failed_artifact_names_the_mapper(self):
        mapper = TableMapper(TABLE, error=ValueError("Assigning atom-mapping failed"))
        artifact = _mapped_reference(premapped=None, mapper=mapper)

        assert artifact.mapping_status == MappingStatus.MAPPING_FAILED
        assert artifact.mapping_tool == "table-mapper"
        assert "Assigning atom-mapping failed" in artifact.failure_reason
        assert artifact.mapped_tree is None

    def test_mapper_output_that_is_not_route_wide_fails(self):
        table = {**TABLE, m0_key(CHLORIDE): CHLORINATION_INCONSISTENT}
        artifact = _mapped_reference(premapped=None, mapper=TableMapper(table))

        assert artifact.mapping_status == MappingStatus.MAPPING_FAILED
        assert "not a route-wide mapping" in artifact.failure_reason

    def test_mapper_that_cannot_run_is_unavailable_not_failed(self):
        mapper = TableMapper(TABLE, error=MapperUnavailableError("conda: not found"))
        artifact = _mapped_reference(premapped=None, mapper=mapper)

        assert artifact.mapping_status == MappingStatus.MAPPER_UNAVAILABLE

    def test_success_requires_mapped_chemistry(self):
        data = json.loads(_mapped_reference().to_json())
        data.update(mapped_tree=None, artifact_digest="")
        with pytest.raises(ValueError):
            MappedDerivedReference.model_validate(data)

    def test_unresolved_reference_is_invalid_reference(self):
        ref = _reference()
        broken = ref.model_copy(update={"molecules": [
            m.model_copy(update={"reported_smiles": None, "m0": None, "m1": None,
                                 "structure_resolution": StructureResolutionStatus.NOT_REPORTED})
            if m.molecule_id == "amine" else m
            for m in ref.molecules]})
        artifact = map_reference(broken, derive_reference(broken, "h"))

        assert artifact.mapping_status == MappingStatus.INVALID_REFERENCE
        assert artifact.route_tree is None


class TestDigest:
    def test_serialization_is_deterministic(self):
        first, second = _mapped_reference(), _mapped_reference()

        assert first.artifact_digest == second.artifact_digest
        strip = lambda a: {k: v for k, v in json.loads(a.to_json()).items() if k != "created_at"}  # noqa: E731
        assert strip(first) == strip(second)
        assert list(json.loads(first.to_json())) == sorted(json.loads(first.to_json()))

    def test_round_trip_keeps_digest(self):
        artifact = _mapped_reference()
        loaded = MappedDerivedReference.model_validate_json(artifact.to_json())
        assert loaded.artifact_digest == artifact.artifact_digest

    def test_digest_excludes_timestamp(self):
        artifact = _mapped_reference()
        data = json.loads(artifact.to_json())
        data["created_at"] = "1999-01-01T00:00:00+00:00"
        assert MappedDerivedReference.model_validate(data).artifact_digest == artifact.artifact_digest

    def test_digest_changes_when_source_changes(self):
        changed = _mapped_reference(_reference(note="yield disputed"))
        assert changed.source_digest != _mapped_reference().source_digest
        assert changed.artifact_digest != _mapped_reference().artifact_digest

    def test_digest_changes_when_mapping_policy_changes(self, monkeypatch):
        before = _mapped_reference()
        monkeypatch.setattr(lm, "MAPPING_POLICY_VERSION", lm.MAPPING_POLICY_VERSION + 1)
        after = _mapped_reference()
        assert after.mapping_policy_version != before.mapping_policy_version
        assert after.artifact_digest != before.artifact_digest

    def test_digest_changes_when_mapped_output_changes(self):
        relabelled = {"s1": CHLORINATION.replace(":11]", ":12]"), "s2": AMIDATION}
        assert _mapped_reference(premapped=relabelled).artifact_digest != _mapped_reference().artifact_digest

    def test_tampered_artifact_is_rejected(self):
        data = json.loads(_mapped_reference().to_json())
        data["mapped_reactions"] = list(reversed(data["mapped_reactions"]))
        with pytest.raises(ValueError, match="artifact_digest"):
            MappedDerivedReference.model_validate(data)


# ============================================================ cache


class TestStore:
    def test_save_is_stable_and_loadable(self, tmp_path):
        store = MappedArtifactStore(tmp_path)
        artifact = _mapped_reference()
        path = store.save(artifact)

        assert path == store.path("reference", "ref_amide")
        assert json.loads(path.read_text()) == json.loads(artifact.to_json())
        assert store.load("reference", "ref_amide").artifact_digest == artifact.artifact_digest
        assert not list(path.parent.glob("*.tmp"))

    def test_environment_outcomes_are_not_cached(self, tmp_path):
        store = MappedArtifactStore(tmp_path)
        assert store.save(_mapped_reference(premapped=None)) is None
        assert store.load("reference", "ref_amide") is None

    def test_corrupt_file_is_ignored(self, tmp_path):
        store = MappedArtifactStore(tmp_path)
        path = store.save(_mapped_reference())
        path.write_text(path.read_text().replace(AMIDE_MAPPED, AMIDE_MAPPED.replace(":10]", ":99]"), 1))
        with pytest.warns(UserWarning, match="unreadable"):
            assert store.load("reference", "ref_amide") is None

    def test_current_cache_is_used_without_remapping(self, tmp_path):
        store = MappedArtifactStore(tmp_path)
        ref = _reference()
        cached = map_reference(ref, derive_reference(ref, "h"), TableMapper(TABLE))
        store.save(cached)
        mapper = TableMapper(TABLE)

        result = resolve_reference(ref, derive_reference(ref, "h"), mapper, store)

        assert result.artifact_digest == cached.artifact_digest
        assert mapper.calls == 0

    def test_artifact_from_a_mapping_environment_is_consumable_without_a_mapper(self, tmp_path):
        store = MappedArtifactStore(tmp_path)
        ref = _reference()
        store.save(map_reference(ref, derive_reference(ref, "h"), TableMapper(TABLE)))

        result = resolve_reference(ref, derive_reference(ref, "h"), RxnutilsRouteMapper(environ={}), store)

        assert result.mapping_status == MappingStatus.SUCCESS
        assert result.mapping_tool == "table-mapper"

    def _stale_setup(self, tmp_path):
        store = MappedArtifactStore(tmp_path)
        old = _reference()
        store.save(map_reference(old, derive_reference(old, "h"), TableMapper(TABLE)))
        return store, _reference(note="yield disputed")

    def test_stale_without_mapper_is_explicitly_stale(self, tmp_path):
        store, changed = self._stale_setup(tmp_path)

        result = resolve_reference(changed, derive_reference(changed, "h"), RxnutilsRouteMapper(environ={}), store)

        assert result.mapping_status == MappingStatus.STALE
        assert "source_digest" in result.failure_reason
        assert result.mapped_tree is None

    def test_stale_with_mapper_is_regenerated(self, tmp_path):
        store, changed = self._stale_setup(tmp_path)
        mapper = TableMapper(TABLE)

        result = resolve_reference(changed, derive_reference(changed, "h"), mapper, store)

        assert mapper.calls == 1
        assert result.mapping_status == MappingStatus.SUCCESS
        assert result.source_digest == changed.content_hash()
        assert store.load("reference", "ref_amide").source_digest == changed.content_hash()

    @pytest.mark.parametrize("field", ["mapping_policy_version", "normalization_policy",
                                       "rdkit_version", "reaction_utils_version"])
    def test_policy_and_environment_changes_make_cache_stale(self, tmp_path, monkeypatch, field):
        store = MappedArtifactStore(tmp_path)
        ref = _reference()
        store.save(map_reference(ref, derive_reference(ref, "h"), TableMapper(TABLE)))
        if field == "mapping_policy_version":
            monkeypatch.setattr(lm, "MAPPING_POLICY_VERSION", lm.MAPPING_POLICY_VERSION + 1)
        elif field == "normalization_policy":
            monkeypatch.setattr(lm, "NORMALIZATION_POLICY", "other")
        else:
            env = lm._environment()
            monkeypatch.setattr(lm, "_environment", lambda: {**env, field: "0.0"})

        assert lm.stale_fields(store.load("reference", "ref_amide"), ref.content_hash(),
                               derive_reference(ref, "h").content_hash()) == [field]

    def test_mapper_version_change_regenerates(self, tmp_path):
        store = MappedArtifactStore(tmp_path)
        ref = _reference()
        store.save(map_reference(ref, derive_reference(ref, "h"), TableMapper(TABLE, version="1")))
        newer = TableMapper(TABLE, version="2")

        result = resolve_reference(ref, derive_reference(ref, "h"), newer, store)

        assert newer.calls == 1
        assert result.mapping_tool_version == "2"

    def test_supplied_stale_artifact_is_not_used_by_the_evaluator(self, tmp_path):
        supplied = _mapped_reference()
        changed = _reference(note="yield disputed")
        checkpoint = Checkpoint(tmp_path / "cp", {"schema": 1})
        checkpoint.save(m0_key(AMIDE), [_candidate({AMIDE: AMIDATION, CHLORIDE: CHLORINATION})], False)
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[changed]), "h",
                                        mapper=RxnutilsRouteMapper(environ={}),
                                        mapped_references={"ref_amide": supplied})

        target = evaluator.evaluate(Checkpoint.open_readonly(tmp_path / "cp")).per_target[0]

        assert {m.mapping_status for m in target.reference_matches} == {"stale_mapping"}
        assert all(m.route_similarity is None for m in target.reference_matches)
        assert target.similarity_evaluable is False


# ============================================================ pre-mapped routes


class TestPreMapped:
    def test_premapped_reference_does_not_invoke_mapper(self):
        spy = TableMapper(TABLE)
        artifact = _mapped_reference(mapper=spy)

        assert artifact.mapping_status == MappingStatus.SUCCESS
        assert spy.calls == 0

    def test_premapped_candidate_does_not_invoke_mapper(self):
        spy = TableMapper(TABLE)
        route = _candidate({AMIDE: AMIDATION, CHLORIDE: CHLORINATION})
        artifact = map_candidate(route, derive_candidate(route, "h"), spy)

        assert artifact.mapping_status == MappingStatus.SUCCESS
        assert artifact.mapping_tool == "aizynthfinder-template-application"
        # The official reader re-emits the stored maps in canonical text
        assert [_canon(r) for r in artifact.mapped_reactions] == [_canon(AMIDATION), _canon(CHLORINATION)]
        assert spy.calls == 0

    def test_candidate_does_not_mutate_route(self):
        route = _candidate({AMIDE: AMIDATION, CHLORIDE: CHLORINATION})
        before = route.model_dump(mode="json")
        map_candidate(route, derive_candidate(route, "h"))
        assert route.model_dump(mode="json") == before

    def test_partial_premapped_steps_are_not_used(self):
        spy = TableMapper(TABLE)
        artifact = _mapped_reference(premapped={"s2": AMIDATION}, mapper=spy)

        assert any("do not cover the route" in w for w in artifact.warnings)
        # Falls through to the mapper, which maps the whole route from scratch
        assert spy.calls == 1 and artifact.mapping_tool == "table-mapper"

    def test_evaluator_similarity_matches_official_functions(self, tmp_path):
        """Pre-mapped end to end: artifacts in, official rxnutils numbers out."""
        import rxnutils.routes.comparison as comp
        import rxnutils.routes.readers as readers

        ref = _reference(two_step=False)
        artifact = map_reference(ref, derive_reference(ref, "h"), premapped={"s2": AMIDATION},
                                 premapped_identity=CURATED)
        # Benzoic acid route, numbered independently of the reference
        renumbered = {
            AMIDE: "[O:4]=[C:3]([OH:20])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[CH3:1][NH2:2]>>"
                   "[CH3:1][NH:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1",
        }
        route = Route(
            target=AMIDE, reactions=[Reaction(product=AMIDE, precursors=[ACID, AMINE])],
            leaves=[Molecule(smiles=s, in_stock=True) for s in (ACID, AMINE)], solved=True,
            tree={"type": "mol", "smiles": AMIDE, "children": [{
                "type": "reaction", "smiles": "", "metadata": {"mapped_reaction_smiles": _retro(renumbered[AMIDE])},
                "children": [{"type": "mol", "smiles": ACID}, {"type": "mol", "smiles": AMINE}]}]},
        )
        checkpoint = Checkpoint(tmp_path / "cp", {"schema": 1})
        checkpoint.save(m0_key(AMIDE), [route], False)
        spy = TableMapper(TABLE)
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[ref]), "h", mapper=spy,
                                        mapped_references={ref.reference_id: artifact})

        target = evaluator.evaluate(Checkpoint.open_readonly(tmp_path / "cp")).per_target[0]

        ref_route = readers.read_aizynthfinder_dict({"type": "mol", "smiles": AMIDE, "children": [{
            "type": "reaction", "metadata": {"mapped_reaction_smiles": _retro(AMIDATION)},
            "children": [{"type": "mol", "smiles": CHLORIDE}, {"type": "mol", "smiles": AMINE}]}]})
        cand_route = readers.read_aizynthfinder_dict(route.tree)
        cand_route.remap(ref_route)
        official = comp.simple_route_similarity([ref_route, cand_route])[0, 1]
        assert official == pytest.approx(1.0)
        assert target.selected_route_similarity == official
        assert target.max_similarity_retained == official
        assert target.baseline_selected_similarity == official
        assert spy.calls == 0


# ============================================================ shared target numbering


def _one_step_candidate(forward: str, precursors=(CHLORIDE, AMINE), target=AMIDE) -> MappedDerivedCandidate:
    route = Route(
        target=target, reactions=[Reaction(product=target, precursors=list(precursors))],
        leaves=[Molecule(smiles=s, in_stock=True) for s in precursors], solved=True,
        tree={"type": "mol", "smiles": target, "children": [{
            "type": "reaction", "smiles": "", "metadata": {"mapped_reaction_smiles": _retro(forward)},
            "children": [{"type": "mol", "smiles": s} for s in precursors]}]},
    )
    return map_candidate(route, derive_candidate(route, "h"))


def _one_step_reference():
    ref = _reference(two_step=False)
    return map_reference(ref, derive_reference(ref, "h"), premapped={"s2": AMIDATION}, premapped_identity=CURATED)


class TestSharedTargetNumbering:
    RENUMBERED = ("[O:4]=[C:3](Cl)[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[CH3:1][NH2:2]>>"
                  "[CH3:1][NH:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1")

    def test_independent_numbering_is_aligned_onto_the_reference(self):
        import rxnutils.routes.comparison as comp

        reference = _one_step_reference()
        renumbered = _one_step_candidate(self.RENUMBERED)
        same = _one_step_candidate(AMIDATION)
        assert renumbered.mapped_target_smiles != reference.mapped_target_smiles

        aligned = align_to_reference(reference, renumbered)
        control = align_to_reference(reference, same)

        assert aligned.status == MappingStatus.SUCCESS
        assert _canon(f">>{aligned.candidate_route.mapped_root_smiles}") == _canon(f">>{AMIDE_MAPPED}")
        pair = [aligned.reference_route, aligned.candidate_route]
        control_pair = [control.reference_route, control.candidate_route]
        assert comp.simple_route_similarity(pair)[0, 1] == comp.simple_route_similarity(control_pair)[0, 1] == 1.0
        # Artifacts are left as they were
        assert renumbered.mapped_target_smiles != reference.mapped_target_smiles

    def test_unaligned_numbering_would_not_compare(self):
        """Why alignment is required: raw independent numbers disagree."""
        import rxnutils.routes.comparison as comp

        pair = [_one_step_reference().synthesis_route(), _one_step_candidate(self.RENUMBERED).synthesis_route()]
        assert comp.simple_bond_forming_similarity(pair)[0, 1] == 0.0

    def test_different_target_is_target_mismatch(self):
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        other = _one_step_candidate(
            "[CH3:1][C:2](=[O:3])Cl.[OH:4][c:5]1[cH:6][cH:7][cH:8][cH:9][c:10]1[C:11](=[O:12])[OH:13]>>"
            "[CH3:1][C:2](=[O:3])[O:4][c:5]1[cH:6][cH:7][cH:8][cH:9][c:10]1[C:11](=[O:12])[OH:13]",
            precursors=("CC(=O)Cl", "O=C(O)c1ccccc1O"), target=aspirin,
        )
        assert other.mapping_status == MappingStatus.SUCCESS

        alignment = align_to_reference(_one_step_reference(), other)

        assert alignment.status == MappingStatus.TARGET_MISMATCH
        assert alignment.candidate_route is None

    def test_target_mismatch_is_decided_before_mapping(self):
        unmapped = _one_step_candidate(AMIDATION)
        unmapped = unmapped.model_copy(update={"route_tree": {**unmapped.route_tree, "smiles": "CCO"}})
        reference = _mapped_reference(premapped=None)
        assert align_to_reference(reference, unmapped).status == MappingStatus.TARGET_MISMATCH

    def test_stereo_stress_target_aligns_by_m1(self):
        """A stereo-specified reference aligns with an unspecified candidate (M1),
        while M0 keeps them distinct for exact recovery."""
        ester = "C[C@H](OC(C)=O)c1ccccc1"
        flat = "CC(OC(C)=O)c1ccccc1"
        mapped = ("[CH3:1][C:2](=[O:3])Cl.[OH:4][C@@H:5]([CH3:6])[c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1>>"
                  "[CH3:1][C:2](=[O:3])[O:4][C@@H:5]([CH3:6])[c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1")
        reference = _one_step_candidate(mapped, precursors=("CC(=O)Cl", "C[C@@H](O)c1ccccc1"), target=ester)
        candidate = _one_step_candidate(mapped.replace("@@", ""), precursors=("CC(=O)Cl", "CC(O)c1ccccc1"),
                                        target=flat)
        assert reference.mapping_status == candidate.mapping_status == MappingStatus.SUCCESS
        assert m0_key(ester) != m0_key(flat)

        assert align_to_reference(reference, candidate).status == MappingStatus.SUCCESS


# ============================================================ bad mapping


class TestValidation:
    def test_valid_route_passes(self):
        assert validate_route_mapping(_forward_tree()) == []

    def test_duplicate_map_numbers(self):
        errors = validate_route_mapping(_forward_tree(amidation=AMIDATION.replace("[CH3:10][NH:1]", "[CH3:1][NH:1]")))
        assert any("duplicate map numbers in product" in e for e in errors)

    def test_missing_root_mapping(self):
        errors = validate_route_mapping(_forward_tree(amidation=AMIDATION.replace(">>[CH3:10]", ">>[CH3]")))
        assert any("unmapped heavy atoms" in e for e in errors)

    def test_reactant_numbers_absent_from_product(self):
        errors = validate_route_mapping(_forward_tree(amidation=AMIDATION.replace("(Cl)", "([Cl:30])")))
        assert any("absent from product" in e for e in errors)

    def test_inconsistent_intermediate_maps(self):
        errors = validate_route_mapping(_forward_tree(chlorination=CHLORINATION_INCONSISTENT))
        assert any("disagree with its parent" in e for e in errors)

    def test_target_number_reused_off_target(self):
        errors = validate_route_mapping(_forward_tree(chlorination=CHLORINATION_TARGET_REUSE))
        assert any("disagree with its parent" in e for e in errors)

    def test_template_smarts_is_not_a_molecular_mapping(self):
        template = "[C:2](=[O;D1;H0:3])-[N;H0;D3;+0:1]>>Cl-[C:2]=[O;D1;H0:3].[N;D1;H0;+0:1]"
        errors = validate_route_mapping(_forward_tree(amidation=template))
        assert any("does not parse as molecules" in e for e in errors)

    def test_mapping_of_other_molecules(self):
        errors = validate_route_mapping(_forward_tree(chlorination=AMIDATION))
        assert any("different molecule" in e for e in errors)

    def test_partially_mapped_route(self):
        errors = validate_route_mapping(_forward_tree(chlorination=None))
        assert any("no mapped reaction SMILES" in e for e in errors)

    def test_rejected_stored_maps_are_not_repaired(self):
        route = _candidate({AMIDE: AMIDATION, CHLORIDE: CHLORINATION_TARGET_REUSE})
        artifact = map_candidate(route, derive_candidate(route, "h"), RxnutilsRouteMapper(environ={}))

        assert artifact.mapping_status == MappingStatus.MAPPER_UNAVAILABLE
        assert any(w.startswith("stored mapping rejected") for w in artifact.warnings)

    def test_rejected_stored_maps_are_replaced_by_an_available_mapper(self):
        spy = TableMapper(TABLE)
        route = _candidate({AMIDE: AMIDATION, CHLORIDE: CHLORINATION_INCONSISTENT})
        artifact = map_candidate(route, derive_candidate(route, "h"), spy)

        assert spy.calls == 1
        assert artifact.mapping_status == MappingStatus.SUCCESS
        assert artifact.mapping_tool == "table-mapper"


# ============================================================ tree-less candidates


class TestTreelessCandidates:
    def test_saved_tree_and_flat_reconstruction_map_alike(self):
        saved, flat = _candidate(tree=True), _candidate(tree=False)
        mapped_saved = map_candidate(saved, derive_candidate(saved, "h"), TableMapper(TABLE))
        mapped_flat = map_candidate(flat, derive_candidate(flat, "h"), TableMapper(TABLE))
        premapped = _candidate({AMIDE: AMIDATION, CHLORIDE: CHLORINATION})
        mapped_stored = map_candidate(premapped, derive_candidate(premapped, "h"))

        assert {a.mapping_status for a in (mapped_saved, mapped_flat, mapped_stored)} == {MappingStatus.SUCCESS}
        assert mapped_saved.route_chemistry_digest == mapped_flat.route_chemistry_digest \
            == mapped_stored.route_chemistry_digest
        canon = lambda a: sorted(_canon(r) for r in a.mapped_reactions)  # noqa: E731
        assert canon(mapped_saved) == canon(mapped_flat) == canon(mapped_stored)

    @pytest.mark.parametrize("case", ["multiple producers", "unreachable", "leaves differ"])
    def test_ambiguous_flat_routes_stay_refused(self, case):
        reactions = [Reaction(product=AMIDE, precursors=[CHLORIDE, AMINE]),
                     Reaction(product=CHLORIDE, precursors=[ACID, THIONYL])]
        leaves = [ACID, THIONYL, AMINE]
        if case == "multiple producers":
            reactions.append(Reaction(product=CHLORIDE, precursors=[ACID, "O=C(Cl)C(=O)Cl"]))
        elif case == "unreachable":
            reactions.append(Reaction(product="CCO", precursors=["CC=O"]))
        else:
            leaves = [ACID, AMINE]
        route = Route(target=AMIDE, reactions=reactions, solved=True, tree=None,
                      leaves=[Molecule(smiles=s, in_stock=True) for s in leaves])

        artifact = map_candidate(route, derive_candidate(route, "h"), TableMapper(TABLE))

        assert artifact.mapping_status == MappingStatus.INVALID_ROUTE
        assert artifact.route_tree is None
        assert "flat reconstruction" in artifact.failure_reason


# ============================================================ composites


class TestComposites:
    def test_composite_can_be_mapped_but_stays_ineligible(self, tmp_path):
        composite = _reference(composite=True)
        derived = derive_reference(composite, "h")
        artifact = map_reference(composite, derived, premapped=PREMAPPED, premapped_identity=CURATED)
        assert artifact.mapping_status == MappingStatus.SUCCESS
        assert derived.exact_eligibility == ExactEligibility.EXCLUDED

        checkpoint = Checkpoint(tmp_path / "cp", {"schema": 1})
        checkpoint.save(m0_key(AMIDE), [_candidate({AMIDE: AMIDATION, CHLORIDE: CHLORINATION})], False)
        evaluator = SimilarityEvaluator(LiteratureReferenceSet(references=[composite]), "h",
                                        mapped_references={composite.reference_id: artifact})
        target = evaluator.evaluate(Checkpoint.open_readonly(tmp_path / "cp")).per_target[0]

        assert target.eligible_reference_ids == []
        assert target.excluded_reference_ids == [composite.reference_id]
        assert target.reference_matches == []
        pair = evaluator._compare_pair(derived, None, "c", "h", 1, None)
        assert pair.status == SimilarityStatus.REFERENCE_CONVERSION_FAILED


# ============================================================ mapper capability


class TestMapperCapability:
    def test_unset_environment_is_unavailable(self):
        assert probe_rxnutils_mapper({"PATH": ""}).capability == MapperCapability.UNAVAILABLE

    def test_env_path_without_conda_is_incompatible(self, tmp_path):
        probe = probe_rxnutils_mapper({"PATH": "", "RXNMAPPER_ENV_PATH": str(tmp_path)})
        assert probe.capability == MapperCapability.ENVIRONMENT_INCOMPATIBLE

    def test_missing_env_directory_is_incompatible(self, tmp_path):
        conda = tmp_path / "conda"
        conda.write_text("")
        conda.chmod(0o755)
        probe = probe_rxnutils_mapper({"PATH": str(tmp_path), "RXNMAPPER_ENV_PATH": str(tmp_path / "missing")})
        assert probe.capability == MapperCapability.ENVIRONMENT_INCOMPATIBLE

    def test_conda_and_env_are_available(self, tmp_path):
        conda = tmp_path / "conda"
        conda.write_text("")
        conda.chmod(0o755)
        probe = probe_rxnutils_mapper({"PATH": str(tmp_path), "RXNMAPPER_ENV_PATH": str(tmp_path)})
        assert probe.capability == MapperCapability.AVAILABLE

    def test_python_rxnmapper_package_alone_is_not_a_mapper(self, tmp_path, monkeypatch):
        """rxnutils runs rxnmapper through conda, never in-process."""
        (tmp_path / "rxnmapper").mkdir()
        (tmp_path / "rxnmapper" / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(tmp_path))
        assert probe_rxnutils_mapper({"PATH": ""}).capability == MapperCapability.UNAVAILABLE

    def test_automatic_mapping_request_here(self):
        """With whatever this environment offers, an unmapped reference is either
        mapped by the real mapper or reported MAPPER_UNAVAILABLE - never faked."""
        mapper = RxnutilsRouteMapper()
        artifact = _mapped_reference(premapped=None, mapper=mapper)
        if mapper.capability() == MapperCapability.AVAILABLE:
            assert artifact.mapping_status in (MappingStatus.SUCCESS, MappingStatus.MAPPING_FAILED)
        else:
            assert artifact.mapping_status == MappingStatus.MAPPER_UNAVAILABLE


@pytest.mark.skipif(RxnutilsRouteMapper().capability() != MapperCapability.AVAILABLE,
                    reason="needs RXNMAPPER_ENV_PATH and conda for rxnutils route mapping")
def test_real_rxnutils_mapping_is_route_wide():
    artifact = _mapped_reference(premapped=None, mapper=RxnutilsRouteMapper())
    assert artifact.mapping_status == MappingStatus.SUCCESS
    assert validate_route_mapping(artifact.mapped_tree) == []
