"""Deterministic derived representations for exact literature recovery benchmark.

This module provides the transformation from curated LiteratureReference to
chemically-precise DerivedReference, and from generated Route to DerivedCandidate.
These derived objects are reproducible, disposable, and form the basis for
graph-exact comparison.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from functools import cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reagent.core.chem import m0_key, m1_key
from reagent.core.models import Route
from reagent.eval.literature import (
    LiteratureReference,
    MoleculeRole,
    ReferenceMolecule,
    ReferenceStep,
    RouteCompleteness,
    RouteScope,
    StereoRelation,
    StructureResolutionStatus,
)


class DerivationStatus(str, Enum):
    SUCCESS = "success"
    UNRESOLVED_MOLECULE = "unresolved_molecule"
    AMBIGUOUS_MOLECULE = "ambiguous_molecule"
    MISSING_SMILES = "missing_smiles"
    PARSE_FAILED = "parse_failed"
    MULTIPLE_PRODUCER_AMBIGUITY = "multiple_producer_ambiguity"
    INFERRED_STEP_PRESENT = "inferred_step_present"
    INCOMPLETE_ROUTE = "incomplete_route"
    GRAPH_DERIVATION_FAILED = "graph_derivation_failed"


class ExactEligibility(str, Enum):
    CORE_EXACT_ELIGIBLE = "core_exact_eligible"
    STEREO_STRESS = "stereo_stress"
    EXCLUDED = "excluded"


class DerivedMolecule(BaseModel):
    """A molecule with its exact M0/M1 keys for comparison."""
    model_config = ConfigDict(extra="forbid")

    molecule_id: str = Field(min_length=1)
    m0: str = Field(min_length=1)
    m1: str = Field(min_length=1)


class DerivedStep(BaseModel):
    """A step with exact reaction identity (product M0 + multiset of precursor M0s)."""
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    product_m0: str = Field(min_length=1)
    precursor_m0s: tuple[str, ...] = Field(default_factory=tuple)
    inferred: bool = False

    @model_validator(mode="after")
    def sort_precursors(self) -> DerivedStep:
        """Canonicalize precursor order - multiset semantics."""
        self.precursor_m0s = tuple(sorted(self.precursor_m0s))
        return self

    def exact_key(self) -> tuple[str, tuple[str, ...]]:
        """Exact reaction key: (product_M0, sorted(precursor_M0s))."""
        return (self.product_m0, self.precursor_m0s)


class DerivedReference(BaseModel):
    """Deterministic derived representation of a LiteratureReference for exact comparison."""
    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(min_length=1)
    source_content_hash: str = Field(min_length=1)
    derivation_schema_version: Literal[1] = 1
    rdkit_version: str = Field(min_length=1)
    generator_code_hash: str = Field(min_length=1)
    target_m0: str = Field(min_length=1)
    molecules: list[DerivedMolecule] = Field(default_factory=list)
    steps: list[DerivedStep] = Field(default_factory=list)
    starting_material_m0s: tuple[str, ...] = Field(default_factory=tuple)
    intermediate_m0s: tuple[str, ...] = Field(default_factory=tuple)
    graph_signature: str | None = None
    exact_eligibility: ExactEligibility = ExactEligibility.EXCLUDED
    exclusion_reasons: list[str] = Field(default_factory=list)

    def to_deterministic_dict(self) -> dict:
        """Serialize for stable hashing - dict keys sorted, lists preserved where semantic."""
        data = self.model_dump(mode="json", by_alias=True)
        def canonicalize(obj):
            if isinstance(obj, dict):
                return {k: canonicalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                if all(isinstance(x, dict) and "molecule_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("molecule_id", ""))]
                if all(isinstance(x, dict) and "step_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("step_id", ""))]
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


def _derive_molecule(mol: ReferenceMolecule) -> DerivedMolecule | None:
    """Derive M0/M1 from a reference molecule. Returns None if structure unresolved."""
    if mol.structure_resolution != StructureResolutionStatus.RESOLVED:
        return None
    if not mol.reported_smiles:
        return None
    m0 = m0_key(mol.reported_smiles)
    m1 = m1_key(mol.reported_smiles)
    if m0 is None or m1 is None:
        return None
    return DerivedMolecule(molecule_id=mol.molecule_id, m0=m0, m1=m1)


def _derive_step(step: ReferenceStep, mol_lookup: dict[str, DerivedMolecule]) -> DerivedStep | None:
    """Derive exact reaction identity from a reference step."""
    product_mol = mol_lookup.get(step.product_id)
    if not product_mol:
        return None
    precursor_mols = [mol_lookup.get(pid) for pid in step.precursor_ids]
    if any(pm is None for pm in precursor_mols):
        return None
    precursor_m0s = tuple(pm.m0 for pm in precursor_mols)
    return DerivedStep(
        step_id=step.step_id,
        product_m0=product_mol.m0,
        precursor_m0s=precursor_m0s,
        inferred=step.inferred,
    )


def _build_ref_graph_signature(
    target_molecule_id: str,
    molecule_id_to_m0: dict[str, str],
    steps: list[DerivedStep],
    step_precursor_ids: list[tuple[str, ...]],
    step_product_id: list[str],
    starting_material_ids: set[str],
) -> str | None:
    """
    Build deterministic topology-sensitive graph signature using route-local molecule_ids.

    This uses molecule_ids (node occurrences) as graph keys, with M0 as node labels.
    Multiple occurrences of the same M0 at different graph positions are preserved.

    signature(node_id) = ("leaf", M0) if leaf
    signature(node_id) = ("reaction", product_M0, multiset(signature(precursor_node_id))) if produced

    Returns None if genuine multiple-producer ambiguity detected (same node_id produced by multiple steps).
    """

    # Build producer map: product molecule_id -> list of step indices producing it
    producers: dict[str, list[int]] = {}
    for i, prod_id in enumerate(step_product_id):
        producers.setdefault(prod_id, []).append(i)

    # Check for genuine multiple-producer ambiguity: same NODE produced by multiple steps
    for prod_id, step_indices in producers.items():
        if len(step_indices) > 1:
            return None

    @cache
    def sig(node_id: str) -> tuple:
        if node_id in starting_material_ids:
            return ("leaf", molecule_id_to_m0.get(node_id, ""))
        prod_steps = producers.get(node_id)
        if not prod_steps:
            return None
        step_idx = prod_steps[0]
        precursor_ids = step_precursor_ids[step_idx]
        # Check if all precursors have M0 entries
        for p in precursor_ids:
            if p not in molecule_id_to_m0:
                return None
        precursor_sigs = tuple(sorted(sig(p) for p in precursor_ids))
        if any(s is None for s in precursor_sigs):
            return None
        return ("reaction", molecule_id_to_m0.get(node_id, ""), precursor_sigs)

    result = sig(target_molecule_id)
    if result is None:
        return None
    return json.dumps(result, sort_keys=True)


def _build_cand_graph_signature(
    route: Route,
    derivation_errors: list[str],
) -> tuple[str | None, str]:
    """
    Build graph signature from Route using its tree (if available) or reactions.

    Returns: (graph_signature, target_m0)
    """
    # Try to use the tree for faithful topology
    if route.tree and isinstance(route.tree, dict):
        return _build_signature_from_tree(route.tree, derivation_errors)

    # Fallback: reconstruct from flattened reactions with synthetic node IDs
    return _build_signature_from_reactions(route, derivation_errors)


def _build_signature_from_tree(
    tree: dict,
    derivation_errors: list[str],
) -> tuple[str | None, str]:
    """
    Build graph signature from AiZynthFinder tree dict.

    The tree has alternating mol -> reaction -> mol nodes.
    We assign synthetic node IDs during traversal.
    """
    m0_lookup = {}
    step_precursor_ids = []
    step_product_ids = []
    starting_material_ids = set()
    node_counter = [0]
    target_m0 = ""

    def get_node_id() -> str:
        node_counter[0] += 1
        return f"n{node_counter[0]}"

    def traverse(node: dict, parent_reaction_id: str | None = None) -> str:
        nonlocal target_m0
        node_id = get_node_id()

        smiles = node.get("smiles", "")
        m0 = m0_key(smiles) if smiles else ""
        if m0 is None:
            m0 = ""
        m0_lookup[node_id] = m0

        if parent_reaction_id is None:
            target_m0 = m0

        children = node.get("children", [])
        if not children:
            starting_material_ids.add(node_id)
            return node_id

        if len(children) != 1 or children[0].get("type") != "reaction":
            return node_id

        rxn_node = children[0]

        precursor_ids = []
        for reactant_node in rxn_node.get("children", []):
            precursor_id = traverse(reactant_node)
            precursor_ids.append(precursor_id)

        step_precursor_ids.append(tuple(precursor_ids))
        step_product_ids.append(node_id)

        return node_id

    m0_lookup = {}
    step_precursor_ids = []
    step_product_ids = []
    starting_material_ids = set()
    target_m0 = ""

    root_id = traverse(tree)

    # Build producers map
    producers: dict[str, list[int]] = {}
    for i, prod_id in enumerate(step_product_ids):
        producers.setdefault(prod_id, []).append(i)

    # Check for genuine multiple-producer ambiguity
    for prod_id, step_indices in producers.items():
        if len(step_indices) > 1:
            return None, ""


    @cache
    def sig(node_id: str) -> tuple:
        if node_id in starting_material_ids:
            return ("leaf", m0_lookup.get(node_id, ""))
        prod_steps = producers.get(node_id)
        if not prod_steps:
            return None
        step_idx = prod_steps[0]
        precursor_ids = step_precursor_ids[step_idx]
        precursor_sigs = tuple(sorted(sig(p) for p in precursor_ids))
        if any(s is None for s in precursor_sigs):
            return None
        return ("reaction", m0_lookup.get(node_id, ""), precursor_sigs)

    # Find root: product not used as precursor anywhere
    all_precursors = set()
    for precs in step_precursor_ids:
        all_precursors.update(precs)
    root_candidates = set(step_product_ids) - all_precursors
    if not root_candidates:
        return None, ""
    root_id = root_candidates.pop()

    result = sig(root_id)
    if result is None:
        return None, ""
    return json.dumps(result, sort_keys=True), target_m0


def reconstruct_tree_from_steps(
    target_m0: str,
    steps: list[tuple[str, tuple[str, ...]]],
    leaf_m0s: list[str] | tuple[str, ...],
    derivation_errors: list[str],
) -> dict | None:
    """
    Rebuild an AiZynthFinder-style tree from flat (product M0, precursor M0s) steps.

    A flat list does not record which occurrence of a molecule a step expands, so
    each molecule is linked to the step producing its M0. When that link is not
    unique, when a step is unreachable from the target, or when the rebuilt leaves
    differ from the recorded leaves, the topology cannot be recovered faithfully
    and None is returned instead of a guess.
    """
    producers: dict[str, tuple[str, ...]] = {}
    for product, precursors in steps:
        precursors = tuple(sorted(precursors))
        if not product or not all(precursors):
            derivation_errors.append("flat reconstruction: unparseable molecule")
            return None
        if producers.setdefault(product, precursors) != precursors:
            derivation_errors.append(f"flat reconstruction: multiple producers for {product}")
            return None
    if target_m0 not in producers:
        derivation_errors.append("flat reconstruction: target is not produced by any step")
        return None

    used: set[str] = set()
    leaves: list[str] = []

    def build(m0: str, path: frozenset[str]) -> dict | None:
        if m0 not in producers:
            leaves.append(m0)
            return {"type": "mol", "smiles": m0}
        if m0 in path:
            derivation_errors.append(f"flat reconstruction: cycle through {m0}")
            return None
        used.add(m0)
        children = [build(p, path | {m0}) for p in producers[m0]]
        if any(c is None for c in children):
            return None
        return {
            "type": "mol",
            "smiles": m0,
            "children": [{
                "type": "reaction",
                "smiles": ".".join(producers[m0]) + ">>" + m0,
                "metadata": {},
                "children": children,
            }],
        }

    tree = build(target_m0, frozenset())
    if tree is None:
        return None
    if used != set(producers):
        derivation_errors.append("flat reconstruction: steps unreachable from the target")
        return None
    if sorted(leaves) != sorted(leaf_m0s):
        derivation_errors.append("flat reconstruction: rebuilt leaves differ from recorded leaves")
        return None
    return tree


def _build_signature_from_reactions(
    route: Route,
    derivation_errors: list[str],
) -> tuple[str | None, str]:
    """
    Fallback: build graph signature from flattened reactions.

    The tree is rebuilt by M0 linkage and then signed exactly like an original
    Route.tree, so both paths agree whenever the reconstruction is unambiguous.
    """
    steps = [
        (m0_key(rxn.product) or "", tuple(m0_key(p) or "" for p in rxn.precursors))
        for rxn in route.reactions
    ]
    leaf_m0s = [m0_key(leaf.smiles) or "" for leaf in route.leaves]
    tree = reconstruct_tree_from_steps(m0_key(route.target) or "", steps, leaf_m0s, derivation_errors)
    if tree is None:
        return None, ""
    return _build_signature_from_tree(tree, derivation_errors)


def derive_reference(
    ref: LiteratureReference,
    generator_code_hash: str,
) -> DerivedReference:
    """Deterministically derive exact-comparison representation from curated reference."""
    import rdkit

    # Step 1: Derive molecules
    derived_molecules: dict[str, DerivedMolecule] = {}
    exclusion_reasons: list[str] = []

    for mol in ref.molecules:
        if mol.structure_resolution != StructureResolutionStatus.RESOLVED:
            exclusion_reasons.append(f"molecule {mol.molecule_id}: structure_resolution={mol.structure_resolution.value}")
            continue
        derived = _derive_molecule(mol)
        if derived is None:
            exclusion_reasons.append(f"molecule {mol.molecule_id}: missing or unparseable SMILES")
            continue
        derived_molecules[mol.molecule_id] = derived

    # Step 2: Derive steps
    derived_steps: list[DerivedStep] = []
    step_precursor_ids = []
    step_product_ids = []

    for step in ref.steps:
        if step.inferred:
            exclusion_reasons.append(f"step {step.step_id}: inferred=true")
            continue
        derived = _derive_step(step, derived_molecules)
        if derived is None:
            exclusion_reasons.append(f"step {step.step_id}: missing molecule in lookup")
            continue
        derived_steps.append(derived)
        step_precursor_ids.append(tuple(step.precursor_ids))
        step_product_ids.append(step.product_id)

    # Step 3: Identify target molecule
    target_mol = None
    for mol in ref.molecules:
        if mol.role == MoleculeRole.TARGET:
            target_mol = mol
            break

    if target_mol is None:
        exclusion_reasons.append("no target molecule declared")
        target_m0 = ""
    elif target_mol.molecule_id not in derived_molecules:
        exclusion_reasons.append("target molecule structure unresolved")
        target_m0 = ""
    else:
        target_m0 = derived_molecules[target_mol.molecule_id].m0

    # Step 4: Starting materials and intermediates (preserve multiplicity)
    starting_material_m0s = []
    intermediate_m0s = []
    starting_material_ids = set()

    for mol in ref.molecules:
        dm = derived_molecules.get(mol.molecule_id)
        if dm is None:
            continue
        if mol.role == MoleculeRole.STARTING_MATERIAL:
            starting_material_m0s.append(dm.m0)
            starting_material_ids.add(mol.molecule_id)
        elif mol.role == MoleculeRole.INTERMEDIATE:
            intermediate_m0s.append(dm.m0)

    # Step 5: Build graph signature using route-local molecule_ids
    molecule_id_to_m0 = {mid: dm.m0 for mid, dm in derived_molecules.items()}
    step_precursor_ids = [tuple(step.precursor_ids) for step in ref.steps if not step.inferred]
    step_product_ids = [step.product_id for step in ref.steps if not step.inferred]
    starting_material_ids = {mol.molecule_id for mol in ref.molecules
                             if mol.role == MoleculeRole.STARTING_MATERIAL and mol.molecule_id in derived_molecules}
    target_molecule_id = target_mol.molecule_id if target_mol else ""

    graph_signature = _build_ref_graph_signature(
        target_molecule_id,
        molecule_id_to_m0,
        [s for s in derived_steps],  # only non-inferred steps
        [tuple(step.precursor_ids) for step in ref.steps if not step.inferred],
        [step.product_id for step in ref.steps if not step.inferred],
        starting_material_ids,
    )
    if graph_signature is None:
        exclusion_reasons.append("graph_signature: multiple producers or graph gap")

    # Step 6: Determine exact eligibility
    exact_eligibility = ExactEligibility.EXCLUDED
    if not exclusion_reasons:
        if ref.route_scope == RouteScope.FULL_ROUTE:
            if ref.route_completeness == RouteCompleteness.COMPLETE:
                if ref.graph_complete():
                    stereo_rel = ref.stereo_metadata.relation_to_reagent
                    if stereo_rel in (StereoRelation.EXACTLY_COMPATIBLE, StereoRelation.RACEMATE_COMPATIBLE):
                        exact_eligibility = ExactEligibility.CORE_EXACT_ELIGIBLE
                    elif stereo_rel == StereoRelation.REFERENCE_MORE_SPECIFIC:
                        exact_eligibility = ExactEligibility.STEREO_STRESS
                    elif stereo_rel in (StereoRelation.CONFLICTING, StereoRelation.UNRESOLVED):
                        exact_eligibility = ExactEligibility.EXCLUDED
                        exclusion_reasons.append(f"stereo_relation={stereo_rel.value}")
                    else:
                        exact_eligibility = ExactEligibility.EXCLUDED
                        exclusion_reasons.append(f"stereo_relation={stereo_rel.value}")
                else:
                    exclusion_reasons.append("graph_complete=false")
            else:
                exclusion_reasons.append(f"route_completeness={ref.route_completeness.value}")
        else:
            exclusion_reasons.append(f"route_scope={ref.route_scope.value}")

    return DerivedReference(
        reference_id=ref.reference_id,
        source_content_hash=ref.content_hash(),
        rdkit_version=rdkit.__version__,
        generator_code_hash=generator_code_hash,
        target_m0=target_m0,
        molecules=sorted(derived_molecules.values(), key=lambda m: m.molecule_id),
        steps=sorted(derived_steps, key=lambda s: s.step_id),
        starting_material_m0s=tuple(sorted(starting_material_m0s)),
        intermediate_m0s=tuple(sorted(intermediate_m0s)),
        graph_signature=graph_signature,
        exact_eligibility=exact_eligibility,
        exclusion_reasons=exclusion_reasons,
    )


class DerivedCandidate(BaseModel):
    """Deterministic derived representation of a generated Route for exact comparison."""
    model_config = ConfigDict(extra="forbid")

    route_id: str = Field(min_length=1)
    target_m0: str = Field(default="")
    steps: list[DerivedStep] = Field(default_factory=list)
    leaf_m0s: tuple[str, ...] = Field(default_factory=tuple)
    graph_signature: str | None = None
    derivation_errors: list[str] = Field(default_factory=list)
    is_solved: bool = False

    def to_deterministic_dict(self) -> dict:
        data = self.model_dump(mode="json", by_alias=True)
        def canonicalize(obj):
            if isinstance(obj, dict):
                return {k: canonicalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                if all(isinstance(x, dict) and "step_id" in x for x in obj):
                    return [canonicalize(x) for x in sorted(obj, key=lambda d: d.get("step_id", ""))]
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


def derive_candidate(
    route: Route,
    generator_code_hash: str,
) -> DerivedCandidate:
    """Deterministically derive exact-comparison representation from a generated Route."""
    derivation_errors: list[str] = []

    # Build graph signature using tree or reactions
    graph_signature, target_m0 = _build_cand_graph_signature(route, derivation_errors)

    # Derive steps for comparison (still needed for step-by-step comparison)
    derived_steps: list[DerivedStep] = []
    for i, rxn in enumerate(route.reactions):
        product_m0 = m0_key(rxn.product)
        if product_m0 is None:
            product_m0 = ""
            derivation_errors.append(f"step {i} product unparseable: {rxn.product}")

        precursor_m0s = []
        for prec in rxn.precursors:
            pm0 = m0_key(prec)
            if pm0 is None:
                pm0 = ""
                derivation_errors.append(f"step {i} precursor unparseable: {prec}")
            precursor_m0s.append(pm0)

        derived_steps.append(DerivedStep(
            step_id=f"step_{i}",
            product_m0=product_m0,
            precursor_m0s=tuple(sorted(precursor_m0s)),
            inferred=False,
        ))

    # Derive leaf M0s - preserve multiplicity
    leaf_m0s = []
    for leaf in route.leaves:
        lm0 = m0_key(leaf.smiles)
        if lm0 is None:
            lm0 = ""
            derivation_errors.append(f"leaf unparseable: {leaf.smiles}")
        leaf_m0s.append(lm0)

    # Check target SMILES parseability
    if m0_key(route.target) is None:
        derivation_errors.append(f"target SMILES unparseable: {route.target}")

    # Deterministic route ID based on content hash (includes topology via graph_signature)
    route_content = f"{route.target}|" + "|".join(
        f"{rxn.product}>>{'.'.join(rxn.precursors)}" for rxn in route.reactions
    )
    # Include graph_signature in the hash to distinguish candidates with same reactions but different topology
    graph_sig = graph_signature or ""
    route_id = hashlib.sha256(f"{route_content}|{graph_sig}".encode()).hexdigest()[:16]

    return DerivedCandidate(
        route_id=route_id,
        target_m0=m0_key(route.target) or "",
        steps=sorted(derived_steps, key=lambda s: s.step_id),
        leaf_m0s=tuple(sorted(leaf_m0s)),
        graph_signature=_build_cand_graph_signature(route, derivation_errors)[0],
        derivation_errors=derivation_errors,
        is_solved=route.solved,
    )