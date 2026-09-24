"""Mapped derived artifacts: the atom-mapping boundary of the literature benchmark.

Mapping-dependent evaluators (Phase 3A graded similarity, and later Phase 3B
disconnection diagnostics) consume a mapped derived artifact instead of mapping
routes themselves:

    LiteratureReference -> DerivedReference -> MappedDerivedReference
    Route               -> DerivedCandidate -> MappedDerivedCandidate

An artifact records which chemistry was mapped, by which tool and version, under
which policy, and the exact mapped output, and is content-addressed by
``artifact_digest``. It never records ranks, scores, search state or stock.

Mapping sources, in order:

1. Maps already present on the route (curated forward-form maps for a reference,
   or the maps AiZynthFinder stores on a saved ``Route.tree``). They are used
   only if they pass :func:`validate_route_mapping`.
2. An external route mapper (:class:`RouteMapper`), when one is available.
3. Otherwise the artifact is ``MAPPER_UNAVAILABLE``. No map is ever invented.

Map numbers are kept exactly as the official reader or mapper emits them.
Comparable target numbering is established per comparison by
:func:`align_to_reference`, the one shared implementation of that step.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

import rxnutils.routes.base as base
import rxnutils.routes.readers as readers
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import Chem
from rxnutils.chem.utils import split_rsmi

from reagent.core.chem import m0_key, m1_key
from reagent.core.models import Route
from reagent.eval.checkpoint import _write
from reagent.eval.literature import LiteratureReference, MoleculeRole
from reagent.eval.literature_derived import (
    DerivedCandidate,
    DerivedReference,
    reconstruct_tree_from_steps,
)

ARTIFACT_SCHEMA_VERSION = 1
# Bump when validation rules or mapping-source precedence change.
MAPPING_POLICY_VERSION = 1
# Forward-form ("reactants>>product") reaction SMILES as emitted by the official
# rxnutils reader or mapper. Map numbers are not renumbered; molecule identity in
# validation is M1, because mappers may invert stereo and rxnutils tolerates it.
NORMALIZATION_POLICY = "rxnutils-forward/as-emitted/m1-identity/v1"


class MappingStatus(str, Enum):
    SUCCESS = "success"
    MAPPER_UNAVAILABLE = "mapper_unavailable"
    MAPPING_FAILED = "mapping_failed"
    INVALID_REFERENCE = "invalid_reference"
    INVALID_ROUTE = "invalid_route"
    TARGET_MISMATCH = "target_mismatch"
    # A cached artifact whose inputs changed, with no mapper to regenerate it
    STALE = "stale"


class MapperCapability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    IMPORT_BROKEN = "import_broken"
    ENVIRONMENT_INCOMPATIBLE = "environment_incompatible"


@dataclass(frozen=True)
class MapperIdentity:
    tool: str
    version: str


class MapperUnavailableError(RuntimeError):
    """The mapper backend could not be run at all (as opposed to failing on chemistry)."""


class RouteMapper(Protocol):
    """An external route-wide atom mapper.

    ``map_tree`` takes a forward-form route tree without maps and returns a copy
    whose reaction nodes carry ``metadata["mapped_reaction_smiles"]``, consistent
    from the root compound down. It raises :class:`MapperUnavailableError` when
    the backend cannot run and any other exception when mapping fails.
    """

    identity: MapperIdentity

    def capability(self) -> MapperCapability: ...

    def map_tree(self, tree: dict) -> dict: ...


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@dataclass(frozen=True)
class MapperProbe:
    capability: MapperCapability
    namerxn: str | None
    conda: str | None
    rxnmapper_env: str | None
    detail: str


def probe_rxnutils_mapper(environ: dict[str, str] | None = None) -> MapperProbe:
    """Report whether ``SynthesisRoute.assign_atom_mapping`` can run here.

    rxnutils never imports rxnmapper in-process. It runs
    ``conda run -p $RXNMAPPER_ENV_PATH python mapping.py`` for every mapping call,
    NameRxn or not, so a reachable conda and RXNMAPPER_ENV_PATH are required.
    NameRxn (the ``namerxn`` binary) is optional and only adds classification.
    Nothing is executed here.
    """
    env = os.environ if environ is None else environ
    try:
        from rxnutils.pipeline.actions.reaction_mod import NameRxn, RxnMapper  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure is the finding
        return MapperProbe(MapperCapability.IMPORT_BROKEN, None, None, None, f"rxnutils mapping actions: {exc}")
    namerxn = shutil.which("namerxn", path=env.get("PATH"))
    conda = (
        str(Path(env["CONDA_PATH"]) / "conda") if "CONDA_PATH" in env
        else shutil.which("conda", path=env.get("PATH"))
    )
    rxnmapper_env = env.get("RXNMAPPER_ENV_PATH")
    if not rxnmapper_env:
        return MapperProbe(MapperCapability.UNAVAILABLE, namerxn, conda, None, "RXNMAPPER_ENV_PATH is not set")
    if conda is None or not Path(conda).exists():
        return MapperProbe(MapperCapability.ENVIRONMENT_INCOMPATIBLE, namerxn, conda, rxnmapper_env,
                           "RXNMAPPER_ENV_PATH is set but no conda executable was found")
    if not Path(rxnmapper_env).is_dir():
        return MapperProbe(MapperCapability.ENVIRONMENT_INCOMPATIBLE, namerxn, conda, rxnmapper_env,
                           "RXNMAPPER_ENV_PATH does not name a directory")
    return MapperProbe(MapperCapability.AVAILABLE, namerxn, conda, rxnmapper_env, "rxnmapper environment found")


class RxnutilsRouteMapper:
    """Route-wide mapping through the official ``SynthesisRoute.assign_atom_mapping``."""

    def __init__(self, rxnmapper_version: str | None = None, environ: dict[str, str] | None = None):
        self.probe = probe_rxnutils_mapper(environ)
        backends = ["namerxn"] if self.probe.namerxn else []
        # rxnmapper runs in its own environment, so its version is only known if declared
        backends.append(f"rxnmapper=={rxnmapper_version or 'unrecorded'}")
        self.identity = MapperIdentity(
            tool="rxnutils.SynthesisRoute.assign_atom_mapping",
            version=f"reaction-utils=={_dist_version('reaction-utils')};" + ";".join(backends),
        )

    def capability(self) -> MapperCapability:
        return self.probe.capability

    def map_tree(self, tree: dict) -> dict:
        route = base.SynthesisRoute(copy.deepcopy(tree))
        try:
            route.assign_atom_mapping(overwrite=True, only_rxnmapper=False)
        except FileNotFoundError as exc:
            raise MapperUnavailableError(str(exc)) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"mapper subprocess failed: {exc}") from exc
        return route.reaction_tree


# ---------------------------------------------------------------- validation


def _mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles) if smiles else None


def _maps(mol: Chem.Mol) -> list[int]:
    return [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum()]


def _fragments(mol: Chem.Mol) -> list[Chem.Mol]:
    return list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))


def _frag_m1(mol: Chem.Mol) -> str | None:
    return m1_key(Chem.MolToSmiles(mol))


def _node_m1s(smiles: str) -> list[str | None]:
    return [m1_key(part) for part in smiles.split(".")]


def _labelled(mol: Chem.Mol, keep: set[int]) -> str:
    """Stereo-free canonical SMILES keeping only the map numbers in ``keep``."""
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() not in keep:
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _reaction_nodes(tree: dict):
    """Yield (molecule node, reaction node) pairs, root first."""
    for reaction in tree.get("children") or []:
        yield tree, reaction
        for child in reaction.get("children", []):
            yield from _reaction_nodes(child)


def has_any_mapping(tree: dict) -> bool:
    return any(r.get("metadata", {}).get("mapped_reaction_smiles") for _, r in _reaction_nodes(tree))


def every_reaction_mapped(tree: dict) -> bool:
    return all(r.get("metadata", {}).get("mapped_reaction_smiles") for _, r in _reaction_nodes(tree))


def validate_route_mapping(tree: dict) -> list[str]:
    """Check that a forward-form tree carries a route-wide molecular mapping.

    Returns the list of violations; an empty list means the mapping is usable.
    Nothing is repaired. The rules:

    * every reaction has a mapped reaction SMILES whose product and reactants are,
      up to M1, the molecules of the tree (so template SMARTS or per-step maps of
      other molecules are rejected);
    * no map number repeats within a product or within one reaction's reactants;
    * every mapped reactant atom appears in the product;
    * every heavy atom of the root compound is mapped;
    * an intermediate's mapped product agrees with how its parent reaction maps
      it, and target map numbers never reappear on atoms outside the target.
    """
    errors: list[str] = []
    reactions = list(_reaction_nodes(tree))
    if not reactions:
        return ["route has no reactions"]

    parsed: dict[int, tuple[Chem.Mol, list[Chem.Mol]]] = {}
    for mol_node, reaction in reactions:
        where = f"reaction to {mol_node.get('smiles', '?')}"
        rsmi = reaction.get("metadata", {}).get("mapped_reaction_smiles")
        if not rsmi:
            errors.append(f"{where}: no mapped reaction SMILES")
            continue
        try:
            reactants_smiles, _, product_smiles = split_rsmi(rsmi)
        except ValueError:
            errors.append(f"{where}: not a reaction SMILES: {rsmi}")
            continue
        product, reactants = _mol(product_smiles), _mol(reactants_smiles)
        if product is None or reactants is None:
            errors.append(f"{where}: mapped SMILES does not parse as molecules: {rsmi}")
            continue
        if _frag_m1(product) != m1_key(mol_node.get("smiles", "")):
            errors.append(f"{where}: mapped product is a different molecule: {product_smiles}")
            continue
        reactant_frags = _fragments(reactants)
        child_m1s = [m for child in reaction.get("children", []) for m in _node_m1s(child.get("smiles", ""))]
        if Counter(_frag_m1(f) for f in reactant_frags) != Counter(child_m1s):
            errors.append(f"{where}: mapped reactants are different molecules: {reactants_smiles}")
            continue
        product_maps, reactant_maps = _maps(product), _maps(reactants)
        if len(product_maps) != len(set(product_maps)):
            errors.append(f"{where}: duplicate map numbers in product")
        if len(reactant_maps) != len(set(reactant_maps)):
            errors.append(f"{where}: duplicate map numbers across reactants")
        stray = sorted(set(reactant_maps) - set(product_maps))
        if stray:
            errors.append(f"{where}: reactant map numbers absent from product: {stray}")
        parsed[id(reaction)] = (product, reactant_frags)

    root_reaction = tree["children"][0]
    if id(root_reaction) in parsed:
        root_product = parsed[id(root_reaction)][0]
        unmapped = sum(1 for a in root_product.GetAtoms() if not a.GetAtomMapNum())
        if unmapped:
            errors.append(f"root compound has {unmapped} unmapped heavy atoms")
        target_maps = set(_maps(root_product))
    else:
        target_maps = set()

    # Intermediates: the child reaction's product against the parent's reactant
    for _, reaction in reactions:
        if id(reaction) not in parsed:
            continue
        unused = list(parsed[id(reaction)][1])
        for child in reaction.get("children", []):
            child_reactions = child.get("children") or []
            if not child_reactions or id(child_reactions[0]) not in parsed:
                continue
            child_product = parsed[id(child_reactions[0])][0]
            child_m1 = m1_key(child.get("smiles", ""))
            match = None
            for frag in unused:
                if _frag_m1(frag) != child_m1:
                    continue
                parent_maps = set(_maps(frag))
                intruders = (set(_maps(child_product)) & target_maps) - parent_maps
                if not intruders and _labelled(child_product, parent_maps) == _labelled(frag, parent_maps):
                    match = frag
                    break
            if match is None:
                errors.append(f"intermediate {child.get('smiles', '?')}: map numbers disagree with its parent reaction")
            else:
                unused.remove(match)
    return errors


# ---------------------------------------------------------------- topology


def _strip_tree(tree: dict, keep_maps: bool) -> dict:
    """Minimal rxnutils tree: type, smiles, children and (optionally) the mapping."""
    node = {"type": "mol", "smiles": tree["smiles"]}
    if tree.get("children"):
        reaction = tree["children"][0]
        metadata = {}
        if keep_maps and reaction.get("metadata", {}).get("mapped_reaction_smiles"):
            metadata["mapped_reaction_smiles"] = reaction["metadata"]["mapped_reaction_smiles"]
        node["children"] = [{
            "type": "reaction",
            "metadata": metadata,
            "children": [_strip_tree(c, keep_maps) for c in reaction.get("children", [])],
        }]
    return node


def _canonical_topology(tree: dict) -> str:
    """Order-independent, map-free description of which chemistry a tree holds."""
    def canon(node: dict):
        children = [canon(c) for c in (node.get("children") or [{}])[0].get("children", [])] if node.get("children") else []
        return [m0_key(node["smiles"]) or node["smiles"], sorted(children, key=json.dumps)]
    return json.dumps(canon(tree), sort_keys=True)


def reference_tree(ref: LiteratureReference) -> tuple[dict | None, dict[str, dict], str | None]:
    """Forward-form tree of a curated reference, with its reaction nodes by step_id.

    Returns ``(tree, reactions_by_step_id, failure_reason)``. Inferred steps are not
    route evidence and are left out, matching exact-recovery derivation.
    """
    m0s = {m.molecule_id: m0_key(m.reported_smiles) for m in ref.molecules if m.reported_smiles}
    target = next((m for m in ref.molecules if m.role == MoleculeRole.TARGET), None)
    if target is None or not m0s.get(target.molecule_id):
        return None, {}, "target molecule is missing or unresolved"
    producers: dict[str, list] = {}
    for step in ref.steps:
        if not step.inferred:
            producers.setdefault(step.product_id, []).append(step)
    by_step: dict[str, dict] = {}

    def build(molecule_id: str, path: frozenset[str]) -> dict:
        m0 = m0s.get(molecule_id)
        if not m0:
            raise ValueError(f"molecule {molecule_id} is unresolved")
        node = {"type": "mol", "smiles": m0}
        steps = producers.get(molecule_id, [])
        if not steps:
            return node
        if len(steps) > 1:
            raise ValueError(f"molecule {molecule_id} has several producing steps")
        if molecule_id in path:
            raise ValueError(f"cycle through {molecule_id}")
        step = steps[0]
        reaction = {
            "type": "reaction",
            "metadata": {},
            "children": [build(p, path | {molecule_id}) for p in step.precursor_ids],
        }
        by_step[step.step_id] = reaction
        node["children"] = [reaction]
        return node

    try:
        tree = build(target.molecule_id, frozenset())
    except ValueError as exc:
        return None, {}, str(exc)
    if not tree.get("children"):
        return None, {}, "target has no producing step"
    return tree, by_step, None


def candidate_tree(route: Route | DerivedCandidate) -> tuple[dict | None, bool, str | None]:
    """Candidate topology: the original Route.tree, else an unambiguous flat rebuild.

    Returns ``(forward_tree, carries_stored_maps, failure_reason)``. A saved
    AiZynthFinder tree holds retro-form maps on every reaction, which only the
    official reader converts to forward form. A DerivedCandidate has only flat steps.
    """
    if isinstance(route, DerivedCandidate):
        errors: list[str] = []
        tree = reconstruct_tree_from_steps(
            route.target_m0, [(s.product_m0, s.precursor_m0s) for s in route.steps], route.leaf_m0s, errors,
        )
        if tree is None:
            return None, False, "; ".join(errors) or "flat reconstruction failed"
        return _strip_tree(tree, keep_maps=False), False, None
    if route.tree and isinstance(route.tree, dict):
        if every_reaction_mapped(route.tree):
            try:
                converted = readers.read_aizynthfinder_dict(route.tree).reaction_tree
            except Exception as exc:  # noqa: BLE001 - malformed stored maps
                return _strip_tree(route.tree, keep_maps=False), False, f"stored maps unreadable: {exc}"
            return _strip_tree(converted, keep_maps=True), True, None
        return _strip_tree(route.tree, keep_maps=False), False, None
    errors: list[str] = []
    steps = [(m0_key(r.product) or "", tuple(m0_key(p) or "" for p in r.precursors)) for r in route.reactions]
    tree = reconstruct_tree_from_steps(
        m0_key(route.target) or "", steps, [m0_key(leaf.smiles) or "" for leaf in route.leaves], errors,
    )
    if tree is None:
        return None, False, "; ".join(errors) or "flat reconstruction failed"
    return _strip_tree(tree, keep_maps=False), False, None


# ---------------------------------------------------------------- artifacts


def _source_digest(route: Route | DerivedCandidate) -> str:
    return hashlib.sha256(
        json.dumps(route.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


_DIGEST_EXCLUDED = {"artifact_digest", "created_at", "warnings"}


class MappedDerivedRoute(BaseModel):
    """A route's mapped derived chemistry. Content-addressed; no ranks or scores."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = ARTIFACT_SCHEMA_VERSION
    subject_kind: Literal["reference", "candidate"]
    subject_id: str = Field(min_length=1)
    source_digest: str = Field(min_length=1)
    derived_digest: str = Field(min_length=1)
    route_chemistry_digest: str | None = None

    mapping_status: MappingStatus
    mapping_tool: str = ""
    mapping_tool_version: str = ""
    normalization_policy: str = NORMALIZATION_POLICY
    mapping_policy_version: int = MAPPING_POLICY_VERSION
    rdkit_version: str = Field(min_length=1)
    reaction_utils_version: str = Field(min_length=1)

    # Unmapped forward topology, kept whenever it could be built
    route_tree: dict | None = None
    mapped_target_smiles: str | None = None
    mapped_reactions: list[str] = Field(default_factory=list)
    mapped_tree: dict | None = None

    failure_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str = ""
    artifact_digest: str = ""

    @model_validator(mode="after")
    def check_invariants(self) -> MappedDerivedRoute:
        if self.mapping_status == MappingStatus.SUCCESS:
            if not (self.mapped_tree and self.mapped_reactions and self.mapped_target_smiles):
                raise ValueError("a SUCCESS artifact must carry its mapped route")
            if self.failure_reason is not None or not self.mapping_tool:
                raise ValueError("a SUCCESS artifact names its mapping tool and has no failure reason")
        else:
            if self.mapped_tree is not None or self.mapped_reactions or self.mapped_target_smiles:
                raise ValueError(f"a {self.mapping_status.value} artifact must not carry mapped chemistry")
            if not self.failure_reason:
                raise ValueError(f"a {self.mapping_status.value} artifact must give a failure reason")
        expected = self.compute_digest()
        if not self.artifact_digest:
            self.artifact_digest = expected
        elif self.artifact_digest != expected:
            raise ValueError("artifact_digest does not match the artifact content")
        return self

    def digest_payload(self) -> dict:
        return {k: v for k, v in self.model_dump(mode="json").items() if k not in _DIGEST_EXCLUDED}

    def compute_digest(self) -> str:
        payload = json.dumps(self.digest_payload(), sort_keys=True, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False)

    def synthesis_route(self) -> base.SynthesisRoute | None:
        """A fresh rxnutils route: mapped when SUCCESS, else the bare topology."""
        tree = self.mapped_tree if self.mapping_status == MappingStatus.SUCCESS else self.route_tree
        return base.SynthesisRoute(copy.deepcopy(tree)) if tree else None

    @property
    def root_smiles(self) -> str | None:
        return self.route_tree["smiles"] if self.route_tree else None


class MappedDerivedReference(MappedDerivedRoute):
    subject_kind: Literal["reference"] = "reference"


class MappedDerivedCandidate(MappedDerivedRoute):
    subject_kind: Literal["candidate"] = "candidate"


def _environment() -> dict:
    import rdkit
    return {"rdkit_version": rdkit.__version__, "reaction_utils_version": _dist_version("reaction-utils")}


def _build(cls, *, subject_id, source_digest, derived_digest, route_tree, status, tool="", version="",
           mapped_tree=None, failure_reason=None, warnings_=()):
    fields = dict(
        subject_id=subject_id, source_digest=source_digest, derived_digest=derived_digest,
        route_chemistry_digest=(
            hashlib.sha256(_canonical_topology(route_tree).encode()).hexdigest() if route_tree else None
        ),
        mapping_status=status, mapping_tool=tool, mapping_tool_version=version,
        route_tree=route_tree, failure_reason=failure_reason, warnings=list(warnings_),
        mapping_policy_version=MAPPING_POLICY_VERSION, normalization_policy=NORMALIZATION_POLICY,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **_environment(),
    )
    if status == MappingStatus.SUCCESS:
        route = base.SynthesisRoute(copy.deepcopy(mapped_tree))
        fields.update(
            mapped_tree=mapped_tree,
            mapped_reactions=route.atom_mapped_reaction_smiles(),
            mapped_target_smiles=route.mapped_root_smiles,
        )
    return cls(**fields)


def _map_topology(cls, *, subject, route_tree, premapped_tree, premapped_identity, mapper, notes):
    """Shared precedence: valid stored maps, then an available mapper, else unavailable."""
    warnings_ = list(notes)
    if premapped_tree is not None:
        problems = validate_route_mapping(premapped_tree)
        if not problems:
            return _build(cls, **subject, route_tree=route_tree, status=MappingStatus.SUCCESS,
                          tool=premapped_identity.tool, version=premapped_identity.version,
                          mapped_tree=premapped_tree, warnings_=warnings_)
        warnings_.append("stored mapping rejected: " + "; ".join(problems))
    if mapper is None or mapper.capability() != MapperCapability.AVAILABLE:
        capability = mapper.capability().value if mapper is not None else "no mapper configured"
        return _build(cls, **subject, route_tree=route_tree, status=MappingStatus.MAPPER_UNAVAILABLE,
                      failure_reason=f"route needs automatic mapping; mapper {capability}", warnings_=warnings_)
    try:
        mapped = _strip_tree(mapper.map_tree(copy.deepcopy(route_tree)), keep_maps=True)
    except MapperUnavailableError as exc:
        return _build(cls, **subject, route_tree=route_tree, status=MappingStatus.MAPPER_UNAVAILABLE,
                      failure_reason=f"mapper could not run: {exc}", warnings_=warnings_)
    except Exception as exc:  # noqa: BLE001 - the mapper's failure is the result
        return _build(cls, **subject, route_tree=route_tree, status=MappingStatus.MAPPING_FAILED,
                      tool=mapper.identity.tool, version=mapper.identity.version,
                      failure_reason=f"{type(exc).__name__}: {exc}", warnings_=warnings_)
    problems = validate_route_mapping(mapped)
    if problems:
        return _build(cls, **subject, route_tree=route_tree, status=MappingStatus.MAPPING_FAILED,
                      tool=mapper.identity.tool, version=mapper.identity.version,
                      failure_reason="mapper output is not a route-wide mapping: " + "; ".join(problems),
                      warnings_=warnings_)
    return _build(cls, **subject, route_tree=route_tree, status=MappingStatus.SUCCESS,
                  tool=mapper.identity.tool, version=mapper.identity.version,
                  mapped_tree=mapped, warnings_=warnings_)


def map_reference(
    ref: LiteratureReference,
    derived: DerivedReference,
    mapper: RouteMapper | None = None,
    premapped: dict[str, str] | None = None,
    premapped_identity: MapperIdentity | None = None,
) -> MappedDerivedReference:
    """Map a curated reference's derived chemistry.

    ``premapped`` gives forward-form mapped reaction SMILES by ``step_id`` (for
    example from a dedicated mapping environment); ``premapped_identity`` says
    what produced them. Eligibility is not decided here: a composite or partial
    reference can be mapped for segment use and stays ineligible downstream.
    """
    if derived.reference_id != ref.reference_id or derived.source_content_hash != ref.content_hash():
        raise ValueError("derived reference does not belong to this reference")
    subject = dict(subject_id=ref.reference_id, source_digest=ref.content_hash(),
                   derived_digest=derived.content_hash())
    tree, by_step, reason = reference_tree(ref)
    if tree is None:
        return _build(MappedDerivedReference, **subject, route_tree=None,
                      status=MappingStatus.INVALID_REFERENCE, failure_reason=reason)
    premapped_tree, notes = None, []
    if premapped:
        if premapped_identity is None:
            raise ValueError("premapped reactions need a premapped_identity")
        unknown = sorted(set(premapped) - set(by_step))
        missing = sorted(set(by_step) - set(premapped))
        if unknown or missing:
            notes.append(f"premapped steps do not cover the route (missing {missing}, unknown {unknown})")
        else:
            # A second build gives a separate tree to annotate, leaving the topology bare
            premapped_tree, premapped_by_step, _ = reference_tree(ref)
            for step_id, rsmi in premapped.items():
                premapped_by_step[step_id]["metadata"]["mapped_reaction_smiles"] = rsmi
    return _map_topology(MappedDerivedReference, subject=subject, route_tree=tree,
                         premapped_tree=premapped_tree, premapped_identity=premapped_identity,
                         mapper=mapper, notes=notes)


def map_candidate(
    route: Route | DerivedCandidate,
    derived: DerivedCandidate,
    mapper: RouteMapper | None = None,
    stored_maps_identity: MapperIdentity | None = None,
) -> MappedDerivedCandidate:
    """Map a generated route, preferring the original Route.tree and its stored maps.

    ``stored_maps_identity`` records what produced maps saved on the tree; by
    default they are attributed to AiZynthFinder template application.
    """
    subject = dict(subject_id=derived.route_id, source_digest=_source_digest(route),
                   derived_digest=derived.content_hash())
    tree, has_maps, reason = candidate_tree(route)
    if tree is None:
        return _build(MappedDerivedCandidate, **subject, route_tree=None,
                      status=MappingStatus.INVALID_ROUTE, failure_reason=reason)
    notes = [reason] if reason else []
    identity = stored_maps_identity or MapperIdentity("aizynthfinder-template-application", "unrecorded")
    return _map_topology(MappedDerivedCandidate, subject=subject, route_tree=_strip_tree(tree, keep_maps=False),
                         premapped_tree=tree if has_maps else None, premapped_identity=identity,
                         mapper=mapper, notes=notes)


def map_synthesis_route(
    route: base.SynthesisRoute,
    kind: Literal["reference", "candidate"],
    subject_id: str,
    identity: MapperIdentity,
    mapper: RouteMapper | None = None,
) -> MappedDerivedRoute:
    """Wrap an already-built forward-form rxnutils route supplied by a caller."""
    tree = _strip_tree(route.reaction_tree, keep_maps=True)
    digest = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()
    cls = MappedDerivedReference if kind == "reference" else MappedDerivedCandidate
    subject = dict(subject_id=subject_id, source_digest=digest, derived_digest=digest)
    if not tree.get("children"):
        return _build(cls, **subject, route_tree=tree, status=MappingStatus.INVALID_ROUTE,
                      failure_reason="route has no reactions")
    return _map_topology(cls, subject=subject, route_tree=_strip_tree(tree, keep_maps=False),
                         premapped_tree=tree if has_any_mapping(tree) else None,
                         premapped_identity=identity, mapper=mapper, notes=[])


# ---------------------------------------------------------------- alignment


@dataclass
class Alignment:
    """Two routes on the reference's target numbering, or why they cannot be."""
    status: MappingStatus
    reason: str
    side: Literal["reference", "candidate"] | None = None
    reference_route: base.SynthesisRoute | None = None
    candidate_route: base.SynthesisRoute | None = None


def align_to_reference(reference: MappedDerivedRoute, candidate: MappedDerivedRoute) -> Alignment:
    """Put a candidate's mapped route on the reference target's atom numbering.

    Independently produced map numbers are never compared directly. Targets are
    compatible when their M1 keys agree (stereo-stress references compare by M1,
    as in Phase 3A); different compounds are TARGET_MISMATCH before any mapping
    is consulted. The candidate is re-numbered on fresh copies with the official
    ``SynthesisRoute.remap``, so neither artifact changes.
    """
    ref_root = m1_key(reference.root_smiles or "")
    cand_root = m1_key(candidate.root_smiles or "")
    if ref_root is None or ref_root != cand_root:
        return Alignment(MappingStatus.TARGET_MISMATCH,
                         f"Reference root {ref_root} differs from candidate root {cand_root}")
    for side in (reference, candidate):
        if side.mapping_status != MappingStatus.SUCCESS:
            return Alignment(side.mapping_status, f"{side.subject_kind}: {side.failure_reason}", side.subject_kind)
    ref_route, cand_route = reference.synthesis_route(), candidate.synthesis_route()
    cand_route.remap(ref_route)
    return Alignment(MappingStatus.SUCCESS, "success", reference_route=ref_route, candidate_route=cand_route)


# ---------------------------------------------------------------- persistence


def stale_fields(
    artifact: MappedDerivedRoute,
    source_digest: str,
    derived_digest: str,
    mapper_identity: MapperIdentity | None = None,
) -> list[str]:
    """Inputs on which a cached artifact no longer agrees with the current run.

    ``mapper_identity`` is checked only for artifacts produced by that kind of
    mapper, and only when the caller has one to compare against: an artifact made
    in a dedicated mapping environment is consumable where no mapper exists.
    """
    env = _environment()
    stale = []
    if artifact.source_digest != source_digest:
        stale.append("source_digest")
    if artifact.derived_digest != derived_digest:
        stale.append("derived_digest")
    if artifact.mapping_policy_version != MAPPING_POLICY_VERSION:
        stale.append("mapping_policy_version")
    if artifact.normalization_policy != NORMALIZATION_POLICY:
        stale.append("normalization_policy")
    if artifact.rdkit_version != env["rdkit_version"]:
        stale.append("rdkit_version")
    if artifact.reaction_utils_version != env["reaction_utils_version"]:
        stale.append("reaction_utils_version")
    if mapper_identity is not None and artifact.mapping_tool == mapper_identity.tool \
            and artifact.mapping_tool_version != mapper_identity.version:
        stale.append("mapping_tool_version")
    return stale


class MappedArtifactStore:
    """One JSON file per mapped artifact, written atomically with sorted keys."""

    # Environment-dependent outcomes are recomputed, never cached
    CACHEABLE = {MappingStatus.SUCCESS, MappingStatus.MAPPING_FAILED,
                 MappingStatus.INVALID_REFERENCE, MappingStatus.INVALID_ROUTE}

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path(self, kind: str, subject_id: str) -> Path:
        return self.directory / kind / f"{hashlib.sha256(subject_id.encode()).hexdigest()}.json"

    def save(self, artifact: MappedDerivedRoute) -> Path | None:
        if artifact.mapping_status not in self.CACHEABLE:
            return None
        path = self.path(artifact.subject_kind, artifact.subject_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, json.loads(artifact.to_json()))
        return path

    def load(self, kind: Literal["reference", "candidate"], subject_id: str) -> MappedDerivedRoute | None:
        path = self.path(kind, subject_id)
        if not path.exists():
            return None
        cls = MappedDerivedReference if kind == "reference" else MappedDerivedCandidate
        try:
            artifact = cls.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            warnings.warn(f"Ignoring unreadable mapped artifact {path}: {exc}")
            return None
        return artifact if artifact.subject_id == subject_id else None


def _stale(cached: MappedDerivedRoute, fields: list[str]) -> MappedDerivedRoute:
    cls = type(cached)
    return _build(cls, subject_id=cached.subject_id, source_digest=cached.source_digest,
                  derived_digest=cached.derived_digest, route_tree=cached.route_tree,
                  status=MappingStatus.STALE,
                  failure_reason="cached mapping is stale (" + ", ".join(fields) + ") and no mapper can regenerate it")


def resolve_artifact(fresh_inputs, compute, cached: MappedDerivedRoute | None,
                     mapper: RouteMapper | None, store: MappedArtifactStore | None) -> MappedDerivedRoute:
    """Use a current cached artifact, else compute; never pass a stale one as current."""
    source_digest, derived_digest = fresh_inputs
    available = mapper is not None and mapper.capability() == MapperCapability.AVAILABLE
    if cached is not None:
        fields = stale_fields(cached, source_digest, derived_digest, mapper.identity if available else None)
        if not fields:
            return cached
    result = compute()
    if cached is not None and result.mapping_status == MappingStatus.MAPPER_UNAVAILABLE:
        return _stale(cached, fields)
    if store is not None:
        store.save(result)
    return result


def resolve_reference(ref, derived, mapper=None, store=None, supplied=None, premapped=None, premapped_identity=None):
    cached = supplied if supplied is not None else (store.load("reference", ref.reference_id) if store else None)
    return resolve_artifact(
        (ref.content_hash(), derived.content_hash()),
        lambda: map_reference(ref, derived, mapper, premapped, premapped_identity),
        cached, mapper, store,
    )


def resolve_candidate(route, derived, mapper=None, store=None, stored_maps_identity=None):
    cached = store.load("candidate", derived.route_id) if store else None
    return resolve_artifact(
        (_source_digest(route), derived.content_hash()),
        lambda: map_candidate(route, derived, mapper, stored_maps_identity),
        cached, mapper, store,
    )
