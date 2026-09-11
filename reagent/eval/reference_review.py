"""Freeze checkpoint selections for independent, evidence-backed chemistry review.

No search, model calls, or automatic chemical-validity labels. Structural reference
agreement is deliberately reported separately from a human's validity judgment.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import Chem

from reagent.core.chem import canonical
from reagent.core.models import Reaction
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.harness import _select
from reagent.optimize.aggregate import WEIGHT_PROFILES


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def molecule_key(smiles: str) -> str:
    """Ignore atom-map labels, retain stereochemistry, salts and protonation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid molecule: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def reaction_key(reaction: dict) -> tuple:
    reaction = Reaction.model_validate(reaction)
    if not reaction.precursors:
        raise ValueError("Reference reactions require precursors")
    return molecule_key(reaction.product), tuple(sorted(
        molecule_key(smiles) for smiles in reaction.precursors
    ))


def reference_keys(target: str, reactions: list[dict]) -> Counter:
    """Require a connected, acyclic reference with one disconnection per product."""
    keys = Counter(reaction_key(r) for r in reactions)
    products = {}
    for product, precursors in keys.elements():
        if product in products:
            raise ValueError("Reference has multiple reactions for the same product")
        products[product] = precursors
    root = molecule_key(target)
    if root not in products:
        raise ValueError("Reference does not produce target")
    visited = set()

    def visit(product, active):
        if product in active:
            raise ValueError("Reference contains a reaction cycle")
        if product not in products or product in visited:
            return
        for precursor in products[product]:
            visit(precursor, active | {product})
        visited.add(product)

    visit(root, set())
    if visited != set(products):
        raise ValueError("Reference contains disconnected reactions")
    return keys


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["unreviewed", "supported", "unsupported", "uncertain"] = "unreviewed"
    reviewer: str = ""
    evidence: str = ""

    @model_validator(mode="after")
    def attributed(self):
        if self.verdict != "unreviewed" and not (self.reviewer.strip() and self.evidence.strip()):
            raise ValueError("Reviewed judgments require reviewer and evidence/justification")
        return self


class ReviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Judgment
    steps: list[Judgment]


class Reviews(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    bundle_sha256: str
    review_basis: Literal["unspecified", "self_literature", "expert"] = "unspecified"
    targets: dict[str, ReviewRow]


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    verified_by: str = Field(min_length=1)
    reactions: list[Reaction] = Field(min_length=1)

    @model_validator(mode="after")
    def nonblank(self):
        if not all(value.strip() for value in (self.source, self.locator, self.verified_by)):
            raise ValueError("References require source, locator and verified_by")
        return self


class References(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    bundle_sha256: str
    targets: dict[str, list[Reference]]


def prepare(groups: list[tuple[str, list[tuple[str, str]], Path]], profile: str) -> dict:
    """Require every prespecified target, including failed searches, before export."""
    records = []
    provenance = []
    for group, targets, directory in groups:
        checkpoint = Checkpoint.open_readonly(directory)
        hashes = {}
        for name, smiles in targets:
            smiles = canonical(smiles)
            if smiles is None:
                raise ValueError(f"Invalid target: {name}")
            result = checkpoint.load(smiles)
            if result is None:
                raise ValueError(f"Checkpoint missing {name}: {directory}")
            hashes[smiles] = digest(result.model_dump(mode="json"))
            solved = [route for route in result.routes if route.solved]
            # Historical features must not silently supply a different scoring rubric.
            for route in solved:
                route.features = {}
            selected = _select(solved, WEIGHT_PROFILES[profile])[1] if solved else None
            records.append({
                "id": hashlib.sha256(smiles.encode()).hexdigest()[:16],
                "name": name, "group": group, "target": smiles,
                "time_capped": result.time_capped, "candidates": len(result.routes),
                "route": None if selected is None else {
                    "reactions": [
                        {"product": r.product, "precursors": r.precursors}
                        for r in selected.reactions
                    ],
                    "leaves": [leaf.smiles for leaf in selected.leaves],
                },
            })
        provenance.append({
            "group": group, "directory": str(directory.resolve()),
            "manifest": json.loads((directory / "manifest.json").read_text()),
            "results_sha256": hashes,
        })
    if len({record['id'] for record in records}) != len(records):
        raise ValueError("Duplicate targets in review cohort")
    root = Path(__file__).resolve().parents[1]
    implementation = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("eval", "features", "optimize", "core")
        for p in sorted((root / folder).glob("*.py"))
    }
    return {
        "schema_version": 1, "profile": profile, "weights": WEIGHT_PROFILES[profile],
        "scoring_dependencies": {name: version(name) for name in ("rdkit", "numpy")},
        "selection_implementation_sha256": implementation,
        "sources": provenance, "targets": records,
    }


def worksheets(bundle: dict) -> tuple[dict, dict, dict]:
    header = {"schema_version": 1, "bundle_sha256": digest(bundle)}
    reviews = {**header, "review_basis": "self_literature", "targets": {
        row["id"]: {"route": Judgment().model_dump(), "steps": [
            Judgment().model_dump() for _ in row["route"]["reactions"]
        ]}
        for row in bundle["targets"] if row["route"] is not None
    }}
    references = {**header, "targets": {row["id"]: [] for row in bundle["targets"]}}
    # Deliberately omit search method, profile, scores and feasibility predictions.
    blinded = {**header, "targets": [
        {key: row[key] for key in ("id", "target", "route")}
        for row in bundle["targets"]
    ]}
    return reviews, references, blinded


def _counts(judgments: list[Judgment]) -> dict:
    counts = Counter(j.verdict for j in judgments)
    decided = counts["supported"] + counts["unsupported"]
    return {
        "total": len(judgments),
        **{key: counts[key] for key in ("supported", "unsupported", "uncertain", "unreviewed")},
        "decided_fraction": decided / len(judgments) if judgments else None,
        "supported_fraction_among_decided": counts["supported"] / decided if decided else None,
    }


def report(bundle: dict, reviews: dict, references: dict) -> dict:
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema")
    reviews = Reviews.model_validate(reviews)
    references = References.model_validate(references)
    expected_digest = digest(bundle)
    if reviews.bundle_sha256 != expected_digest or references.bundle_sha256 != expected_digest:
        raise ValueError("Worksheets do not belong to this frozen bundle")
    rows = {row["id"]: row for row in bundle["targets"]}
    solved = {key for key, row in rows.items() if row["route"] is not None}
    if set(reviews.targets) != solved or set(references.targets) != set(rows):
        raise ValueError("Worksheet target IDs differ from the frozen cohort")
    route_judgments, step_judgments, comparisons = [], [], []
    for key, row in rows.items():
        route = row["route"]
        if route is not None:
            review = reviews.targets[key]
            if len(review.steps) != len(route["reactions"]):
                raise ValueError(f"Step count differs for {key}")
            route_judgments.append(review.route)
            step_judgments.extend(review.steps)
        refs = references.targets[key]
        if not refs:
            continue
        reference_sets = []
        for ref in refs:
            reactions = [r.model_dump() for r in ref.reactions]
            reference_sets.append(reference_keys(row["target"], reactions))
        prediction = Counter(reaction_key(r) for r in route["reactions"]) if route else Counter()
        comparisons.append({
            "id": key,
            "selected_route_exact_reaction_match": bool(route) and prediction in reference_sets,
            "matched_steps": max(sum((prediction & ref).values()) for ref in reference_sets),
            "selected_steps": sum(prediction.values()),
        })
    return {
        "schema_version": 1, "bundle_sha256": expected_digest,
        "targets": len(rows), "solved_targets": len(solved),
        "solve_rate": len(solved) / len(rows) if rows else None,
        "review_basis": reviews.review_basis,
        "route_judgments": _counts(route_judgments), "step_judgments": _counts(step_judgments),
        "reference_covered_targets": len(comparisons),
        "selected_route_exact_reaction_match_rate": (
            sum(row["selected_route_exact_reaction_match"] for row in comparisons) / len(comparisons)
            if comparisons else None
        ),
        "reference_comparisons": comparisons,
        "limitations": [
            "Reference agreement is exact reaction-multiset agreement, not experimental validity.",
            "Alternative correct syntheses can disagree with a reference.",
            "Judgments are attributed reviews, not laboratory validation or proof of expertise.",
            "Training overlap is unverified; this is not a held-out accuracy benchmark.",
            "The fixed convenience cohort does not establish population-wide accuracy.",
        ],
    }


@click.group()
def review() -> None:
    """Prepare independent chemistry review from saved routes and summarize evidence."""


@review.command("prepare")
@click.option("--moderate-checkpoint", required=True, type=click.Path(exists=True, file_okay=False,
                                                                  path_type=Path))
@click.option("--hard-checkpoint", required=True, type=click.Path(exists=True, file_okay=False,
                                                              path_type=Path))
@click.option("--profile", type=click.Choice(list(WEIGHT_PROFILES)), default="build-it-yourself",
              show_default=True)
@click.option("--output", required=True, type=click.Path(path_type=Path))
def prepare_command(moderate_checkpoint: Path, hard_checkpoint: Path, profile: str,
                    output: Path) -> None:
    """Freeze all 49 targets; refuse to overwrite an existing review directory."""
    from reagent.eval.targets import HARD_TARGETS, TARGETS

    try:
        if output.exists():
            raise ValueError(f"Output already exists: {output}")
        bundle = prepare([
            ("moderate", TARGETS, moderate_checkpoint), ("hard", HARD_TARGETS, hard_checkpoint),
        ], profile)
        reviews, references, blinded = worksheets(bundle)
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (("bundle", bundle), ("reviews", reviews),
                            ("references", references), ("blinded", blinded)):
            (output / f"{name}.json").write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Prepared {len(bundle['targets'])} targets in {output}")


@review.command("report")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report_command(directory: Path) -> None:
    """Print JSON metrics; unreviewed accuracy is null, never zero or a pass."""
    try:
        values = [json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
                  for name in ("bundle", "reviews", "references")]
        result = report(*values)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2, allow_nan=False))


@review.command("page")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def page_command(directory: Path) -> None:
    """Create a local page with molecular drawings and self-review forms."""
    from reagent.eval.review_page import render_page

    try:
        path = render_page(directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Open {path} in your browser. Export reviews.json to save your judgments.")
