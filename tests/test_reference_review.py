"""Evidence accounting must not turn missing reviews into chemical accuracy."""

import copy
import json

import pytest
from click.testing import CliRunner

from reagent.cli import main
from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.reference_review import (
    molecule_key,
    prepare,
    reaction_key,
    reference_keys,
    report,
    worksheets,
)


@pytest.fixture
def cohort(tmp_path):
    checkpoint = Checkpoint(tmp_path / "checkpoint", {"schema": 1})
    checkpoint.save("CCO", [Route(
        target="CCO", solved=True,
        reactions=[Reaction(product="CCO", precursors=["CC=O"],
                            metadata={"policy_probability": 0.8})],
        leaves=[Molecule(smiles="CC=O", in_stock=True)],
    )], False)
    checkpoint.save("CCC", [], True)
    return [("test", [("ethanol", "CCO"), ("propane", "CCC")], checkpoint.directory)]


def test_freeze_is_reproducible_and_never_writes_checkpoints(cohort):
    directory = cohort[0][2]
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    bundle = prepare(cohort, "build-it-yourself")
    assert bundle == prepare(cohort, "build-it-yourself")
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert bundle["targets"][1]["route"] is None
    reviews, references, blinded = worksheets(bundle)
    assert "policy_probability" not in json.dumps(blinded)
    assert "weights" not in blinded
    result = report(bundle, reviews, references)
    assert result["targets"] == 2
    assert result["solve_rate"] == 0.5
    assert result["expert_routes"]["unreviewed"] == 1
    assert result["expert_routes"]["supported_fraction_among_decided"] is None
    assert result["selected_route_exact_reaction_match_rate"] is None


def test_missing_targets_fail_instead_of_shrinking_denominator(cohort):
    cohort[0][1].append(("missing", "C"))
    with pytest.raises(ValueError, match="missing"):
        prepare(cohort, "build-it-yourself")


def test_cohort_uses_the_same_canonical_keys_as_evaluate(cohort):
    original = prepare(cohort, "build-it-yourself")
    cohort[0][1][0] = ("ethanol", "OCC")
    assert prepare(cohort, "build-it-yourself") == original


def test_stereo_is_retained_maps_and_precursor_order_are_ignored():
    assert molecule_key("[CH3:1][CH2:2][OH:3]") == molecule_key("OCC")
    assert molecule_key("N[C@@H](C)C(=O)O") != molecule_key("N[C@H](C)C(=O)O")
    assert reaction_key({"product": "CCO", "precursors": ["C", "CO"]}) == reaction_key(
        {"product": "OCC", "precursors": ["OC", "C"]})


def test_reference_match_is_separate_from_expert_validity(cohort):
    bundle = prepare(cohort, "build-it-yourself")
    reviews, references, _ = worksheets(bundle)
    key = bundle["targets"][0]["id"]
    references["targets"][key] = [{
        "source": "Synthetic test fixture, not published chemistry", "locator": "test step 1",
        "verified_by": "fixture author",
        "reactions": [{"product": "OCC", "precursors": ["O=CC"]}],
    }]
    result = report(bundle, reviews, references)
    assert result["reference_covered_targets"] == 1
    assert result["selected_route_exact_reaction_match_rate"] == 1
    assert result["expert_routes"]["supported_fraction_among_decided"] is None
    # A failed search with a reference still belongs in the reference denominator.
    references["targets"][bundle["targets"][1]["id"]] = [{
        **references["targets"][key][0],
        "reactions": [{"product": "CCC", "precursors": ["CC=C"]}],
    }]
    assert report(bundle, reviews, references)["selected_route_exact_reaction_match_rate"] == 0.5


def test_judgments_require_attribution_and_uncertainty_is_not_success(cohort):
    bundle = prepare(cohort, "build-it-yourself")
    reviews, references, _ = worksheets(bundle)
    key = bundle["targets"][0]["id"]
    row = reviews["targets"][key]
    row["route"] = {"verdict": "supported"}
    with pytest.raises(ValueError, match="reviewer"):
        report(bundle, reviews, references)
    row["route"] = {"verdict": "uncertain", "reviewer": "chemist A",
                    "evidence": "Conditions are missing"}
    row["steps"] = [{"verdict": "unsupported", "reviewer": "chemist A",
                     "evidence": "Fixture rejection"}]
    result = report(bundle, reviews, references)
    assert result["expert_routes"]["uncertain"] == 1
    assert result["expert_routes"]["supported_fraction_among_decided"] is None
    assert result["expert_steps"]["supported_fraction_among_decided"] == 0


@pytest.mark.parametrize("change", ["digest", "id", "steps", "verdict"])
def test_stale_or_malformed_reviews_are_rejected(cohort, change):
    bundle = prepare(cohort, "build-it-yourself")
    reviews, references, _ = worksheets(bundle)
    key = bundle["targets"][0]["id"]
    if change == "digest":
        bundle["profile"] = "changed"
    elif change == "id":
        reviews["targets"].pop(key)
    elif change == "steps":
        reviews["targets"][key]["steps"] = []
    else:
        reviews["targets"][key]["route"]["verdict"] = "valid"
    with pytest.raises(ValueError):
        report(bundle, reviews, references)


@pytest.mark.parametrize("reactions", [
    [{"product": "CCC", "precursors": ["C"]}],
    [{"product": "CCO", "precursors": ["CCO"]}],
    [{"product": "CCO", "precursors": ["C"]}, {"product": "CCC", "precursors": ["C"]}],
    [{"product": "CCO", "precursors": []}],
    [{"product": "CCO", "precursors": ["invalid"]}],
])
def test_invalid_or_disconnected_references_are_rejected(reactions):
    with pytest.raises(ValueError):
        reference_keys("CCO", reactions)


def test_cli_creates_report_and_refuses_overwrite(cohort, tmp_path, monkeypatch):
    from reagent.eval import targets

    monkeypatch.setattr(targets, "TARGETS", cohort[0][1])
    monkeypatch.setattr(targets, "HARD_TARGETS", [])
    directory = tmp_path / "review"
    args = ["review", "prepare", "--moderate-checkpoint", str(cohort[0][2]),
            "--hard-checkpoint", str(cohort[0][2]), "--output", str(directory)]
    runner = CliRunner()
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert runner.invoke(main, args).exit_code != 0
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    result = runner.invoke(main, ["review", "report", str(directory)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["targets"] == 2


def test_references_require_provenance(cohort):
    bundle = prepare(cohort, "build-it-yourself")
    reviews, references, _ = worksheets(bundle)
    key = bundle["targets"][0]["id"]
    references["targets"][key] = [{"reactions": copy.deepcopy(
        bundle["targets"][0]["route"]["reactions"])}]
    with pytest.raises(ValueError):
        report(bundle, reviews, references)
