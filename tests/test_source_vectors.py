"""The cached source comparison must not invent missing route identities."""

import runpy
from pathlib import Path

import pytest
from click.testing import CliRunner

from reagent.cli import main

compare = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "docs/measurements/compare_source_vectors.py")
)["compare"]


def test_identical_ties_do_not_inflate_target_means():
    scores = dict.fromkeys(
        ("feasibility", "availability", "cost", "safety", "construction",
         "sustainability", "efficiency"), 0.5,
    )
    result = compare({"a": [scores, scores], "b": []}, [("a", "C"), ("b", "CC")])
    assert result["solved_targets"] == 1
    assert result["unsolved_targets"] == ["b"]
    assert result["changed_score_vectors"] == 0
    assert result["mean_scores_over_solved_targets"]["source"] == scores
    assert result["per_target"][0]["modes"]["source"]["candidate_indices"] == [0, 1]


def test_distinct_tied_vectors_require_the_missing_route_identity():
    # This objective carries no weight, so either vector could win by signature.
    with pytest.raises(ValueError, match="need route identities"):
        compare({"a": [{"unweighted": 0.1}, {"unweighted": 0.9}]}, [("a", "C")])


def test_evaluate_source_and_balanced_share_search_and_report_source(monkeypatch):
    from reagent.singlestep import aizynth

    calls = []

    class Backend:
        search_hit_time_limit = False

        def __init__(self, config, **kwargs):
            calls.append(kwargs)

        def plan(self, smiles, max_routes):
            return []

    monkeypatch.setattr(aizynth, "AiZynthBackend", Backend)
    monkeypatch.setattr("reagent.cli.aizynth_config", lambda: Path("unused.yml"))
    results = [
        CliRunner().invoke(main, ["evaluate", "--max-targets", "1", "--mode", mode])
        for mode in ("balanced", "source")
    ]
    for result in results:
        assert result.exit_code == 0, result.output
        assert "=== source-led ===" in result.output
        assert "=== feasibility-led ===" in result.output
    assert calls[0] == calls[1]
    assert calls[0]["max_leaf_fraction"] is None
    assert results[0].output == results[1].output
