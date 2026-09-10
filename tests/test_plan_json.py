"""`plan --json` must emit one parseable document on stdout, and nothing else.

The whole point of the flag is that a caller can pipe stdout into a parser. If
any progress line, warning or note leaks onto stdout the document stops parsing,
and the failure is silent for anyone who is not checking. So these assert on
`json.loads(stdout)` rather than on substrings.
"""

import json

import pytest
from click.testing import CliRunner

from reagent.cli import main
from reagent.core.models import Molecule, Reaction, Route


class _Backend:
    """Enough surface for `plan`, and no more: a caller may supply any object
    with .plan(), so the JSON document must not demand a full backend."""

    search_hit_time_limit = False

    def __init__(self, *args, **kwargs):
        pass

    def plan(self, smiles, max_routes=10):
        return [
            Route(
                target=smiles,
                solved=True,
                reactions=[Reaction(product=smiles, precursors=["CC(=O)O", "CCN"])],
                leaves=[
                    Molecule(smiles="CC(=O)O", in_stock=True),
                    Molecule(smiles="CCN", in_stock=False),
                ],
            )
        ]


@pytest.fixture
def cli(monkeypatch, tmp_path):
    import reagent.singlestep.aizynth as aizynth

    monkeypatch.setattr(aizynth, "AiZynthBackend", _Backend)
    monkeypatch.chdir(tmp_path)
    return CliRunner()


def _document(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_stdout_is_a_single_parseable_document(cli):
    document = _document(cli.invoke(main, ["plan", "CCO", "--json"]))
    assert document["schema"] == 1
    assert document["target"] == "CCO"
    assert document["mode"] == "balanced"


def test_progress_never_reaches_stdout(cli):
    """The regression that would break every consumer: a status line on stdout."""
    result = cli.invoke(main, ["plan", "CCO", "--json"])
    assert result.exit_code == 0
    assert "Loading search backend" not in result.stdout
    assert "Target:" not in result.stdout
    json.loads(result.stdout)  # parses, which is the real check


def test_a_route_carries_its_reactions_and_leaves(cli):
    route = _document(cli.invoke(main, ["plan", "CCO", "--json"]))["routes"][0]
    assert route["solved"] is True
    assert route["steps"] == 1
    assert route["reactions"] == [
        {"step": 1, "precursors": ["CC(=O)O", "CCN"], "product": "CCO", "rsmi": None}
    ]
    assert route["leaves"] == [
        {"smiles": "CC(=O)O", "in_stock": True},
        {"smiles": "CCN", "in_stock": False},
    ]


def test_the_buy_versus_build_number_is_present(cli):
    """`largest_leaf_fraction` is the figure that distinguishes making a
    molecule from buying one, so a machine consumer needs it, not just a human
    reading the header."""
    route = _document(cli.invoke(main, ["plan", "CCO", "--json"]))["routes"][0]
    assert 0.0 <= route["largest_leaf_fraction"] <= 2.0
    assert isinstance(route["buys_most_of_target"], bool)


def test_the_mode_constraint_is_reported(cli):
    document = _document(cli.invoke(main, ["plan", "CCO", "--json", "--mode", "build"]))
    assert document["mode"] == "build"
    assert document["max_leaf_fraction"] == 0.6


def test_features_appear_only_when_asked(cli):
    plain = _document(cli.invoke(main, ["plan", "CCO", "--json"]))
    assert "features" not in plain["routes"][0]
    detailed = _document(cli.invoke(main, ["plan", "CCO", "--json", "--show-features"]))
    assert "construction" in detailed["routes"][0]["features"]


def test_a_partial_backend_does_not_break_the_document(cli):
    """_Backend deliberately lacks `algorithms` and `last_search_stats`. The
    search block is informational and must degrade rather than raise."""
    search = _document(cli.invoke(main, ["plan", "CCO", "--json"]))["search"]
    assert search["algorithms"] is None
    assert search["hit_time_limit"] is False


def test_text_output_is_unchanged_without_the_flag(cli):
    result = cli.invoke(main, ["plan", "CCO"])
    assert result.exit_code == 0
    assert "=== Route 1 (solved, 1 steps" in result.output
    assert "leaves (*=in stock): CC(=O)O*, CCN" in result.output
