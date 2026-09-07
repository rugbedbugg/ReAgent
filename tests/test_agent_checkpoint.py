"""Agent checks can use saved routes without loading search models."""

import pytest
from click.testing import CliRunner

from reagent.cli import main
from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.checkpoint import Checkpoint


@pytest.fixture
def agent_cli(monkeypatch):
    from reagent.agents.llm.ollama_client import OllamaClient
    from reagent.eval import targets

    monkeypatch.setattr(targets, "TARGETS", [("ethanol", "CCO"), ("propane", "CCC")])
    monkeypatch.setattr(OllamaClient, "complete", lambda *a, **k: '{"score": 0.5, "rationale": "mock"}')
    return CliRunner()


def saved_route():
    return Route(target="CCO", solved=True,
                 reactions=[Reaction(product="CCO", precursors=["C"],
                                     metadata={"policy_probability": 0.8})],
                 leaves=[Molecule(smiles="C", in_stock=True)])


def test_agent_cli_reads_routes_without_search_data(tmp_path, monkeypatch, agent_cli):
    saved = Checkpoint(tmp_path, {"schema": 1})
    saved.save("CCO", [saved_route()], False)
    saved.save("CCC", [], False)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint scoring must not load the search backend or config")

    monkeypatch.setattr("reagent.cli.aizynth_config", forbidden)
    monkeypatch.setattr("reagent.singlestep.aizynth.AiZynthBackend", forbidden)
    result = agent_cli.invoke(main, ["check-agents", "--checkpoint", str(tmp_path),
                                    "--max-targets", "2", "--local"])
    assert result.exit_code == 0, result.output
    assert "assessments scored: 7" in result.output
    assert "Loading search backend" not in result.output
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_incomplete_checkpoint_fails_before_any_model_call(tmp_path, monkeypatch, agent_cli):
    saved = Checkpoint(tmp_path, {"schema": 1})
    saved.save("CCO", [saved_route()], False)

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete evidence must fail before model calls")

    monkeypatch.setattr("reagent.agents.llm.ollama_client.OllamaClient.complete", forbidden)
    result = agent_cli.invoke(main, ["check-agents", "--checkpoint", str(tmp_path),
                                    "--max-targets", "2", "--local"])
    assert result.exit_code != 0
    assert "missing propane" in result.output


def test_hashed_stock_is_accepted_and_forwarded(tmp_path, monkeypatch, agent_cli):
    seen = []

    class Backend:
        def __init__(self, config, **kwargs):
            seen.append(kwargs)

        def plan(self, smiles, max_routes):
            return [saved_route()]

    monkeypatch.setattr("reagent.cli.aizynth_config", lambda: tmp_path / "unused")
    monkeypatch.setattr("reagent.singlestep.aizynth.AiZynthBackend", Backend)
    result = agent_cli.invoke(main, ["check-agents", "--hashed-stock", "--max-targets", "1", "--local"])
    assert result.exit_code == 0, result.output
    assert seen == [{"permissive_stock": None, "hashed_stock": True}]


def test_historical_reader_never_creates_or_writes_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError):
        Checkpoint.open_readonly(tmp_path)
    assert not list(tmp_path.iterdir())
    Checkpoint(tmp_path, {"schema": 1})
    with pytest.raises(ValueError, match="read-only"):
        Checkpoint.open_readonly(tmp_path).save("CCO", [], False)
