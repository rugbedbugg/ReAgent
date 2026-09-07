"""Local inference must preserve the mode through ranking and episode logging."""

import pytest
from click.testing import CliRunner

from reagent.adaptive.memory import EpisodicMemory
from reagent.cli import main
from reagent.core.models import Molecule, Reaction, Route
from reagent.optimize.aggregate import mode_weights


@pytest.mark.parametrize("mode", ["balanced", "build", "source"])
@pytest.mark.parametrize("hybrid", [False, True])
def test_local_plan_preserves_mode(mode, hybrid, tmp_path, monkeypatch):
    class Backend:
        search_hit_time_limit = False

        def __init__(self, *args, **kwargs):
            pass

        def plan(self, target, max_routes):
            return [Route(target=target, solved=True,
                          leaves=[Molecule(smiles="C", in_stock=True)],
                          reactions=[Reaction(product=target, precursors=["C"],
                                              metadata={"policy_probability": 0.8})])]

    memory = EpisodicMemory(tmp_path / "episodes.jsonl")
    monkeypatch.setattr("reagent.singlestep.aizynth.AiZynthBackend", Backend)
    monkeypatch.setattr("reagent.cli.aizynth_config", lambda: tmp_path / "unused")
    monkeypatch.setattr("reagent.adaptive.memory.EpisodicMemory", lambda: memory)
    monkeypatch.setattr("reagent.adaptive.weights.load_weights", lambda: mode_weights("balanced"))
    monkeypatch.setattr("reagent.agents.llm.ollama_client.OllamaClient.complete",
                        lambda *a, **k: '{"score": 0.5, "rationale": "mock"}')
    args = ["plan", "CCO", "--local", "--mode", mode]
    if hybrid:
        args.append("--hybrid")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "=== Ranking" in result.output
    assert memory.all()[0].weights == mode_weights(mode)
