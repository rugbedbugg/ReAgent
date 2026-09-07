"""An interrupted experiment must resume without losing or mixing evidence."""

import json

import pytest
from click.testing import CliRunner

from reagent.cli import main
from reagent.core.models import Molecule, Reaction, Route
from reagent.eval.checkpoint import Checkpoint, search_identity


def route(target="CCO"):
    return Route(
        target=target, solved=True,
        leaves=[Molecule(smiles="C", in_stock=True)],
        reactions=[Reaction(product=target, precursors=["C"],
                            metadata={"policy_probability": 0.8})],
        tree={"type": "mol", "smiles": target},
    )


def test_roundtrip_includes_full_routes_empty_results_and_clock(tmp_path):
    saved = Checkpoint(tmp_path, {"iterations": 500})
    saved.save("CCO", [route()], True)
    saved.save("CCC", [], False)
    reopened = Checkpoint(tmp_path, {"iterations": 500})
    result = reopened.load("CCO")
    assert result.routes == [route()]
    assert result.time_capped is True
    assert reopened.load("CCC").routes == []
    assert reopened.load("CC") is None
    with pytest.raises(ValueError, match="differ"):
        Checkpoint(tmp_path, {"iterations": 100})


def test_failed_replace_preserves_previous_complete_result(tmp_path, monkeypatch):
    saved = Checkpoint(tmp_path, {})
    saved.save("CCO", [route()], False)

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("reagent.eval.checkpoint.os.replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        saved.save("CCO", [], True)
    assert saved.load("CCO").routes == [route()]
    assert saved.load("CCO").time_capped is False
    assert not list(tmp_path.glob("*.tmp"))


def test_corruption_and_wrong_target_are_not_silently_replanned(tmp_path):
    saved = Checkpoint(tmp_path, {})
    saved.save("CCO", [], False)
    path = next(p for p in tmp_path.glob("*.json") if p.name != "manifest.json")
    path.write_text('{"partial":', encoding="utf-8")
    with pytest.raises(ValueError):
        saved.load("CCO")
    path.write_text(json.dumps({"target": "CCC", "routes": [], "time_capped": False}))
    with pytest.raises(ValueError, match="target mismatch"):
        saved.load("CCO")


def test_identity_detects_changed_model_and_stock_contents(tmp_path):
    model = tmp_path / "model.onnx"
    stock = tmp_path / "stock.npy"
    config = tmp_path / "config.yml"
    model.write_bytes(b"model1")
    stock.write_bytes(b"stock1")
    config.write_text(json.dumps({"expansion": {"uspto": str(model)}}))
    kwargs = {"expansion": ["uspto"], "hashed_stock": True, "stock_cache": str(stock)}
    first = search_identity(config, kwargs, 15)
    model.write_bytes(b"model2")
    second = search_identity(config, kwargs, 15)
    assert first != second
    stock.write_bytes(b"stock2")
    assert second != search_identity(config, kwargs, 15)


def test_cli_resumes_only_missing_targets_and_reproduces_report(tmp_path, monkeypatch):
    from reagent.eval import checkpoint, targets
    from reagent.singlestep import aizynth

    monkeypatch.setattr(targets, "TARGETS", [("one", "CCO"), ("two", "CCC"), ("three", "CC")])
    monkeypatch.setattr("reagent.cli.aizynth_config", lambda: tmp_path / "unused")
    monkeypatch.setattr(checkpoint, "search_identity", lambda config, kwargs, count: {
        "kwargs": kwargs, "max_routes": count,
    })
    calls = []
    fail = True

    class Backend:
        search_hit_time_limit = False

        def __init__(self, *args, **kwargs):
            pass

        def plan(self, smiles, max_routes):
            calls.append(smiles)
            if smiles == "CCC" and fail:
                raise RuntimeError("interrupted")
            self.search_hit_time_limit = smiles == "CCO"
            return [] if smiles == "CCC" else [route(smiles)]

    monkeypatch.setattr(aizynth, "AiZynthBackend", Backend)
    runner = CliRunner()
    args = ["evaluate", "--max-targets", "3", "--checkpoint", str(tmp_path / "run")]
    interrupted = runner.invoke(main, args)
    assert isinstance(interrupted.exception, RuntimeError)
    fail = False
    calls.clear()
    resumed = runner.invoke(main, args)
    assert resumed.exit_code == 0, resumed.output
    assert calls == ["CCC", "CC"]
    assert "1 of 3 searches stopped" in resumed.output
    calls.clear()
    completed = runner.invoke(main, args + ["--mode", "source"])
    assert completed.exit_code == 0, completed.output
    assert calls == []  # includes the completed, unsolved target
    assert "Loading search backend" not in completed.output
    fresh = runner.invoke(main, ["evaluate", "--max-targets", "3"])
    assert fresh.exit_code == 0, fresh.output
    assert resumed.output.split("===", 1)[1] == fresh.output.split("===", 1)[1]
    assert completed.output.split("===", 1)[1] == fresh.output.split("===", 1)[1]
    mismatch = runner.invoke(main, args + ["--iterations", "500"])
    assert mismatch.exit_code != 0
    assert "differ" in mismatch.output


def test_parallel_results_are_saved_before_progress_even_if_next_worker_fails(
    tmp_path, monkeypatch,
):
    import multiprocessing
    from types import SimpleNamespace

    from reagent.eval.parallel import plan_targets

    saved = Checkpoint(tmp_path, {})
    progress = []

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, *args):
            yield "one", "CCO", [route()], True
            raise RuntimeError("worker failed")

    monkeypatch.setattr(multiprocessing, "get_context", lambda *a: SimpleNamespace(Pool=Pool))

    def report(name):
        assert saved.load("CCO").time_capped
        progress.append(name)

    with pytest.raises(RuntimeError, match="worker failed"):
        plan_targets(
            [("one", "CCO"), ("two", "CC")], 15, {}, 2,
            on_result=saved.save, on_done=report, retain_results=False,
        )
    assert progress == ["one"]
    assert saved.load("CCO").routes == [route()]
    assert saved.load("CC") is None
