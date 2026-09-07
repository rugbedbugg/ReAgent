"""Queue failure and memory gates must preserve experiment ordering."""

import json
import subprocess
import sys

import pytest

from reagent.eval import queue


@pytest.mark.parametrize("alive", [True, False])
def test_windows_recovery_handles_present_and_exited_children(monkeypatch, alive):
    command = [r"C:\Program Files\Python\python.exe", "-m", "reagent.cli", "evaluate"]
    monkeypatch.setattr(queue.sys, "platform", "win32")
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=subprocess.list2cmdline(command) if alive else "", stderr="",
    ))
    assert queue.existing_run_alive({"pid": 123, "command": command}) is alive


@pytest.mark.skipif(sys.platform not in ("linux", "win32"), reason="supported process probes")
def test_recovery_recognizes_a_real_child_and_its_exit():
    command = [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline() == "ready\n"
        assert queue.existing_run_alive({"pid": process.pid, "command": command})
        assert not queue.existing_run_alive({"pid": process.pid, "command": ["other"]})
    finally:
        process.terminate()
        process.wait(timeout=10)
        process.stdout.close()
    assert not queue.existing_run_alive({"pid": process.pid, "command": command})


def plan():
    return {
        "schema": 1, "minimum_start_mb": 4608, "minimum_running_mb": 768,
        "jobs": [
            {"id": name, "args": ["--max-targets", "1"], "checkpoint": f"data/{name}"}
            for name in ("one", "two")
        ],
    }


def test_queue_stops_on_failure_and_retries_checkpointed_jobs(tmp_path, monkeypatch):
    calls = []
    fail = True
    monkeypatch.setattr(queue, "available_memory_mb", lambda: 8192)

    def run(command, log, floor, on_start):
        calls.append(command[-1])
        on_start(123)
        return {"returncode": 1 if fail else 0}

    monkeypatch.setattr(queue, "run_command", run)
    assert queue.execute(plan(), "digest", tmp_path) == 1
    assert calls == ["data/one"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "failed"
    assert state["jobs"]["one"]["attempts"][0]["pid"] == 123
    fail = False
    calls.clear()
    assert queue.execute(plan(), "digest", tmp_path) == 0
    assert calls == ["data/one", "data/two"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert len(state["jobs"]["one"]["attempts"]) == 2
    assert state["status"] == "completed"
    with pytest.raises(ValueError, match="Plan changed"):
        queue.execute(plan(), "different", tmp_path)


def test_queue_waits_for_memory_before_launching(tmp_path, monkeypatch):
    readings = iter([1024, 8192, 8192])
    events = []
    monkeypatch.setattr(queue, "available_memory_mb", lambda: next(readings))
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: events.append("wait"))

    def run(*args):
        events.append("run")
        return {"returncode": 0}

    monkeypatch.setattr(queue, "run_command", run)
    assert queue.execute(plan(), "digest", tmp_path) == 0
    assert events == ["wait", "run", "run"]


def test_low_memory_stops_own_child_and_records_reason(tmp_path, monkeypatch):
    class Child:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

    child = Child()
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(queue, "available_memory_mb", lambda: 500)
    monkeypatch.setattr(queue, "stop_process", lambda p: setattr(p, "returncode", -15))
    pids = []
    result = queue.run_command(["unused"], tmp_path / "log", 768, pids.append)
    assert result == {"returncode": -15, "reason": "low memory"}
    assert pids == [123]


def test_queue_lock_rejects_a_second_runner(tmp_path):
    with queue.queue_lock(tmp_path / "lock"):
        with pytest.raises(OSError), queue.queue_lock(tmp_path / "lock"):
            pytest.fail("second runner acquired the lock")
    with queue.queue_lock(tmp_path / "lock"):
        pass


def test_plan_rejects_shared_checkpoint_directories(tmp_path):
    data = plan()
    data["jobs"][1]["checkpoint"] = data["jobs"][0]["checkpoint"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        queue.load_plan(path)


def test_restart_waits_for_recorded_child_before_reusing_checkpoint(tmp_path, monkeypatch):
    state = {"plan_sha256": "digest", "jobs": {"one": {
        "status": "running", "attempts": [{"pid": 123, "command": ["old"]}],
    }}}
    (tmp_path / "state.json").write_text(json.dumps(state))
    alive = iter([True, False])
    events = []
    monkeypatch.setattr(queue, "existing_run_alive", lambda metadata: next(alive))
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: events.append("wait"))
    monkeypatch.setattr(queue, "available_memory_mb", lambda: 8192)

    def run(*args):
        events.append("run")
        return {"returncode": 0}

    monkeypatch.setattr(queue, "run_command", run)
    assert queue.execute(plan(), "digest", tmp_path) == 0
    assert events == ["wait", "run", "run"]
