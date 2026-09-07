"""Run a checked-in evaluation plan serially with memory gates and checkpoints.

Usage: uv run --no-sync python -m reagent.eval.queue --plan data/evaluations/plans/v0.3.0.json
This runner never commits, publishes, installs packages or downloads models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from reagent.eval.checkpoint import _write
from reagent.eval.parallel import available_memory_mb


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def queue_lock(path):
    """OS-released lock: a killed runner must not leave a stale lock behind."""
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def existing_run_alive(metadata):
    """Recognize a recorded process by PID and command, not PID alone."""
    if sys.platform == "win32":
        # Popen builds this exact Windows command line from our argument list.
        # CIM also returns nothing for an exited process, allowing a clean
        # restart without treating a stale PID as a surviving queue child.
        pid = int(metadata["pid"])
        script = (
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
            "$ErrorActionPreference = 'Stop'; "
            f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
            "if ($p) { if ($null -eq $p.CommandLine) { throw 'Cannot read child command line' }; "
            "[Console]::Write($p.CommandLine) }"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, encoding="utf-8", check=True, timeout=15,
        )
        return result.stdout.strip() == subprocess.list2cmdline(metadata["command"])
    if not sys.platform.startswith("linux"):
        raise ValueError("Process recovery is supported on Linux and Windows")
    try:
        raw = Path(f"/proc/{int(metadata['pid'])}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    argv = [part.decode() for part in raw.split(b"\0") if part]
    expected = metadata["command"]
    return bool(argv) and Path(argv[0]).name == Path(expected[0]).name and argv[1:] == expected[1:]


def load_plan(path):
    raw = path.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema") != 1 or not plan.get("jobs"):
        raise ValueError("Expected a schema-1 plan with jobs")
    seen = set()
    checkpoints = set()
    for job in plan["jobs"]:
        name = job["id"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name):
            raise ValueError(f"Invalid job id: {name}")
        if name in seen or job["checkpoint"] in checkpoints:
            raise ValueError("Each job needs a unique id and checkpoint directory")
        if not isinstance(job["args"], list) or not all(isinstance(a, str) for a in job["args"]):
            raise ValueError("Job arguments must be strings")
        if "--checkpoint" in job["args"] or "--jobs" in job["args"]:
            raise ValueError("The runner supplies --checkpoint and --jobs 1")
        seen.add(name)
        checkpoints.add(job["checkpoint"])
    return plan, hashlib.sha256(raw).hexdigest()


def command_for(job):
    return [sys.executable, "-m", "reagent.cli", "evaluate", *job["args"],
            "--jobs", "1", "--checkpoint", job["checkpoint"]]


def stop_process(process):
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_command(command, log_path, minimum_running_mb, on_start):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            on_start(process.pid)
            while process.poll() is None:
                memory = available_memory_mb()
                if memory < minimum_running_mb:
                    stop_process(process)
                    return {"returncode": process.returncode, "reason": "low memory"}
                time.sleep(5)
        except BaseException:
            if process.poll() is None:
                stop_process(process)
            raise
    return {"returncode": process.returncode}


def execute(plan, digest, output, wait_for=None):
    output.mkdir(parents=True, exist_ok=True)
    minimum_start_mb = int(
        os.environ.get("REAGENT_MINIMUM_START_MB", plan["minimum_start_mb"])
    )
    with queue_lock(output / "queue.lock"):
        state_path = output / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "plan_sha256": digest, "jobs": {},
        }
        if state["plan_sha256"] != digest:
            raise ValueError("Plan changed; use a new --output directory")

        def save(status):
            state.update(status=status, updated_at=now())
            _write(state_path, state)

        # A killed runner can leave its child alive. Do not launch a second
        # writer on restart; wait for the exact recorded command to exit first.
        for previous in state["jobs"].values():
            if previous["status"] in ("running", "interrupted"):
                attempt = previous["attempts"][-1]
                if "pid" in attempt:
                    save("waiting for previous queue child")
                    while existing_run_alive(attempt):
                        time.sleep(15)

        if wait_for:
            metadata = json.loads(wait_for.read_text())
            save("waiting for existing evaluation")
            print(f"Waiting for existing evaluation PID {metadata['pid']}", flush=True)
            while existing_run_alive(metadata):
                time.sleep(15)

        for job in plan["jobs"]:
            previous = state["jobs"].setdefault(job["id"], {"attempts": []})
            if any(
                attempt.get("returncode") == 0 and "reason" not in attempt
                for attempt in previous.get("attempts", [])
            ):
                previous["status"] = "completed"
                continue
            state["current_job"] = job["id"]
            save("waiting for memory")
            announced = False
            while available_memory_mb() < minimum_start_mb:
                if not announced:
                    print(f"Waiting for {minimum_start_mb} MB available before {job['id']}",
                          flush=True)
                    announced = True
                time.sleep(30)
            log = output / f"{job['id']}-{len(previous['attempts']) + 1}.log"
            attempt = {"started_at": now(), "command": command_for(job), "log": str(log)}
            previous["attempts"].append(attempt)
            previous["status"] = "running"
            save("running")
            print(f"Starting {job['id']}; log {log}", flush=True)
            started = time.monotonic()

            def child_started(pid):
                attempt["pid"] = pid
                save("running")

            try:
                result = run_command(
                    attempt["command"], log, plan["minimum_running_mb"], child_started,
                )
            except BaseException:
                previous["status"] = "interrupted"
                save("interrupted")
                raise
            attempt.update(result, ended_at=now(), elapsed_seconds=time.monotonic() - started)
            failed = result["returncode"] != 0 or "reason" in result
            previous["status"] = "failed" if failed else "completed"
            save("failed" if failed else "running")
            if failed:
                print(f"Stopped at {job['id']}: {result}. Restart to reuse checkpoints.", flush=True)
                return 1
            print(f"Completed {job['id']}", flush=True)
        save("completed")
        print("Evaluation queue completed. Review results before the follow-up tasks.", flush=True)
        return 0


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/v0.3.0-queue"))
    parser.add_argument("--wait-for-run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan, digest = load_plan(args.plan)
    if args.dry_run:
        for job in plan["jobs"]:
            print(job["id"], json.dumps(command_for(job)))
        return 0
    return execute(plan, digest, args.output, args.wait_for_run)


if __name__ == "__main__":
    raise SystemExit(main())
