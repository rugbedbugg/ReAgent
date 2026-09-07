"""Exercise launcher environment and argument forwarding with a stand-in executable."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("custom", [False, True])
def test_native_launcher_data_path_and_arguments(tmp_path, custom):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "print(json.dumps([os.environ['REAGENT_DATA'], sys.argv[1:]]))\n"
        "sys.exit(7)\n"
    )
    env = dict(os.environ)
    env.pop("REAGENT_DATA", None)
    if sys.platform == "win32":
        env["LOCALAPPDATA"] = str(tmp_path / "Local App Data")
        source = ROOT / "SUBMISSIONS/chocolatey/tools/reagent.cmd"
        script = source.read_text().replace(
            '"%~dp0venv\\Scripts\\reagent.exe"', subprocess.list2cmdline([sys.executable, str(probe)]),
        )
        launcher = tmp_path / "reagent.cmd"
        launcher.write_text(script)
        command = ["cmd.exe", "/d", "/c", str(launcher), "plan", "argument with spaces"]
        expected = str(Path(env["LOCALAPPDATA"]) / "reagent")
    else:
        env["XDG_DATA_HOME"] = str(tmp_path / "data home")
        source = ROOT / "SUBMISSIONS/aur/reagent.sh"
        script = source.read_text().replace(
            "/opt/reagent/bin/reagent", shlex.join([sys.executable, str(probe)]),
        )
        command = ["sh", "-c", script, "reagent", "plan", "argument with spaces"]
        expected = str(Path(env["XDG_DATA_HOME"]) / "reagent")
    if custom:
        expected = str(tmp_path / "custom data")
        env["REAGENT_DATA"] = expected
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == [expected, ["plan", "argument with spaces"]]
