"""The `monitor` service's loop sleeps a valid interval between passes (monitor-loop.sh).

The loop runs with stubs of `python` and `sleep` on PATH: `python` notes the
pass, `sleep` notes its argument and ends the loop's shell, so a case is one
pass and one sleep.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

LOOP = Path(__file__).parents[1] / "monitor-loop.sh"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")


def run_loop(tmp_path: Path, interval: str | None) -> tuple[list[str], list[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    passes, sleeps = tmp_path / "passes", tmp_path / "sleeps"
    for name, body in (
        ("python", f'echo "$*" >> "{passes}"'),
        ("sleep", f'echo "$1" >> "{sleeps}"; kill "$PPID"'),
    ):
        stub = bin_dir / name
        stub.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        stub.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "RECOMMENDITOS_MONITOR_INTERVAL_S"
    }
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    if interval is not None:
        env["RECOMMENDITOS_MONITOR_INTERVAL_S"] = interval
    result = subprocess.run(
        ["sh", str(LOOP)], env=env, capture_output=True, text=True, timeout=30, check=False
    )
    return (
        passes.read_text(encoding="utf-8").splitlines(),
        sleeps.read_text(encoding="utf-8").splitlines(),
        result.stderr,
    )


@pytest.mark.parametrize("interval", ["900", "60", "1"])
def test_the_loop_runs_a_pass_then_sleeps_the_interval(tmp_path, interval):
    passes, sleeps, stderr = run_loop(tmp_path, interval)

    assert passes == ["-m recommenditos_monitoring run-once"]
    assert sleeps == [interval]
    assert stderr == ""


@pytest.mark.parametrize("interval", ["", "0", "000", "-5", "1.5", "15m", "abc", " 60"])
def test_an_interval_that_is_not_a_positive_whole_number_falls_back_to_900_with_a_warning(
    tmp_path, interval
):
    passes, sleeps, stderr = run_loop(tmp_path, interval)

    assert passes == ["-m recommenditos_monitoring run-once"]
    assert sleeps == ["900"]
    assert (
        f"RECOMMENDITOS_MONITOR_INTERVAL_S is '{interval}', not a positive whole number" in stderr
    )


def test_no_interval_sleeps_900_without_a_warning(tmp_path):
    _, sleeps, stderr = run_loop(tmp_path, None)

    assert sleeps == ["900"]
    assert stderr == ""
