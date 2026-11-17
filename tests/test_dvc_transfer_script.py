""".github/scripts/dvc-transfer.sh retries a DVC transfer that fails (#375).

A stand-in `uv` on the PATH fails a given number of times and then succeeds,
so the tests check the retries without DagsHub. Like tests/test_deploy_script.py,
they run where bash is a Linux shell, as on GitHub's runners.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from recommenditos.config import PROJ_ROOT

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="dvc-transfer.sh is a bash script for GitHub's Linux runners",
)

SCRIPT = PROJ_ROOT / ".github" / "scripts" / "dvc-transfer.sh"

#: Records its arguments, one call per line, and fails its first FAILURES calls.
FAKE_UV = """#!/usr/bin/env bash
echo "$*" >> "$CALLS"
made=$(grep -c . "$CALLS")
[ "$made" -gt "$FAILURES" ]
"""


def run(
    tmp_path: Path, failures: int, *arguments: str
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """The script's result against a `uv` that fails `failures` times, and the calls it got."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(FAKE_UV, encoding="utf-8", newline="\n")
    uv.chmod(0o755)
    calls = tmp_path / "calls"
    calls.touch()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CALLS": str(calls),
        "FAILURES": str(failures),
        "DVC_TRANSFER_WAIT": "0",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    return result, calls.read_text(encoding="utf-8").splitlines()


def test_a_transfer_that_succeeds_runs_once(tmp_path: Path):
    result, calls = run(tmp_path, 0, "pull", "models/lgbm-basic", "data/monitoring")

    assert result.returncode == 0
    assert calls == ["run dvc pull models/lgbm-basic data/monitoring"]
    assert "::warning::" not in result.stdout


def test_a_failed_transfer_is_tried_again_and_each_retry_is_logged(tmp_path: Path):
    result, calls = run(tmp_path, 2, "push")

    assert result.returncode == 0
    assert calls == ["run dvc push"] * 3
    assert result.stdout.count("::warning::dvc push failed") == 2


def test_a_transfer_that_keeps_failing_fails_after_the_last_attempt(tmp_path: Path):
    result, calls = run(tmp_path, 5, "pull")

    assert result.returncode == 1
    assert len(calls) == 3
    assert "::error::dvc pull failed 3 times" in result.stdout
