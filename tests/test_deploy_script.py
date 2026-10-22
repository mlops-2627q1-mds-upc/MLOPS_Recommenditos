"""deploy/deploy.sh refuses every command but its three, before it touches anything.

The Deploy workflow's SSH key may run nothing but deploy.sh, which reads the
command the client asked for from SSH_ORIGINAL_COMMAND (docs/docs/deployment.md).
The drill that deploys and rolls back is tests/integration/test_deploy_rollback.py.
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
    reason="deploy.sh is a bash script for the Linux VM",
)

SHA = "0123456789abcdef0123456789abcdef01234567"

REFUSED = [
    "",
    "deploy",
    f"deploy {SHA} extra",
    f"deploy '{SHA};id'",
    f"deploy {SHA};id",
    "deploy $(id)",
    "deploy ../../etc",
    "deploy -rf",
    "deploy latest",
    f"deploy {SHA[:12]}",
    f"deploy {SHA.upper()}",
    f"deploy {SHA[:-1]}é",
    f"status\ndeploy {SHA}",
    "bash",
    "rollback now",
    "status --all",
]

ACCEPTED = [f"deploy {SHA}", "rollback", "status"]


def run_over_ssh(stack_dir: Path, command: str) -> subprocess.CompletedProcess[str]:
    """deploy.sh as sshd starts it for the restricted key, in a locale whose ranges
    include letters such as é in a-f, as en_US.UTF-8's do."""
    environment = {
        **os.environ,
        "STACK_DIR": str(stack_dir),
        "SSH_ORIGINAL_COMMAND": command,
        "LC_ALL": "en_US.UTF-8",
    }
    return subprocess.run(
        ["bash", str(PROJ_ROOT / "deploy" / "deploy.sh")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=30,
        check=False,
    )


@pytest.mark.req("NFR-15")
@pytest.mark.parametrize("command", REFUSED, ids=[repr(command) for command in REFUSED])
def test_the_restricted_key_runs_nothing_but_the_three_commands(tmp_path: Path, command: str):
    env_file = tmp_path / ".env"
    env_file.write_text("API_IMAGE_TAG=good\n", encoding="utf-8")

    result = run_over_ssh(tmp_path, command)

    assert result.returncode == 64, result.stdout + result.stderr
    assert env_file.read_text(encoding="utf-8") == "API_IMAGE_TAG=good\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env"]


@pytest.mark.req("NFR-15")
@pytest.mark.parametrize("command", ACCEPTED)
def test_the_three_commands_get_past_the_check(tmp_path: Path, command: str):
    """Without `.env`, an accepted command stops at the next check, before Docker."""
    result = run_over_ssh(tmp_path, command)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "run bootstrap.sh first" in result.stdout
