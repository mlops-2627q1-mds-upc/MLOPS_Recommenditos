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
    f"status\ndeploy {SHA}",
    "bash",
    "rollback now",
    "status --all",
]

ACCEPTED = [f"deploy {SHA}", "rollback", "status"]

#: A locale whose ranges include letters such as é in a-f, where a pattern
#: compared without LC_ALL=C would take `é` for a hex digit.
WIDE_LOCALE = "en_US.UTF-8"


def locale_available(name: str) -> bool:
    """Whether the system has `name`; bash falls back to C silently otherwise."""
    if shutil.which("locale") is None:
        return False
    listed = subprocess.run(
        ["locale", "-a"], capture_output=True, text=True, timeout=30, check=False
    )
    wanted = name.lower().replace("-", "")
    return any(line.lower().replace("-", "") == wanted for line in listed.stdout.splitlines())


def run_over_ssh(
    stack_dir: Path, command: str, locale: str = "C.UTF-8"
) -> subprocess.CompletedProcess[str]:
    """deploy.sh as sshd starts it for the restricted key."""
    environment = {
        **os.environ,
        "STACK_DIR": str(stack_dir),
        "SSH_ORIGINAL_COMMAND": command,
        "LC_ALL": locale,
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


@pytest.mark.req("NFR-15")
@pytest.mark.skipif(
    not locale_available(WIDE_LOCALE),
    reason=f"{WIDE_LOCALE} is not installed, and in C the check would pass without LC_ALL=C",
)
def test_a_letter_outside_hex_is_refused_whatever_the_callers_locale(tmp_path: Path):
    (tmp_path / ".env").write_text("API_IMAGE_TAG=good\n", encoding="utf-8")

    result = run_over_ssh(tmp_path, f"deploy {SHA[:-1]}é", locale=WIDE_LOCALE)

    assert result.returncode == 64, result.stdout + result.stderr
