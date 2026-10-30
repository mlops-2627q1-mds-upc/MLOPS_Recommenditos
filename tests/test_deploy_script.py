"""deploy/deploy.sh refuses every command but its three, before it touches anything.

The Deploy workflow's self-hosted runner on the VM may run nothing but deploy.sh,
through sudo as the deploy user (deploy/install-runner.sh writes that sudoers
rule). sudo passes the arguments as they are, without a shell, so the script's
own check of their shape is what stands between the runner and the stack
(docs/docs/deployment.md#turn-continuous-deployment-on).
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
    [],
    [""],
    ["deploy"],
    ["deploy", ""],
    ["deploy", SHA, "extra"],
    [f"deploy {SHA}"],
    ["deploy", f"{SHA};id"],
    ["deploy", f"{SHA}\n"],
    ["deploy", f" {SHA}"],
    ["deploy", "$(id)"],
    ["deploy", "../../etc"],
    ["deploy", "-rf"],
    ["deploy", "latest"],
    ["deploy", SHA[:12]],
    ["deploy", SHA.upper()],
    ["--help"],
    ["bash"],
    ["rollback", "now"],
    ["status", "--all"],
    ["status", "deploy", SHA],
    ["deploy", SHA, "--registry-token"],
    ["deploy", SHA, "--registry-token-stdin=x"],
    ["deploy", "--registry-token-stdin", SHA],
    ["deploy", SHA, "--registry-token-stdin", "extra"],
    ["rollback", "--registry-token-stdin"],
]

FLAG = "--registry-token-stdin"
#: Stands in for the registry token: any one word of visible characters.
SENTINEL = "stand-in-for-the-job-token-0123456789"
ACCEPTED = [["deploy", SHA], ["deploy", SHA, FLAG], ["rollback"], ["status"]]

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


def run_directly(
    stack_dir: Path,
    arguments: list[str],
    locale: str = "C.UTF-8",
    extra_environment: dict[str, str] | None = None,
    stdin: str = f"{SENTINEL}\n",
) -> subprocess.CompletedProcess[str]:
    """deploy.sh with `arguments` as its argv, as sudo starts it for the runner."""
    environment = {
        **os.environ,
        "STACK_DIR": str(stack_dir),
        "LC_ALL": locale,
        **(extra_environment or {}),
    }
    return subprocess.run(
        ["bash", str(PROJ_ROOT / "deploy" / "deploy.sh"), *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=30,
        check=False,
    )


@pytest.mark.req("NFR-15")
@pytest.mark.parametrize("arguments", REFUSED, ids=[repr(arguments) for arguments in REFUSED])
def test_the_runner_runs_nothing_but_the_three_commands(tmp_path: Path, arguments: list[str]):
    env_file = tmp_path / ".env"
    env_file.write_text("API_IMAGE_TAG=good\n", encoding="utf-8")

    result = run_directly(tmp_path, arguments)

    assert result.returncode == 64, result.stdout + result.stderr
    assert env_file.read_text(encoding="utf-8") == "API_IMAGE_TAG=good\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env"]


@pytest.mark.req("NFR-15")
def test_a_command_in_ssh_original_command_is_not_read_any_more(tmp_path: Path):
    """The SSH path is gone: without arguments, the script refuses to do anything."""
    (tmp_path / ".env").write_text("API_IMAGE_TAG=good\n", encoding="utf-8")

    result = run_directly(tmp_path, [], extra_environment={"SSH_ORIGINAL_COMMAND": "status"})

    assert result.returncode == 64, result.stdout + result.stderr


@pytest.mark.req("NFR-15")
@pytest.mark.parametrize(
    "arguments", ACCEPTED, ids=[" ".join(arguments) for arguments in ACCEPTED]
)
def test_the_three_commands_get_past_the_check(tmp_path: Path, arguments: list[str]):
    """Without `.env`, an accepted command stops at the next check, before Docker."""
    result = run_directly(tmp_path, arguments)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "run bootstrap.sh first" in result.stdout


@pytest.mark.req("NFR-15")
@pytest.mark.skipif(
    not locale_available(WIDE_LOCALE),
    reason=f"{WIDE_LOCALE} is not installed, and in C the check would pass without LC_ALL=C",
)
def test_a_letter_outside_hex_is_refused_whatever_the_callers_locale(tmp_path: Path):
    (tmp_path / ".env").write_text("API_IMAGE_TAG=good\n", encoding="utf-8")

    result = run_directly(tmp_path, ["deploy", f"{SHA[:-1]}é"], locale=WIDE_LOCALE)

    assert result.returncode == 64, result.stdout + result.stderr


@pytest.mark.req("NFR-15")
@pytest.mark.parametrize(
    "stdin",
    ["", "\n", "two words\n", f" {SENTINEL}\n", "x" * 4097 + "\n"],
    ids=["nothing", "empty line", "a space", "leading space", "too long"],
)
def test_the_registry_flag_needs_one_token_on_standard_input(tmp_path: Path, stdin: str):
    env_file = tmp_path / ".env"
    env_file.write_text("API_IMAGE_TAG=good\n", encoding="utf-8")

    result = run_directly(tmp_path, ["deploy", SHA, FLAG], stdin=stdin)

    assert result.returncode == 64, result.stdout + result.stderr
    assert "no registry token" in result.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env"]


#: A stand-in for `docker` that records each call's arguments and environment,
#: and what `docker login` reads on standard input. The image is not on the VM,
#: and the pull fails, so deploy.sh stops before it starts anything.
FAKE_DOCKER = """#!/usr/bin/env bash
log=$FAKE_DOCKER_LOG
{ printf 'argv:'; printf ' %s' "$@"; printf '\\n'; env | sed 's/^/env: /'; } >>"$log"
case "$*" in
    *"config --images api"*) echo "ghcr.io/example/api:$API_IMAGE_TAG" ;;
    "image inspect "*) exit 1 ;;
    "login "*)
        printf 'stdin: %s\\n' "$(cat)" >>"$log"
        printf 'config exists: %s\\n' "$([ -d "$DOCKER_CONFIG" ] && echo yes)" >>"$log"
        ;;
    *" pull "*)
        printf 'pull with DOCKER_CONFIG=%s\\n' "${DOCKER_CONFIG:-}" >>"$log"
        exit 1
        ;;
esac
"""


@pytest.mark.req("NFR-15")
def test_the_registry_token_reaches_docker_login_on_stdin_and_nothing_keeps_it(tmp_path: Path):
    stack = tmp_path / "stack"
    stack.mkdir()
    (stack / ".env").write_text(f"API_IMAGE_TAG={'1' * 40}\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    log = tmp_path / "docker.log"

    result = run_directly(
        stack,
        ["deploy", SHA, FLAG],
        extra_environment={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(log),
            "TMPDIR": str(tmp_path),
        },
    )

    calls = log.read_text(encoding="utf-8")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "pulling ghcr.io/example/api:" in result.stdout
    assert f"stdin: {SENTINEL}" in calls
    assert "config exists: yes" in calls
    # Only `docker login` saw the token, on its standard input.
    assert calls.count(SENTINEL) == 1
    pull_config = calls.split("pull with DOCKER_CONFIG=")[1].splitlines()[0]
    assert pull_config.startswith(str(tmp_path))
    assert not Path(pull_config).exists()
    for path in stack.iterdir():
        assert SENTINEL not in path.read_text(encoding="utf-8"), path.name
    assert SENTINEL not in result.stdout + result.stderr
