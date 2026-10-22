"""The rollback drill of FR-15, with deploy/deploy.sh against a stack on this machine.

deploy.sh is what the Deploy workflow runs on the VM (docs/docs/deployment.md#every-deploy).
Here it runs on a copy of the VM's stack directory, `compose.yaml`, `nginx.conf`,
`.env` and the script, as a second Compose project beside the stack under test,
on a port of its own. The good image is the one the stack under test runs; the
broken one is that image with a server that exits as soon as it starts, like an
image whose bundle does not load. The images get a repository of their own, so
deploy.sh's clean-up of old tags never reaches the stack's images.
"""

from collections.abc import Iterator
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess

import httpx2
import pytest
from tests.integration.conftest import Container, docker

from recommenditos.config import PROJ_ROOT

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

PROJECT = "recommenditos-drill"
REPOSITORY = "recommenditos-drill"
BROKEN_DOCKERFILE = """\
FROM {base}
CMD ["python", "-c", "raise SystemExit('broken on purpose by the rollback drill')"]
"""
#: How long deploy.sh waits for a healthy stack. The broken image never gets
#: there, so this bounds the drill; the good one is healthy in seconds.
WAIT_TIMEOUT_S = 60
#: Tags in the form deploy.sh accepts, full commit SHAs, named for the drill.
GOOD, NEWER, NEWEST, BROKEN = (
    hashlib.sha1(name.encode(), usedforsecurity=False).hexdigest()
    for name in ("good", "newer", "newest", "broken")
)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Stack:
    """A copy of the VM's stack directory, driven with deploy.sh as the workflow does."""

    def __init__(self, directory: Path, port: int) -> None:
        self.directory = directory
        self.port = port

    def deploy_sh(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        # Without the variables CI's environment sets for the stack under test,
        # which Compose would prefer over the copy's .env.
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("API_IMAGE", "COMPOSE_", "RECOMMENDITOS_", "HTTP_PORT"))
        }
        environment.update(STACK_DIR=str(self.directory), WAIT_TIMEOUT=str(WAIT_TIMEOUT_S))
        return subprocess.run(
            ["bash", str(self.directory / "deploy.sh"), *arguments],
            capture_output=True,
            text=True,
            env=environment,
            timeout=WAIT_TIMEOUT_S * 3 + 120,
            check=False,
        )

    def deploy_sh_and_hang_up(self, *arguments: str) -> int:
        """Run deploy.sh and close its output once it starts the new tag.

        That is what a cancelled workflow run or a lost SSH connection does to it.
        """
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("API_IMAGE", "COMPOSE_", "RECOMMENDITOS_", "HTTP_PORT"))
        }
        environment.update(STACK_DIR=str(self.directory), WAIT_TIMEOUT=str(WAIT_TIMEOUT_S))
        process = subprocess.Popen(
            ["bash", str(self.directory / "deploy.sh"), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith("deploy.sh: deploying"):
                break
        process.stdout.close()
        return process.wait(timeout=WAIT_TIMEOUT_S * 3 + 120)

    def env_tag(self) -> str:
        lines = (self.directory / ".env").read_text(encoding="utf-8").splitlines()
        return [line.split("=", 1)[1] for line in lines if line.startswith("API_IMAGE_TAG=")][-1]

    def running_image(self) -> str:
        ids = docker(
            "ps",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={PROJECT}",
            "--filter",
            "label=com.docker.compose.service=api",
        ).split()
        assert len(ids) == 1, f"expected one api container of {PROJECT}, found {len(ids)}"
        return docker("inspect", "--format", "{{.Config.Image}}", ids[0])

    def health(self) -> httpx2.Response:
        return httpx2.get(f"http://127.0.0.1:{self.port}/health", timeout=10)


@pytest.fixture
def drill(tmp_path: Path, api_container: Container) -> Iterator[Stack]:
    """The VM's stack directory, three good tags of the image under test and a broken one."""
    good = api_container.inspect()["Config"]["Image"]
    for tag in (GOOD, NEWER, NEWEST):
        docker("tag", good, f"{REPOSITORY}:{tag}")
    subprocess.run(
        ["docker", "build", "--quiet", "--tag", f"{REPOSITORY}:{BROKEN}", "-"],
        input=BROKEN_DOCKERFILE.format(base=good),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    directory = tmp_path / "recommenditos"
    directory.mkdir()
    for name in ("compose.yaml", "nginx.conf", "deploy.sh"):
        shutil.copy(PROJ_ROOT / "deploy" / name, directory / name)
    port = free_port()
    (directory / ".env").write_text(
        f"COMPOSE_PROJECT_NAME={PROJECT}\nAPI_IMAGE={REPOSITORY}\nAPI_IMAGE_TAG=\nHTTP_PORT={port}\n",
        encoding="utf-8",
    )
    try:
        yield Stack(directory, port)
    finally:
        subprocess.run(
            ["docker", "compose", "--project-directory", str(directory), "down", "--volumes"],
            capture_output=True,
            timeout=120,
            check=False,
        )
        for tag in (GOOD, NEWER, NEWEST, BROKEN):
            subprocess.run(
                ["docker", "image", "rm", f"{REPOSITORY}:{tag}"],
                capture_output=True,
                timeout=60,
                check=False,
            )


def last_line(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.req("FR-15", "NFR-13")
def test_a_broken_tag_is_rolled_back_and_the_previous_one_serves_again(drill: Stack) -> None:
    first = drill.deploy_sh("deploy", GOOD)
    assert first.returncode == 0, first.stdout + first.stderr
    assert last_line(first) == f"running: {GOOD}"
    assert drill.health().status_code == 200

    broken = drill.deploy_sh("deploy", BROKEN)

    assert broken.returncode == 1, broken.stdout + broken.stderr
    assert last_line(broken) == f"running: {GOOD}"
    assert drill.env_tag() == GOOD
    assert drill.running_image() == f"{REPOSITORY}:{GOOD}"
    assert drill.health().status_code == 200
    history = (drill.directory / "deploy-history.log").read_text(encoding="utf-8")
    assert f"rollback {BROKEN} -> {GOOD} ok" in history


@pytest.mark.req("FR-15", "NFR-13")
def test_rollback_after_a_failed_smoke_check_brings_the_previous_tag_back(drill: Stack) -> None:
    assert drill.deploy_sh("deploy", GOOD).returncode == 0
    newer = drill.deploy_sh("deploy", NEWER)
    assert newer.returncode == 0, newer.stdout + newer.stderr
    assert drill.running_image() == f"{REPOSITORY}:{NEWER}"

    # What the workflow runs when `make smoke` fails against the new tag.
    rollback = drill.deploy_sh("rollback")

    assert rollback.returncode == 0, rollback.stdout + rollback.stderr
    assert last_line(rollback) == f"running: {GOOD}"
    assert drill.running_image() == f"{REPOSITORY}:{GOOD}"
    assert drill.health().status_code == 200
    # One step back only: a second rollback would bring the failed tag back.
    again = drill.deploy_sh("rollback")
    assert again.returncode == 1
    assert last_line(again) == f"running: {GOOD}"


def has_image(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", f"{REPOSITORY}:{tag}"],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.req("NFR-04")
def test_a_deploy_removes_the_image_two_deploys_back_and_keeps_the_rest(drill: Stack) -> None:
    for tag in (GOOD, NEWER, NEWEST):
        result = drill.deploy_sh("deploy", tag)
        assert result.returncode == 0, result.stdout + result.stderr

    assert not has_image(GOOD)
    # What runs, what `rollback` needs, and an image no deploy has run yet.
    assert has_image(NEWEST)
    assert has_image(NEWER)
    assert has_image(BROKEN)


@pytest.mark.req("FR-15", "NFR-13")
def test_the_rollback_finishes_when_the_client_goes_away(drill: Stack) -> None:
    assert drill.deploy_sh("deploy", GOOD).returncode == 0

    returncode = drill.deploy_sh_and_hang_up("deploy", BROKEN)

    assert returncode == 1
    assert drill.env_tag() == GOOD
    assert drill.running_image() == f"{REPOSITORY}:{GOOD}"
    assert drill.health().status_code == 200
    log = (drill.directory / "deploy-last-run.log").read_text(encoding="utf-8")
    assert log.strip().splitlines()[-1] == f"running: {GOOD}"
