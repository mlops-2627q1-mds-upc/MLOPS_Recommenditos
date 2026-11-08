"""The rollback drill of FR-15, with deploy/deploy.sh against a stack on this machine.

deploy.sh is what the Deploy workflow runs on the VM (docs/docs/deployment.md#every-deploy).
Here it runs on a copy of the VM's stack directory, `.env` and the script, as a
second Compose project beside the stack under test, on a port of its own. A
release is two images under one tag, as CD pushes them: the API image and the
stack image built from this checkout's deploy/ with deploy/stack.Dockerfile,
from which deploy.sh copies compose.yaml and nginx.conf. The good API image is
the one the stack under test runs; the broken one is that image with a server
that exits as soon as it starts, like an image whose bundle does not load. The
images get a repository of their own, so deploy.sh's clean-up of old tags never
reaches the stack's images.
"""

from collections.abc import Callable, Iterator
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tarfile
import time

import httpx2
import pytest
from tests.integration.conftest import COMPOSE_PROJECT, Container, docker
import yaml

from recommenditos.config import PROJ_ROOT

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: Named after the stack under test, so the drills of two stacks never share a
#: project or tear down each other's images.
PROJECT = REPOSITORY = f"{COMPOSE_PROJECT}-drill"
#: The two repositories of a release, named as CD names them (…/api and …/stack).
API, STACK = f"{REPOSITORY}/api", f"{REPOSITORY}/stack"
#: The monitoring image of a release, which the always-on drift job runs (…/monitoring).
MONITORING = f"{REPOSITORY}/monitoring"
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
TAGS = (GOOD, NEWER, NEWEST, BROKEN)
#: The files of deploy/ that never travel with a release (deploy/.dockerignore).
NEVER_TRAVEL = {".dockerignore", "stack.Dockerfile", ".env.template"}
#: A path a changed nginx.conf answers, which the API itself does not have.
MARKER_PATH = "/drill-stack"

#: The services of the stack files that were copied by hand before stack images
#: existed (#279): those files were compose.yaml and nginx.conf alone, without
#: the configuration folders that later services such as Prometheus mount.
BY_HAND_SERVICES = ("api", "proxy")
#: A service a newer compose.yaml adds, with a named volume of its own.
EXTRA_SERVICE = """\
  extra:
    image: ${API_IMAGE}:${API_IMAGE_TAG}
    entrypoint: ["python", "-c", "import time; time.sleep(3600)"]
    init: true
    # Its own probe instead of the image's, which asks a server this service
    # does not run. Not `disable: true`: `up --wait` of the Compose v2 on CI's
    # runners fails on such a service ("has no healthcheck configured"), where
    # Compose 5 waits for it to run.
    healthcheck:
      test: ["CMD", "python", "-c", "pass"]
      interval: 5s
      start_period: 30s
      start_interval: 1s
    volumes:
      - extra-data:/data
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def without_stack_environment() -> dict[str, str]:
    """This process's environment without what CI sets for the stack under test.

    Compose would prefer those variables over the copy's .env.
    """
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(
            ("API_IMAGE", "COMPOSE_", "RECOMMENDITOS_", "HTTP_PORT", "GRAFANA_", "DISCORD_")
        )
    }


def containers_report() -> str:
    """The state of each container of the drill's project, with the log of each that is unwell.

    deploy.sh answers a failed start with "check `docker compose ps` and `... logs`",
    and the fixture removes the containers as soon as the test ends, so a failure
    message has to carry both. `up --wait` names a container "unhealthy" both when
    its probes failed and when its process exited and Docker restarts it: the
    restart count, the end of the last run, the probes, and the log tell the two
    apart. The report is built inside a failing assertion's message, so it never
    raises: whatever goes wrong in it becomes its text, and the assertion's own
    message stays.
    """
    try:
        return _containers_report()
    except Exception as error:  # noqa: BLE001 - a message must not replace the failure it explains
        return f"no report of the containers of {PROJECT}: {type(error).__name__}: {error}"


def _containers_report() -> str:
    listed = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={PROJECT}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if listed.returncode != 0:
        return f"docker ps failed: {listed.stderr.strip()}"
    ids = listed.stdout.split()
    if not ids:
        return f"no container of {PROJECT} exists"
    inspected = subprocess.run(
        ["docker", "inspect", *ids], capture_output=True, text=True, timeout=60, check=False
    )
    if inspected.returncode != 0:
        return f"docker inspect failed: {inspected.stderr.strip()}"
    lines = [f"the containers of {PROJECT}:"]
    for container in json.loads(inspected.stdout):
        state = container["State"]
        health = state.get("Health") or {}
        service = container["Config"]["Labels"].get(
            "com.docker.compose.service", container["Name"]
        )
        # Docker sets the exit code back to 0 when it restarts a container; the
        # end of the last run and its error stay.
        lines.append(
            f"{service}: {state['Status']}, health {health.get('Status', 'none')}, "
            f"exit code {state['ExitCode']}, error {state['Error'] or 'none'}, "
            f"OOM-killed {state['OOMKilled']}, restarts {container['RestartCount']}, "
            f"started {state['StartedAt']}, last run ended {state['FinishedAt']}"
        )
        well = health.get("Status", "healthy") == "healthy" and state["Status"] == "running"
        done = state["Status"] == "exited" and state["ExitCode"] == 0
        if (well or done) and container["RestartCount"] == 0:
            continue
        for probe in health.get("Log", [])[-3:]:
            output = " ".join(probe["Output"].split())[:300]
            lines.append(f"  probe at {probe['Start']}: exit code {probe['ExitCode']} {output}")
        log = subprocess.run(
            ["docker", "logs", "--tail", "40", container["Id"]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        lines.extend(f"  | {line}" for line in (log.stdout + log.stderr).splitlines())
    return "\n".join(lines)


class Stack:
    """A copy of the VM's stack directory, driven with deploy.sh as the workflow does."""

    def __init__(self, directory: Path, port: int, build_root: Path) -> None:
        self.directory = directory
        self.port = port
        self.build_root = build_root

    def build_stack(
        self, tag: str, marker: str | None = None, extra_service: bool = False
    ) -> None:
        """Build the stack image of `tag` from this checkout's deploy/, changed as asked.

        With `marker`, nginx.conf answers MARKER_PATH with it; with
        `extra_service`, compose.yaml has the service `extra` and its volume.
        """
        context = self.build_root / tag
        shutil.rmtree(context, ignore_errors=True)
        shutil.copytree(PROJ_ROOT / "deploy", context)
        if marker is not None:
            nginx = context / "nginx.conf"
            text = nginx.read_text(encoding="utf-8").rstrip()
            assert text.endswith("}"), "nginx.conf no longer ends with its server block"
            location = f'    location = {MARKER_PATH} {{ return 200 "{marker}"; }}\n}}\n'
            nginx.write_text(text[:-1] + location, encoding="utf-8")
        if extra_service:
            compose = context / "compose.yaml"
            text = compose.read_text(encoding="utf-8")
            # Top-level keys start a line; the service and its volume go first
            # under theirs, wherever they are and whatever else they hold.
            assert "\nservices:\n" in text, "compose.yaml has no top-level services"
            text = text.replace("\nservices:\n", "\nservices:\n" + EXTRA_SERVICE, 1)
            if "\nvolumes:\n" in text:
                text = text.replace("\nvolumes:\n", "\nvolumes:\n  extra-data:\n", 1)
            else:
                text += "\nvolumes:\n  extra-data:\n"
            compose.write_text(text, encoding="utf-8")
        subprocess.run(
            [
                "docker",
                "build",
                "--quiet",
                "--file",
                str(context / "stack.Dockerfile"),
                "--tag",
                f"{STACK}:{tag}",
                str(context),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

    def deploy_sh(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = without_stack_environment()
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

        That is what a cancelled workflow run or a closed terminal does to it.
        """
        environment = without_stack_environment()
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

    def explain(self, result: subprocess.CompletedProcess[str]) -> str:
        """What deploy.sh said, then the state of the drill's containers right after."""
        return f"{result.stdout}{result.stderr}\n{containers_report()}"

    def env_tag(self) -> str:
        lines = (self.directory / ".env").read_text(encoding="utf-8").splitlines()
        return [line.split("=", 1)[1] for line in lines if line.startswith("API_IMAGE_TAG=")][-1]

    def stack_files(self) -> Path:
        """The directory the link `stack` points at, which Compose runs with."""
        return (self.directory / "stack").resolve()

    def containers(self, service: str, everything: bool = False) -> list[str]:
        return docker(
            "ps",
            *(["--all"] if everything else []),
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={PROJECT}",
            "--filter",
            f"label=com.docker.compose.service={service}",
        ).split()

    def running_image(self) -> str:
        ids = self.containers("api")
        assert len(ids) == 1, f"expected one api container of {PROJECT}, found {len(ids)}"
        return docker("inspect", "--format", "{{.Config.Image}}", ids[0])

    def api(self) -> Container:
        ids = self.containers("api")
        assert len(ids) == 1, f"expected one api container of {PROJECT}, found {len(ids)}"
        return Container(ids[0])

    def health(self) -> httpx2.Response:
        return httpx2.get(f"http://127.0.0.1:{self.port}/health", timeout=10)

    def marker(self) -> httpx2.Response:
        return httpx2.get(f"http://127.0.0.1:{self.port}{MARKER_PATH}", timeout=10)


def travelling_files() -> set[str]:
    """The tracked files of deploy/ that a release carries, relative to deploy/."""
    tracked = subprocess.run(
        ["git", "ls-files", "deploy"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.split()
    relative = [path.removeprefix("deploy/") for path in tracked]
    # Configuration folders travel whole; at the top, the scripts and NEVER_TRAVEL stay.
    return {
        path
        for path in relative
        if "/" in path or not (path.endswith(".sh") or path in NEVER_TRAVEL)
    }


def project_volumes() -> set[str]:
    return set(
        docker(
            "volume", "ls", "--quiet", "--filter", f"label=com.docker.compose.project={PROJECT}"
        ).split()
    )


@pytest.fixture
def drill(
    tmp_path: Path, api_container: Container, stack_container: Callable[[str], Container]
) -> Iterator[Stack]:
    """The VM's stack directory, three good releases of the image under test and a broken one."""
    # The image ID, not its tag: another checkout's `make stack` may have moved the tag.
    good = api_container.inspect()["Image"]
    for tag in (GOOD, NEWER, NEWEST):
        docker("tag", good, f"{API}:{tag}")
    # Every release has its monitoring image too, as CD pushes it: the drift job
    # runs in every stack. The one the stack under test runs.
    monitoring = stack_container("monitor").inspect()["Image"]
    for tag in TAGS:
        docker("tag", monitoring, f"{MONITORING}:{tag}")
    subprocess.run(
        ["docker", "build", "--quiet", "--tag", f"{API}:{BROKEN}", "-"],
        input=BROKEN_DOCKERFILE.format(base=f"{API}:{GOOD}"),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    # What bootstrap.sh installs: deploy.sh and .env; the stack files come with
    # the first release.
    directory = tmp_path / "recommenditos"
    directory.mkdir()
    shutil.copy(PROJ_ROOT / "deploy" / "deploy.sh", directory / "deploy.sh")
    port = free_port()
    # Grafana on a free port too, beside the stack under test on 3000; a
    # throwaway admin password, and a webhook that goes nowhere.
    (directory / ".env").write_text(
        f"COMPOSE_PROJECT_NAME={PROJECT}\nAPI_IMAGE={API}\nAPI_IMAGE_TAG=\nHTTP_PORT={port}\n"
        f"GRAFANA_PORT={free_port()}\nGRAFANA_ADMIN_PASSWORD={secrets.token_urlsafe(24)}\n"
        "DISCORD_WEBHOOK_URL=http://127.0.0.1:9/discarded\n"
        f"MONITORING_IMAGE={MONITORING}\n",
        encoding="utf-8",
    )
    stack = Stack(directory, port, tmp_path / "stack-images")
    try:
        for tag in TAGS:
            stack.build_stack(tag)
        yield stack
    finally:
        # Without a compose file: the project's containers carry its name.
        subprocess.run(
            ["docker", "compose", "--project-name", PROJECT, "down", "--remove-orphans"],
            capture_output=True,
            timeout=120,
            check=False,
        )
        for volume in project_volumes():
            subprocess.run(
                ["docker", "volume", "rm", volume], capture_output=True, timeout=60, check=False
            )
        for repository in (API, STACK, MONITORING):
            for tag in TAGS:
                subprocess.run(
                    ["docker", "image", "rm", f"{repository}:{tag}"],
                    capture_output=True,
                    timeout=60,
                    check=False,
                )


def last_line(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.req("FR-15", "NFR-13")
def test_a_broken_tag_is_rolled_back_and_the_previous_one_serves_again(drill: Stack) -> None:
    first = drill.deploy_sh("deploy", GOOD)
    assert first.returncode == 0, drill.explain(first)
    assert last_line(first) == f"running: {GOOD}"
    assert drill.health().status_code == 200

    broken = drill.deploy_sh("deploy", BROKEN)

    assert broken.returncode == 1, drill.explain(broken)
    assert last_line(broken) == f"running: {GOOD}"
    assert drill.env_tag() == GOOD
    assert drill.running_image() == f"{API}:{GOOD}"
    assert drill.health().status_code == 200
    history = (drill.directory / "deploy-history.log").read_text(encoding="utf-8")
    assert f"rollback {BROKEN} -> {GOOD} ok" in history


@pytest.mark.req("NFR-13")
def test_a_deploy_keeps_the_log_of_the_api_container_it_replaces(drill: Stack) -> None:
    """compose.yaml logs to the journal, which outlives a recreated container (#313)."""
    if os.environ.get("LOG_DRIVER", "journald") != "journald":
        pytest.skip("LOG_DRIVER sets another log driver than the VM's journald")
    first = drill.deploy_sh("deploy", GOOD)
    assert first.returncode == 0, drill.explain(first)
    (replaced,) = drill.containers("api")
    before = Container(replaced).log_lines()

    newer = drill.deploy_sh("deploy", NEWER)

    assert newer.returncode == 0, drill.explain(newer)
    assert drill.containers("api") != [replaced]
    assert replaced not in drill.containers("api", everything=True)
    journal = subprocess.run(
        ["journalctl", "--no-pager", "-o", "cat", f"CONTAINER_ID={replaced}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert before, "the API logged nothing before the deploy"
    lines = journal.stdout.splitlines()
    assert lines, f"no journal entries of the replaced container: {journal.stderr.strip()}"
    assert set(before) <= set(lines)


@pytest.mark.req("FR-15", "NFR-13")
def test_rollback_after_a_failed_smoke_check_brings_the_previous_release_back(
    drill: Stack,
) -> None:
    good = drill.deploy_sh("deploy", GOOD)
    assert good.returncode == 0, drill.explain(good)
    good_files = drill.stack_files()
    drill.build_stack(NEWER, marker="newer")
    newer = drill.deploy_sh("deploy", NEWER)
    assert newer.returncode == 0, drill.explain(newer)
    assert drill.running_image() == f"{API}:{NEWER}"
    assert drill.marker().text == "newer"

    # What the workflow runs when `make smoke` fails against the new release.
    rollback = drill.deploy_sh("rollback")

    assert rollback.returncode == 0, drill.explain(rollback)
    assert last_line(rollback) == f"running: {GOOD}"
    assert drill.running_image() == f"{API}:{GOOD}"
    assert drill.stack_files() == good_files
    assert drill.marker().status_code == 404
    assert drill.health().status_code == 200
    # One step back only: a second rollback would bring the failed release back.
    again = drill.deploy_sh("rollback")
    assert again.returncode == 1, drill.explain(again)
    assert last_line(again) == f"running: {GOOD}"


@pytest.mark.req("NFR-13")
def test_a_change_to_the_stack_files_reaches_the_vm_and_leaves_env_as_it_is(
    drill: Stack,
) -> None:
    good = drill.deploy_sh("deploy", GOOD)
    assert good.returncode == 0, drill.explain(good)
    assert drill.marker().status_code == 404
    env_before = (drill.directory / ".env").read_text(encoding="utf-8")
    drill.build_stack(NEWER, marker="changed nginx.conf")

    result = drill.deploy_sh("deploy", NEWER)

    assert result.returncode == 0, drill.explain(result)
    assert drill.marker().text == "changed nginx.conf"
    # Only the tag changed in .env, and no stack file came beside it.
    env_after = (drill.directory / ".env").read_text(encoding="utf-8")
    assert env_after == env_before.replace(f"API_IMAGE_TAG={GOOD}\n", f"API_IMAGE_TAG={NEWER}\n")
    files = {
        path.relative_to(drill.stack_files()).as_posix()
        for path in drill.stack_files().rglob("*")
        if path.is_file()
    }
    assert files == travelling_files()
    assert not (drill.directory / "compose.yaml").exists()


@pytest.mark.req("FR-15", "NFR-13")
def test_a_failed_release_brings_back_the_previous_tag_and_stack_files_together(
    drill: Stack,
) -> None:
    good = drill.deploy_sh("deploy", GOOD)
    assert good.returncode == 0, drill.explain(good)
    good_files = drill.stack_files()
    drill.build_stack(BROKEN, marker="broken")

    broken = drill.deploy_sh("deploy", BROKEN)

    assert broken.returncode == 1, drill.explain(broken)
    assert drill.env_tag() == GOOD
    assert drill.stack_files() == good_files
    assert drill.running_image() == f"{API}:{GOOD}"
    assert drill.marker().status_code == 404
    assert drill.health().status_code == 200


@pytest.mark.req("FR-15", "NFR-13")
def test_rolling_back_to_a_release_without_a_service_removes_it_and_keeps_the_volumes(
    drill: Stack,
) -> None:
    good = drill.deploy_sh("deploy", GOOD)
    assert good.returncode == 0, drill.explain(good)
    drill.api().exec("python", "-c", "open('/var/lib/recommenditos/drill-marker', 'w').close()")
    drill.build_stack(NEWER, extra_service=True)
    newer = drill.deploy_sh("deploy", NEWER)
    assert newer.returncode == 0, drill.explain(newer)
    assert len(drill.containers("extra")) == 1
    volumes = project_volumes()
    assert {f"{PROJECT}_prediction-log", f"{PROJECT}_extra-data"} <= volumes

    rollback = drill.deploy_sh("rollback")

    assert rollback.returncode == 0, drill.explain(rollback)
    # GOOD's compose.yaml has no `extra`, so its container is gone (--remove-orphans) ...
    assert drill.containers("extra", everything=True) == []
    # ... and no volume is: Compose removes one only when asked to.
    assert project_volumes() == volumes
    drill.api().exec("test", "-f", "/var/lib/recommenditos/drill-marker")
    assert drill.health().status_code == 200


@pytest.mark.req("NFR-13")
def test_the_first_release_takes_over_the_stack_files_that_were_copied_by_hand(
    drill: Stack,
) -> None:
    """The VM before stack images: the files beside .env, started by the old deploy.sh."""
    compose = yaml.safe_load((PROJ_ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8"))
    services = {name: compose["services"][name] for name in BY_HAND_SERVICES}
    mounted = {
        str(mount).split(":", 1)[0]
        for service in services.values()
        for mount in service.get("volumes", [])
    }
    compose["services"] = services
    compose["volumes"] = {
        name: config for name, config in compose["volumes"].items() if name in mounted
    }
    (drill.directory / "compose.yaml").write_text(
        yaml.safe_dump(compose, sort_keys=False), encoding="utf-8"
    )
    shutil.copy(PROJ_ROOT / "deploy" / "nginx.conf", drill.directory / "nginx.conf")
    nginx = drill.directory / "nginx.conf"
    text = nginx.read_text(encoding="utf-8").rstrip()
    nginx.write_text(
        text[:-1] + f'    location = {MARKER_PATH} {{ return 200 "by hand"; }}\n}}\n',
        encoding="utf-8",
    )
    env_file = drill.directory / ".env"
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            "API_IMAGE_TAG=\n", f"API_IMAGE_TAG={GOOD}\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(drill.directory),
            "up",
            "-d",
            "--no-build",
            "--wait",
        ],
        capture_output=True,
        env=without_stack_environment(),
        timeout=WAIT_TIMEOUT_S * 3,
        check=True,
    )
    assert drill.marker().text == "by hand"

    # The first deploy with the new deploy.sh, of the tag that runs.
    result = drill.deploy_sh("deploy", GOOD)

    assert result.returncode == 0, drill.explain(result)
    assert drill.marker().status_code == 404
    assert not (drill.directory / "compose.yaml").exists()
    assert not (drill.directory / "nginx.conf").exists()
    # The files copied by hand are the previous release.
    rollback = drill.deploy_sh("rollback")
    assert rollback.returncode == 0, drill.explain(rollback)
    assert drill.marker().text == "by hand"
    assert drill.env_tag() == GOOD


def has_image(reference: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", reference],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.req("NFR-04")
def test_a_deploy_removes_the_release_two_deploys_back_and_keeps_the_rest(drill: Stack) -> None:
    drill.build_stack(NEWER, marker="newer")
    drill.build_stack(NEWEST, marker="newest")
    for tag in (GOOD, NEWER, NEWEST):
        result = drill.deploy_sh("deploy", tag)
        assert result.returncode == 0, drill.explain(result)

    assert not has_image(f"{API}:{GOOD}")
    assert not has_image(f"{STACK}:{GOOD}")
    # What runs, what `rollback` needs, and an image no deploy has run yet.
    for tag in (NEWEST, NEWER, BROKEN):
        assert has_image(f"{API}:{tag}")
        assert has_image(f"{STACK}:{tag}")
    # The stack files of NEWEST and NEWER, which `rollback` needs; GOOD's are gone.
    assert len([path for path in (drill.directory / "stacks").iterdir()]) == 2


@pytest.mark.req("FR-15", "NFR-13")
def test_the_rollback_finishes_when_the_client_goes_away(drill: Stack) -> None:
    good = drill.deploy_sh("deploy", GOOD)
    assert good.returncode == 0, drill.explain(good)

    returncode = drill.deploy_sh_and_hang_up("deploy", BROKEN)

    assert returncode == 1, containers_report()
    assert drill.env_tag() == GOOD
    assert drill.running_image() == f"{API}:{GOOD}"
    assert drill.health().status_code == 200
    # tee may still be writing the log when bash has exited.
    log = drill.directory / "deploy-last-run.log"
    deadline = time.monotonic() + 10
    while not log.read_text(encoding="utf-8").endswith(f"running: {GOOD}\n"):
        assert time.monotonic() < deadline, log.read_text(encoding="utf-8")
        time.sleep(0.1)


@pytest.mark.req("NFR-13", "NFR-15")
def test_the_stack_image_holds_the_stack_files_and_nothing_else(drill: Stack) -> None:
    """No secret, template or script of the bootstrap layer travels with a release."""
    expected = travelling_files()
    container = docker("create", f"{STACK}:{GOOD}", "stack-files-only")
    try:
        listing = subprocess.run(
            ["docker", "export", container], capture_output=True, timeout=60, check=True
        ).stdout
    finally:
        docker("rm", container)
    with tarfile.open(fileobj=io.BytesIO(listing)) as archive:
        files = {
            member.name.removeprefix("stack/")
            for member in archive.getmembers()
            if member.isfile() and member.name.startswith("stack/")
        }
    assert files == expected
    assert "compose.yaml" in files and "nginx.conf" in files


@pytest.mark.req("NFR-13")
def test_a_deploy_needs_the_monitoring_image_of_the_release_too(drill: Stack) -> None:
    """The monitoring image of the drift job and the replay is the release's, like the API's.

    deploy.sh lists the images of every profile with this machine's Compose and
    pulls the release's own; without the monitoring image the new release could
    not start its drift job, so nothing changes.
    """
    first = drill.deploy_sh("deploy", GOOD)
    assert first.returncode == 0, drill.explain(first)
    monitoring = docker("image", "inspect", "--format", "{{.Id}}", f"{MONITORING}:{NEWER}")
    docker("image", "rm", f"{MONITORING}:{NEWER}")
    try:
        missing = drill.deploy_sh("deploy", NEWER)

        assert missing.returncode == 1, drill.explain(missing)
        assert f"pulling {MONITORING}:{NEWER} failed; nothing changed" in missing.stdout
        assert drill.env_tag() == GOOD
        assert drill.health().status_code == 200
    finally:
        docker("tag", monitoring, f"{MONITORING}:{NEWER}")

    present = drill.deploy_sh("deploy", NEWER)

    assert present.returncode == 0, drill.explain(present)
    assert f"{MONITORING}:{NEWER} is on the VM already" in present.stdout
    assert drill.env_tag() == NEWER
