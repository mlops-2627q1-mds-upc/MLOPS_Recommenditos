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
#: and what `docker login` reads on standard input. The images listed in
#: $FAKE_LOCAL_IMAGES are on the VM; a pull of one listed in $FAKE_PULLABLE
#: succeeds, any other fails with $FAKE_PULL_ANSWER; the stack image's files are
#: those of $FAKE_STACK_FILES; `compose up` succeeds. The stack's images are the
#: API image, nginx and one image per repository in $FAKE_RELEASE_REPOSITORIES,
#: all with the tag but nginx. `compose config --quiet` succeeds, or, with
#: $REAL_DOCKER set, is the real Compose's check.
FAKE_DOCKER = """#!/usr/bin/env bash
log=$FAKE_DOCKER_LOG
{ printf 'argv:'; printf ' %s' "$@"; printf '\\n'; env | sed 's/^/env: /'; } >>"$log"
case "$1" in
    compose)
        case "$*" in
            *"config --quiet"*) if [ -n "${REAL_DOCKER:-}" ]; then exec "$REAL_DOCKER" "$@"; fi ;;
            *"config --images api"*) echo "ghcr.io/example/api:$API_IMAGE_TAG" ;;
            *"--profile * config --images"*)
                echo "ghcr.io/example/api:$API_IMAGE_TAG"
                echo "nginxinc/nginx-unprivileged:1.30-alpine-slim@sha256:0123"
                while read -r repository; do
                    echo "$repository:$API_IMAGE_TAG"
                done <"${FAKE_RELEASE_REPOSITORIES:-/dev/null}"
                ;;
        esac
        ;;
    image)
        if [ "$2" = inspect ]; then
            grep -qxF "$3" "$FAKE_LOCAL_IMAGES" || exit 1
        fi
        ;;
    login)
        printf 'stdin: %s\\n' "$(cat)" >>"$log"
        printf 'config exists: %s\\n' "$([ -d "$DOCKER_CONFIG" ] && echo yes)" >>"$log"
        ;;
    pull)
        printf 'pull of %s with DOCKER_CONFIG=%s\\n' "$3" "${DOCKER_CONFIG:-}" >>"$log"
        if grep -qxF "$3" "${FAKE_PULLABLE:-/dev/null}"; then
            echo "$3" >>"$FAKE_LOCAL_IMAGES"
            exit 0
        fi
        echo "Error response from daemon: $FAKE_PULL_ANSWER" >&2
        exit 1
        ;;
    create) echo fake-container ;;
    cp) cp -R "$FAKE_STACK_FILES/." "$3" ;;
esac
"""

OLD = "1" * 40
API_IMAGE = "ghcr.io/example/api"
STACK_IMAGE = "ghcr.io/example/stack"
MONITORING_IMAGE = "ghcr.io/example/monitoring"


class FakeVM:
    """A stack directory as the VM has it, with `docker` and `curl` stand-ins first on PATH."""

    def __init__(self, tmp_path: Path) -> None:
        self.stack = tmp_path / "recommenditos"
        self.stack.mkdir()
        self.env_file = self.stack / ".env"
        self.env_file.write_text(
            f"API_IMAGE={API_IMAGE}\nAPI_IMAGE_TAG={OLD}\nRECOMMENDITOS_API_KEY=keep-me\n",
            encoding="utf-8",
        )
        # The stack files that run now, behind the link deploy.sh keeps.
        running = self.stack / "stacks" / "0123456789abcdef"
        running.mkdir(parents=True)
        (running / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
        (self.stack / "stack").symlink_to("stacks/0123456789abcdef")
        self.files = tmp_path / "stack-image"
        self.files.mkdir()
        (self.files / "compose.yaml").write_text("services: {new: {}}\n", encoding="utf-8")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, script in (("docker", FAKE_DOCKER), ("curl", "#!/bin/sh\nexit 0\n")):
            (bin_dir / name).write_text(script, encoding="utf-8")
            (bin_dir / name).chmod(0o755)
        self.local_images = tmp_path / "local-images"
        self.local_images.write_text(f"{API_IMAGE}:{SHA}\n", encoding="utf-8")
        self.log = tmp_path / "docker.log"
        self.log.touch()
        self.environment = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(self.log),
            "FAKE_LOCAL_IMAGES": str(self.local_images),
            "FAKE_STACK_FILES": str(self.files),
            "FAKE_PULL_ANSWER": "manifest unknown",
            "TMPDIR": str(tmp_path),
        }

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return run_directly(self.stack, list(arguments), extra_environment=self.environment)

    def stack_link(self) -> str:
        return os.readlink(self.stack / "stack")


@pytest.mark.req("NFR-15")
def test_the_registry_token_reaches_docker_login_on_stdin_and_nothing_keeps_it(tmp_path: Path):
    vm = FakeVM(tmp_path)
    vm.local_images.write_text("", encoding="utf-8")
    vm.environment["FAKE_PULL_ANSWER"] = "denied"

    result = vm.run("deploy", SHA, FLAG)

    calls = vm.log.read_text(encoding="utf-8")
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"pulling {STACK_IMAGE}:{SHA} failed" in result.stdout
    assert f"stdin: {SENTINEL}" in calls
    assert "config exists: yes" in calls
    # Only `docker login` saw the token, on its standard input.
    assert calls.count(SENTINEL) == 1
    pull_config = calls.split("with DOCKER_CONFIG=")[1].splitlines()[0]
    assert pull_config.startswith(str(tmp_path))
    assert not Path(pull_config).exists()
    for path in vm.stack.rglob("*"):
        if path.is_file():
            assert SENTINEL not in path.read_text(encoding="utf-8"), path.name
    assert SENTINEL not in result.stdout + result.stderr


@pytest.mark.req("NFR-15")
def test_one_login_serves_the_pulls_of_both_images(tmp_path: Path):
    vm = FakeVM(tmp_path)
    vm.local_images.write_text("", encoding="utf-8")

    result = vm.run("deploy", SHA, FLAG)

    calls = vm.log.read_text(encoding="utf-8")
    assert result.returncode == 1, result.stdout + result.stderr
    assert calls.count("argv: login") == 1
    assert f"pull of {STACK_IMAGE}:{SHA}" in calls
    assert f"pull of {API_IMAGE}:{SHA}" in calls
    assert f"pulling {API_IMAGE}:{SHA} failed; nothing changed" in result.stdout


@pytest.mark.req("NFR-15")
def test_no_login_happens_when_the_throwaway_configuration_cannot_be_made(tmp_path: Path):
    """An empty DOCKER_CONFIG would make docker keep the login in ~/.docker."""
    vm = FakeVM(tmp_path)
    vm.environment["TMPDIR"] = str(tmp_path / "missing")

    result = vm.run("deploy", SHA, FLAG)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not create a Docker configuration" in result.stdout
    assert "argv: login" not in vm.log.read_text(encoding="utf-8")


@pytest.mark.req("NFR-13")
def test_a_tag_without_a_stack_image_runs_with_the_stack_files_that_run_now(tmp_path: Path):
    """CD pushed no stack image before #279: such a tag keeps the files, and says so."""
    vm = FakeVM(tmp_path)

    result = vm.run("deploy", SHA)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{SHA} has no stack image" in result.stdout
    assert vm.stack_link() == "stacks/0123456789abcdef"
    assert vm.env_file.read_text(encoding="utf-8") == (
        f"API_IMAGE={API_IMAGE}\nAPI_IMAGE_TAG={SHA}\nRECOMMENDITOS_API_KEY=keep-me\n"
    )
    assert (vm.stack / ".deploy-previous").read_text(encoding="utf-8") == f"{OLD}\n"


@pytest.mark.req("NFR-13")
def test_a_rollback_to_a_tag_deployed_before_stack_images_keeps_the_stack_files(tmp_path: Path):
    """The old deploy.sh recorded the previous tag, but no stack files with it."""
    vm = FakeVM(tmp_path)
    (vm.stack / ".deploy-previous").write_text(f"{SHA}\n", encoding="utf-8")

    result = vm.run("rollback")

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"no stack files are recorded for {SHA}" in result.stdout
    assert vm.stack_link() == "stacks/0123456789abcdef"
    assert f"API_IMAGE_TAG={SHA}\n" in vm.env_file.read_text(encoding="utf-8")


@pytest.mark.req("NFR-13")
def test_a_stack_image_that_cannot_be_pulled_changes_nothing(tmp_path: Path):
    """Only GHCR's answer for a missing tag means "no stack image"; anything else stops."""
    vm = FakeVM(tmp_path)
    vm.environment["FAKE_PULL_ANSWER"] = (
        "Head https://ghcr.io/v2/example/stack/manifests/x: denied"
    )
    env_before = vm.env_file.read_text(encoding="utf-8")

    result = vm.run("deploy", SHA)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "nothing changed" in result.stdout
    assert vm.env_file.read_text(encoding="utf-8") == env_before
    assert vm.stack_link() == "stacks/0123456789abcdef"
    assert " up " not in vm.log.read_text(encoding="utf-8")


@pytest.mark.req("NFR-13", "NFR-15")
def test_a_stack_image_with_a_dot_env_is_refused_and_env_stays(tmp_path: Path):
    vm = FakeVM(tmp_path)
    vm.local_images.write_text(f"{API_IMAGE}:{SHA}\n{STACK_IMAGE}:{SHA}\n", encoding="utf-8")
    (vm.files / ".env").write_text("RECOMMENDITOS_API_KEY=stolen\n", encoding="utf-8")
    env_before = vm.env_file.read_text(encoding="utf-8")

    result = vm.run("deploy", SHA)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "holds a .env" in result.stdout
    assert vm.env_file.read_text(encoding="utf-8") == env_before
    assert vm.stack_link() == "stacks/0123456789abcdef"


@pytest.mark.req("NFR-13")
def test_the_stack_files_of_a_release_are_swapped_in_beside_env(tmp_path: Path):
    vm = FakeVM(tmp_path)
    vm.local_images.write_text(f"{API_IMAGE}:{SHA}\n{STACK_IMAGE}:{SHA}\n", encoding="utf-8")

    result = vm.run("deploy", SHA)

    assert result.returncode == 0, result.stdout + result.stderr
    link = vm.stack_link()
    assert link.startswith("stacks/") and link != "stacks/0123456789abcdef"
    assert (vm.stack / link / "compose.yaml").read_text(encoding="utf-8") == (
        "services: {new: {}}\n"
    )
    # The previous files stay for `rollback`, named with the previous tag.
    assert (vm.stack / ".deploy-previous-stack").read_text(encoding="utf-8") == (
        "0123456789abcdef\n"
    )
    assert (vm.stack / "stacks" / "0123456789abcdef" / "compose.yaml").exists()
    assert "--remove-orphans" in vm.log.read_text(encoding="utf-8")


def test_the_repository_deploy_folder_is_not_a_stack_directory(tmp_path: Path):
    """Run from deploy/ itself, the script would take its compose.yaml for copied files."""
    before = sorted(path.name for path in (PROJ_ROOT / "deploy").iterdir())

    result = run_directly(PROJ_ROOT / "deploy", ["status"])

    assert result.returncode == 1, result.stdout + result.stderr
    assert "the repository's deploy/" in result.stdout
    assert sorted(path.name for path in (PROJ_ROOT / "deploy").iterdir()) == before


@pytest.mark.req("NFR-13")
@pytest.mark.parametrize(
    ("answer", "kept"),
    [
        ("manifest unknown", True),
        ("manifest for ghcr.io/example/stack:x not found: manifest unknown", True),
        ("failed to resolve reference ghcr.io/example/stack:x: not found", False),
        ("Head https://ghcr.io/v2/example/stack/manifests/x: unauthorized", False),
    ],
    ids=["GHCR", "Docker Hub wording", "another not found", "unauthorized"],
)
def test_only_manifest_unknown_counts_as_a_tag_without_a_stack_image(
    tmp_path: Path, answer: str, kept: bool
):
    vm = FakeVM(tmp_path)
    vm.environment["FAKE_PULL_ANSWER"] = answer

    result = vm.run("deploy", SHA)

    assert (result.returncode == 0) is kept, result.stdout + result.stderr
    assert (f"{SHA} has no stack image" in result.stdout) is kept


@pytest.mark.req("NFR-13")
def test_a_stack_image_with_a_link_is_refused(tmp_path: Path):
    """A link could point a bind mount anywhere on the VM."""
    vm = FakeVM(tmp_path)
    vm.local_images.write_text(f"{API_IMAGE}:{SHA}\n{STACK_IMAGE}:{SHA}\n", encoding="utf-8")
    (vm.files / "nginx.conf").symlink_to("/etc/shadow")
    env_before = vm.env_file.read_text(encoding="utf-8")

    result = vm.run("deploy", SHA)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "such as a link" in result.stdout
    assert vm.env_file.read_text(encoding="utf-8") == env_before
    assert vm.stack_link() == "stacks/0123456789abcdef"


def with_monitoring_image(vm: FakeVM, tmp_path: Path) -> None:
    """A release whose compose.yaml names the monitoring image too, and images elsewhere.

    Two of those elsewhere share the namespace's prefix without being in it, so a
    filter that matches the prefix instead of the namespace pulls them too.
    """
    repositories = tmp_path / "release-repositories"
    repositories.write_text(
        f"{MONITORING_IMAGE}\nghcr.io/elsewhere/tool\nghcr.io/example-evil/tool\n"
        "ghcr.io/examples/monitoring\nrecommenditos-monitoring\n",
        encoding="utf-8",
    )
    vm.environment["FAKE_RELEASE_REPOSITORIES"] = str(repositories)
    vm.local_images.write_text(f"{API_IMAGE}:{SHA}\n{STACK_IMAGE}:{SHA}\n", encoding="utf-8")


@pytest.mark.req("NFR-13", "NFR-15")
def test_every_image_of_the_release_in_its_namespace_is_pulled_with_the_one_login(
    tmp_path: Path,
):
    """A service behind a profile, such as the replay, is one `up` would never pull."""
    vm = FakeVM(tmp_path)
    with_monitoring_image(vm, tmp_path)
    pullable = tmp_path / "pullable"
    pullable.write_text(f"{MONITORING_IMAGE}:{SHA}\n", encoding="utf-8")
    vm.environment["FAKE_PULLABLE"] = str(pullable)

    result = vm.run("deploy", SHA, FLAG)

    calls = vm.log.read_text(encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.count("argv: login") == 1
    pulled = [line.split()[2] for line in calls.splitlines() if line.startswith("pull of ")]
    assert pulled == [f"{MONITORING_IMAGE}:{SHA}"]
    assert f"pull of {MONITORING_IMAGE}:{SHA} with DOCKER_CONFIG={tmp_path}" in calls
    assert "--profile * config --images" in calls
    assert f"API_IMAGE_TAG={SHA}\n" in vm.env_file.read_text(encoding="utf-8")


@pytest.mark.req("NFR-13")
def test_a_tag_without_a_stack_image_pulls_only_its_api_image(tmp_path: Path):
    """The stack files that run now name a monitoring image such a tag never had."""
    vm = FakeVM(tmp_path)
    with_monitoring_image(vm, tmp_path)
    vm.local_images.write_text("", encoding="utf-8")
    pullable = tmp_path / "pullable"
    pullable.write_text(f"{API_IMAGE}:{SHA}\n", encoding="utf-8")
    vm.environment["FAKE_PULLABLE"] = str(pullable)

    result = vm.run("deploy", SHA, FLAG)

    calls = vm.log.read_text(encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{SHA} has no stack image" in result.stdout
    pulled = [line.split()[2] for line in calls.splitlines() if line.startswith("pull of ")]
    assert pulled == [f"{STACK_IMAGE}:{SHA}", f"{API_IMAGE}:{SHA}"]
    assert f"API_IMAGE_TAG={SHA}\n" in vm.env_file.read_text(encoding="utf-8")


@pytest.mark.req("NFR-13")
def test_a_release_image_that_cannot_be_pulled_changes_nothing(tmp_path: Path):
    vm = FakeVM(tmp_path)
    with_monitoring_image(vm, tmp_path)
    vm.environment["FAKE_PULL_ANSWER"] = "denied"
    env_before = vm.env_file.read_text(encoding="utf-8")

    result = vm.run("deploy", SHA, FLAG)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"pulling {MONITORING_IMAGE}:{SHA} failed; nothing changed" in result.stdout
    assert vm.env_file.read_text(encoding="utf-8") == env_before
    assert vm.stack_link() == "stacks/0123456789abcdef"
    assert " up " not in vm.log.read_text(encoding="utf-8")


@pytest.mark.req("NFR-04")
def test_a_deploy_removes_the_old_tags_of_every_release_image(tmp_path: Path):
    """Two deploys back goes, the previous tag stays, as for the API image."""
    vm = FakeVM(tmp_path)
    with_monitoring_image(vm, tmp_path)
    older = "2" * 40
    with vm.local_images.open("a", encoding="utf-8") as local:
        for tag in (OLD, older):
            local.write(f"{API_IMAGE}:{tag}\n{MONITORING_IMAGE}:{tag}\n")
        local.write(f"{MONITORING_IMAGE}:{SHA}\n")
    (vm.stack / "deploy-history.log").write_text(
        f"2026-11-01T00:00:00Z deploy {older} ok\n2026-11-02T00:00:00Z deploy {OLD} ok\n",
        encoding="utf-8",
    )

    result = vm.run("deploy", SHA)

    calls = vm.log.read_text(encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"argv: image rm {API_IMAGE}:{older}" in calls
    assert f"argv: image rm {MONITORING_IMAGE}:{older}" in calls
    assert f"argv: image rm {MONITORING_IMAGE}:{OLD}" not in calls
    assert f"argv: image rm {MONITORING_IMAGE}:{SHA}" not in calls


def compose_available() -> bool:
    """Whether the real `docker compose` runs here; `config` needs no daemon.

    Asked only where the module runs at all: on a Windows runner the call hangs.
    """
    docker = shutil.which("docker")
    if sys.platform == "win32" or docker is None:
        return False
    try:
        version = subprocess.run(
            [docker, "compose", "version"], capture_output=True, timeout=30, check=False
        )
    except subprocess.TimeoutExpired:
        return False
    return version.returncode == 0


#: Stand in for secret values in `.env`, such as the API key; the second is
#: one word, as Compose prints the name of a `$NAME` it finds inside a value.
HIDDEN = "stand-in-for-a-value-of-dot-env"
HIDDEN_WORD = "standInForAValueOfDotEnv"
#: A release's compose.yaml that requires a variable, as ours requires the
#: Grafana password, with the API key in the environment and the host port in a
#: field Compose checks.
REQUIRING_COMPOSE = """services:
  api:
    image: busybox:1
    ports: ["${HTTP_PORT:-8080}:8080"]
    environment:
      RECOMMENDITOS_API_KEY: ${RECOMMENDITOS_API_KEY:-}
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?set it in .env}
"""
REJECTED_ENV = {
    "a missing variable": (
        f"RECOMMENDITOS_API_KEY={HIDDEN}\n",
        "required variable GRAFANA_ADMIN_PASSWORD is missing a value: set it in .env",
    ),
    "an unterminated quote": (
        f'RECOMMENDITOS_API_KEY="{HIDDEN}\nGRAFANA_ADMIN_PASSWORD=x{HIDDEN}\n',
        "unterminated quoted value <RECOMMENDITOS_API_KEY>",
    ),
    "an invalid template": (
        f"RECOMMENDITOS_API_KEY={HIDDEN}${{x\nGRAFANA_ADMIN_PASSWORD=x{HIDDEN}\n",
        'Invalid template: "<RECOMMENDITOS_API_KEY>"',
    ),
    "a value in a checked field": (
        f"HTTP_PORT={HIDDEN}\nGRAFANA_ADMIN_PASSWORD=x{HIDDEN}\n",
        "invalid hostPort: <HTTP_PORT>",
    ),
    # Compose prints an invalid template Go-quoted, with `\\` and `\"` escaped.
    "an invalid template with a backslash and a quote": (
        f'RECOMMENDITOS_API_KEY={HIDDEN}\\"{HIDDEN}${{x\nGRAFANA_ADMIN_PASSWORD=x{HIDDEN}\n',
        'Invalid template: "<RECOMMENDITOS_API_KEY>"',
    ),
    # Compose warns that the `$NAME` inside the value is not set.
    "a dollar inside a value": (
        f"GRAFANA_ADMIN_PASSWORD=Xy${HIDDEN_WORD}\nHTTP_PORT=notaport\n",
        "invalid hostPort: <HTTP_PORT>",
    ),
    # A tab in a line of `.env`, after `export`.
    "an unterminated quote after export and a tab": (
        f'export\tRECOMMENDITOS_API_KEY="{HIDDEN}\nGRAFANA_ADMIN_PASSWORD=x{HIDDEN}\n',
        "unterminated quoted value <RECOMMENDITOS_API_KEY>",
    ),
}


def windows_of(secret: str, output: str) -> list[str]:
    """The pieces of 8 characters of `secret` that `output` holds."""
    return [secret[i : i + 8] for i in range(len(secret) - 7) if secret[i : i + 8] in output]


@pytest.mark.req("NFR-13", "NFR-15")
@pytest.mark.skipif(not compose_available(), reason="needs the real docker compose")
@pytest.mark.parametrize(("lines", "named"), REJECTED_ENV.values(), ids=REJECTED_ENV.keys())
def test_a_rejected_configuration_is_named_without_printing_a_value_of_env(
    tmp_path: Path, lines: str, named: str
):
    """The deploy log is public: Compose's error is shown, the values of .env are not."""
    vm = FakeVM(tmp_path)
    vm.local_images.write_text(f"{API_IMAGE}:{SHA}\n{STACK_IMAGE}:{SHA}\n", encoding="utf-8")
    (vm.files / "compose.yaml").write_text(REQUIRING_COMPOSE, encoding="utf-8")
    vm.env_file.write_text(
        f"API_IMAGE={API_IMAGE}\nAPI_IMAGE_TAG={OLD}\n{lines}", encoding="utf-8"
    )
    vm.environment["REAL_DOCKER"] = str(shutil.which("docker"))
    env_before = vm.env_file.read_text(encoding="utf-8")

    result = vm.run("deploy", SHA)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Compose does not accept the compose.yaml" in result.stdout
    for secret in (HIDDEN, HIDDEN_WORD):
        assert windows_of(secret, result.stdout + result.stderr) == [], result.stdout
    assert named in result.stdout, result.stdout
    assert vm.env_file.read_text(encoding="utf-8") == env_before
    assert vm.stack_link() == "stacks/0123456789abcdef"
    assert " up " not in vm.log.read_text(encoding="utf-8")


@pytest.mark.req("NFR-13")
def test_a_stack_image_without_a_compose_yaml_is_refused(tmp_path: Path):
    vm = FakeVM(tmp_path)
    vm.local_images.write_text(f"{API_IMAGE}:{SHA}\n{STACK_IMAGE}:{SHA}\n", encoding="utf-8")
    (vm.files / "compose.yaml").unlink()

    result = vm.run("deploy", SHA)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "holds no compose.yaml" in result.stdout
    assert "nothing changed" in result.stdout
    assert vm.stack_link() == "stacks/0123456789abcdef"
