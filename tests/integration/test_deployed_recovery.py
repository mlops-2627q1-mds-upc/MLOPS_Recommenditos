"""The stack recovers by itself from a crash or a hang of the API (NFR-05).

For the crash, the uvicorn process, not Docker's init in front of it, is killed
with SIGKILL from the host's PID namespace, as the kernel's OOM killer or a
segfault would end it. `docker kill` would not test this: Docker counts it as a
manual stop, and `restart: unless-stopped` then leaves the container down. For the hang, the
server's processes are stopped with SIGSTOP: they live on and hold the port, so
a connection is accepted and never answered, as with a blocked event loop.
Both signals are sent from a throwaway container of the API's own image with
the host's PID namespace, so they need neither sudo nor another image.
"""

from datetime import UTC, datetime
import time

import httpx2
import pytest
from tests.integration.conftest import Container, docker

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: NFR-05: back within 30 seconds of a crash or a hang.
RECOVERY_S = 30

#: The API's tmpfs mounts (deploy/compose.yaml) with the owner and mode they keep
#: after a restart. Without `uid` and `gid` a tmpfs is root's, and the service
#: user cannot write it.
# Paths inside the container, not temporary files of this process.
WRITABLE_TMPFS = {"/tmp": "10001:10001 700", "/tmp/prometheus": "10001:10001 700"}  # noqa: S108


@pytest.mark.req("NFR-05")
def test_the_api_is_healthy_again_within_30_s_of_a_crash(
    client, api_container, record_testsuite_property
):
    before = api_container.inspect()
    restarts, image = before["RestartCount"], before["Image"]
    server = _server_processes(api_container)
    assert server, "the API container runs no server process"

    # The server's own process, the first of its tree, not the init in front of it.
    _signal(image, server[:1], "SIGKILL")
    killed_at = time.monotonic()

    status = None
    while time.monotonic() - killed_at < RECOVERY_S:
        try:
            status = client.get("/health", timeout=2).status_code
        except httpx2.TransportError:
            status = None
        if status == 200 and api_container.inspect()["RestartCount"] > restarts:
            break
        time.sleep(0.5)
    recovered_after = time.monotonic() - killed_at

    after = api_container.inspect()
    assert after["RestartCount"] == restarts + 1, "Docker did not restart the API after the kill"
    assert status == 200, f"/health still answers {status} {recovered_after:.1f} s after the kill"
    stat = api_container.exec("stat", "-c", "%n %u:%g %a", *WRITABLE_TMPFS)
    owners = dict(line.split(" ", 1) for line in stat.splitlines())
    assert owners == WRITABLE_TMPFS, f"after the restart the tmpfs mounts are {owners}"
    # As the service user, the default user of `docker exec`.
    api_container.exec("python", "-c", "import tempfile; tempfile.TemporaryFile().close()")
    # In the JUnit file, which CI and `make test-integration` write.
    record_testsuite_property("recovery_s", round(recovered_after, 1))


@pytest.mark.req("NFR-05")
def test_the_api_is_healthy_again_within_30_s_of_a_hang(
    client, api_container, record_testsuite_property, request
):
    _wait_out_the_start_period(api_container)
    image = api_container.inspect()["Image"]
    frozen = _server_processes(api_container)
    assert frozen, "the API container runs no server process"
    # Thawed again if nothing replaced them, so the stack can be stopped as usual.
    request.addfinalizer(lambda: _signal(image, frozen, "SIGCONT"))

    _signal(image, frozen, "SIGSTOP")
    frozen_at = time.monotonic()

    status = None
    while time.monotonic() - frozen_at < RECOVERY_S:
        try:
            status = client.get("/health", timeout=2).status_code
        except httpx2.TransportError:
            status = None
        if status == 200:
            break
        time.sleep(0.5)
    recovered_after = time.monotonic() - frozen_at

    assert status == 200, (
        f"/health still answers {status} {recovered_after:.1f} s after the API hung; "
        "nothing restarts a hung API"
    )
    record_testsuite_property("hang_recovery_s", round(recovered_after, 1))


def _wait_out_the_start_period(container: Container) -> None:
    """Wait until the healthcheck's start period is over, when a failure counts as a hang.

    The crash test before this one restarts the container, and a hang during
    start-up is not what this test drills.
    """
    state = container.inspect()
    start_period_s = state["Config"].get("Healthcheck", {}).get("StartPeriod", 0) / 1e9
    started_at = datetime.fromisoformat(state["State"]["StartedAt"])
    remaining = start_period_s - (datetime.now(UTC) - started_at).total_seconds()
    if remaining > 0:
        time.sleep(remaining + 1)


def _server_processes(container: Container) -> list[int]:
    """The host PIDs of the server: the image's command first, then every process it started.

    An init that Docker or the image puts in front of the command is left
    running, as is a healthcheck probe, which is no child of the command.
    """
    command = " ".join(container.inspect()["Config"]["Cmd"])
    rows = docker("top", container.id, "-eo", "pid,ppid,args").splitlines()[1:]
    fields = (row.split(None, 2) for row in rows)
    processes = [(int(pid), int(ppid), args) for pid, ppid, args in fields]
    server = [pid for pid, _, args in processes if args == command]
    for parent in server:  # grows while it runs: children, then theirs
        server += [pid for pid, ppid, _ in processes if ppid == parent]
    return server


def _signal(image: str, pids: list[int], name: str) -> None:
    """Send the signal `name` to host processes from a throwaway root container."""
    script = (
        "import os, signal\n"
        f"for pid in {pids}:\n"
        "    try:\n"
        f"        os.kill(pid, signal.{name})\n"
        "    except ProcessLookupError:\n"
        "        pass\n"
    )
    docker(
        "run", "--rm", "--pid=host", "--user=0", "--network=none", "--no-healthcheck",
        "--entrypoint=python", image,
        "-c", script,
    )  # fmt: skip
