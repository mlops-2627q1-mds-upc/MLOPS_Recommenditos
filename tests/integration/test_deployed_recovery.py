"""The stack recovers by itself from a crash of the API process (NFR-05).

The uvicorn process is killed with SIGKILL from the host's PID namespace, as the
kernel's OOM killer or a segfault would end it. `docker kill` would not test
this: Docker counts it as a manual stop, and `restart: unless-stopped` then
leaves the container down. The kill runs in a throwaway container of the API's
own image with the host's PID namespace, so it needs neither sudo nor another
image.
"""

import time

import httpx2
import pytest
from tests.integration.conftest import docker

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: NFR-05: back within 30 seconds of a crash.
RECOVERY_S = 30


@pytest.mark.req("NFR-05")
def test_the_api_is_healthy_again_within_30_s_of_a_crash(
    client, api_container, record_testsuite_property
):
    before = api_container.inspect()
    pid, restarts, image = before["State"]["Pid"], before["RestartCount"], before["Image"]
    assert pid > 0, "the API container has no running process"

    docker(
        "run", "--rm", "--pid=host", "--user=0", "--network=none", "--entrypoint=python", image,
        "-c", f"import os, signal; os.kill({pid}, signal.SIGKILL)",
    )  # fmt: skip
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
    # In the JUnit file, which CI and `make test-integration` write.
    record_testsuite_property("recovery_s", round(recovered_after, 1))
