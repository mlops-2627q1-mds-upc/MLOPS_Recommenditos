"""The container's healthcheck: `python -m recommenditos.api.healthcheck`.

Docker runs it as the image's HEALTHCHECK, every 10 s once the API is up. It
asks `/health` up to twice. When neither attempt gets an answer and the
container is past its start period, the API is hung: its process lives on but
does not answer, which Docker's restart policy cannot see, since it acts only
when the process exits. The probe then kills the server, the init in front of
it exits, and `restart: unless-stopped` starts the container again (NFR-05,
EDN-93, docs/docs/deployment.md).

The kill needs that init, which Compose's `init: true` puts in front of the
server: the kernel drops a signal sent from inside the container to its PID 1
unless the process handles it, and uvicorn's handlers only ask the blocked event
loop to stop. Without an init the probe still fails, so Docker marks the
container unhealthy, but nothing restarts it.

Standard library only, so that a probe stays cheap.
"""

import http.client
import os
from pathlib import Path
import signal
import sys
import urllib.request

#: The API inside the container (the image's CMD).
URL = "http://127.0.0.1:8000/health"

#: Two attempts of 2 s each: a hang ends the server at most about 14 s after it
#: starts (10 s interval plus 4 s), and the API is back about 4 s later, well
#: within NFR-05's 30 s. The image's HEALTHCHECK timeout leaves room for both.
ATTEMPTS = 2
TIMEOUT_S = 2.0

#: The healthcheck's start period. Before it ends, a missing answer is a start
#: still in progress: with several workers uvicorn binds the port before they
#: have loaded the model.
START_PERIOD_S = 60.0


def answers(url: str = URL, timeout: float = TIMEOUT_S) -> bool:
    """Whether `url` answers with a success status within `timeout` seconds."""
    try:
        # An http URL: URL in the container, a local test server in the tests.
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310
            return True
    except (OSError, http.client.HTTPException):
        return False


def container_uptime_s(proc: Path = Path("/proc")) -> float:
    """The seconds since the container's PID 1 started, read from Linux's procfs."""
    since_boot_s = float((proc / "uptime").read_text(encoding="ascii").split()[0])
    # The command name in field 2 may hold spaces; the fields after it do not.
    fields = (proc / "1" / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
    started_ticks = int(fields[19])  # field 22, `starttime`
    return since_boot_s - started_ticks / os.sysconf("SC_CLK_TCK")


def end_the_server() -> None:
    """Kill every process the probe may signal, which is the server, but not the init.

    `kill(-1, ...)` skips the caller and the PID namespace's init by definition.
    """
    os.kill(-1, signal.SIGKILL)


def main() -> int:
    for _ in range(ATTEMPTS):
        if answers():
            return 0
    if container_uptime_s() > START_PERIOD_S:
        # Docker keeps the probe's output in `docker inspect` (State.Health.Log).
        print(
            f"/health did not answer {ATTEMPTS} times in {TIMEOUT_S:g} s; "
            "ending the hung server so that Docker restarts it (NFR-05)"
        )
        end_the_server()
    return 1


if __name__ == "__main__":
    sys.exit(main())
