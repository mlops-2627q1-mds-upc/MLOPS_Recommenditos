"""The container's healthcheck: `python -m recommenditos.api.healthcheck`.

Docker runs it as the image's HEALTHCHECK, every 5 s once the API is up. It
asks `/health` up to twice. When neither attempt gets an answer and the
container is past its start period, the API is hung: its process lives on but
does not answer, which Docker's restart policy cannot see, since it acts only
when the process exits. The probe then kills the server, the init in front of
it exits, and `restart: unless-stopped` starts the container again (NFR-05,
EDN-93, docs/docs/deployment.md). The start period ends early, as Docker's does,
at the first answer: a probe that got one leaves a marker, and a hang after it
is ended at the next probes rather than only after the full start period.

The kill needs Docker's init as PID 1, which Compose's `init: true` provides:
the kernel drops a signal sent from inside the container to its PID 1 unless
the process handles it, and uvicorn's handlers only ask the blocked event loop
to stop. The probe kills nothing unless PID 1 is that init, and then only its
children. Every container started from the image inherits the HEALTHCHECK, also
one that shares the host's PID namespace, where PID 1 is the host's init and
every process of the machine would be in reach. Without the init the probe
still fails, so Docker marks the container unhealthy, but nothing restarts it.

Standard library only, so that a probe stays cheap.
"""

from collections.abc import Callable
import contextlib
import http.client
import os
from pathlib import Path
import signal
import sys
import urllib.request

#: The API inside the container (the image's CMD).
URL = "http://127.0.0.1:8000/health"

#: The healthcheck's interval and start period; the Dockerfile's HEALTHCHECK and
#: deploy/compose.yaml repeat them (tests/api/test_healthcheck.py checks that).
#: Before the first answer and within the start period, a missing answer is a
#: start still in progress: with several workers uvicorn binds the port before
#: they have loaded the model.
INTERVAL_S = 5.0
START_PERIOD_S = 60.0

#: Two attempts of 6 s each. The bounds on the 6 s:
#: - Recovery (NFR-05, 30 s): a hang that starts just after a probe is ended at
#:   most INTERVAL_S + ATTEMPTS * TIMEOUT_S = 5 + 12 = 17 s later, and the API
#:   answers again about 4 s after the kill (measured on a laptop), 21 s in all.
#:   That leaves room for a restart of up to 13 s on a slower machine.
#: - No kill under load: `/health` is a sync route, so it waits once for a
#:   thread of the worker's threadpool, behind the valuations (its dependencies
#:   are async and wait for none). At about 47 ms of CPU per valuation
#:   (recommenditos/api/__main__.py), a 6 s wait needs about 128 requests ahead
#:   of the probe in one worker. The load test (docs/docs/load-testing.md) holds
#:   at most 80 requests in flight, one per Locust user, which queue for at most
#:   about 80 * 47 ms = 3.8 s. With 2 s per attempt, 43 would have sufficed.
#: The image's HEALTHCHECK timeout leaves room for both attempts.
ATTEMPTS = 2
TIMEOUT_S = 6.0

#: The executable Docker's `--init` (Compose's `init: true`) runs as PID 1.
DOCKER_INIT = "/sbin/docker-init"

PROC = Path("/proc")

#: Left by a probe that got an answer, holding the start time of the PID 1 it
#: saw. On the API's `/tmp` tmpfs, which Compose empties at every restart; the
#: start time makes a marker from an earlier run of the container void anyway.
MARKER = Path("/tmp/recommenditos-healthy")  # noqa: S108 - a path in the container

#: Windows has no SIGKILL; the probe runs in the Linux container only.
SIGKILL: int = getattr(signal, "SIGKILL", 9)


def answers(url: str = URL, timeout: float = TIMEOUT_S) -> bool:
    """Whether `url` answers with a success status within `timeout` seconds."""
    try:
        # An http URL: URL in the container, a local test server in the tests.
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310
            return True
    except (OSError, http.client.HTTPException):
        return False


def init_started_ticks(proc: Path = PROC) -> int:
    """When the container's PID 1 started, in clock ticks since boot."""
    return int(_stat_fields(proc, 1)[19])  # field 22, `starttime`


def container_uptime_s(proc: Path = PROC) -> float:
    """The seconds since the container's PID 1 started, read from Linux's procfs."""
    since_boot_s = float((proc / "uptime").read_text(encoding="ascii").split()[0])
    return since_boot_s - init_started_ticks(proc) / os.sysconf("SC_CLK_TCK")


def mark_healthy(marker: Path = MARKER) -> None:
    """Record that the API answered while the current PID 1 runs."""
    started = str(init_started_ticks())
    with contextlib.suppress(OSError):  # no writable /tmp: the start period applies
        if not marker.exists() or marker.read_text(encoding="ascii") != started:
            marker.write_text(started, encoding="ascii")


def was_healthy(marker: Path = MARKER) -> bool:
    """Whether the API answered since the current PID 1 started."""
    try:
        return marker.read_text(encoding="ascii") == str(init_started_ticks())
    except OSError:
        return False


def server_processes(proc: Path = PROC) -> list[int]:
    """The children of Docker's init, which are the server; none when PID 1 is another process."""
    try:
        init = (proc / "1" / "cmdline").read_bytes().split(b"\0")[0]
    except OSError:
        return []
    if init.decode(errors="replace") != DOCKER_INIT:
        return []
    children = []
    for entry in proc.iterdir():
        if entry.name.isdigit() and entry.name != "1":
            try:
                if int(_stat_fields(proc, int(entry.name))[1]) == 1:  # field 4, `ppid`
                    children.append(int(entry.name))
            except (OSError, IndexError, ValueError):
                continue  # ended while we looked
    return sorted(children)


def end_the_server(proc: Path = PROC, kill: Callable[[int, int], None] = os.kill) -> list[int]:
    """Kill the init's children and return their PIDs.

    The init exits with its child, and the kernel then ends every other process
    of the container, the uvicorn workers included.
    """
    pids = server_processes(proc)
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):  # ended meanwhile
            kill(pid, SIGKILL)
    return pids


def _stat_fields(proc: Path, pid: int) -> list[str]:
    """The fields of /proc/<pid>/stat from field 3 on; field 2, the name, may hold spaces."""
    return (proc / str(pid) / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()


def main(marker: Path = MARKER) -> int:
    for _ in range(ATTEMPTS):
        if answers():
            mark_healthy(marker)
            return 0
    if was_healthy(marker) or container_uptime_s() > START_PERIOD_S:
        # Docker keeps the probe's output in `docker inspect` (State.Health.Log).
        killed = end_the_server()
        print(
            f"/health did not answer {ATTEMPTS} times in {TIMEOUT_S:g} s; "
            f"killed {killed or 'nothing, PID 1 is not Docker init'} (NFR-05)"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
