"""`python -m recommenditos.api.healthcheck`: when the probe ends a hung server (NFR-05).

The integration suite drills the whole path, a frozen server that Docker
restarts (tests/integration/test_deployed_recovery.py); these tests pin the
probe's rule on its own, with the kill replaced.
"""

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import re
import socket
import threading

import pytest
import yaml

from recommenditos.api import healthcheck
from recommenditos.config import PROJ_ROOT

#: Seconds from the kill to the next 200 through nginx, measured (deployment.md).
RESTART_S = 4.0


class _Ok(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def healthy_url() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/health"
    server.shutdown()
    server.server_close()


@pytest.fixture
def hung_url() -> Iterator[str]:
    """A port that accepts connections and never answers, as a hung API does."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/health"


@pytest.fixture
def kills(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """The kills `main` would have sent, with nothing killed."""
    sent: list[None] = []
    monkeypatch.setattr(healthcheck, "end_the_server", lambda: sent.append(None))
    return sent


def _answers_in_turn(monkeypatch: pytest.MonkeyPatch, *results: bool) -> None:
    replies = iter(results)
    monkeypatch.setattr(healthcheck, "answers", lambda: next(replies))


def _uptime(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(healthcheck, "container_uptime_s", lambda: seconds)


@pytest.mark.req("NFR-05")
def test_nfr05_an_answer_is_healthy(healthy_url: str):
    assert healthcheck.answers(healthy_url)


@pytest.mark.req("NFR-05")
def test_nfr05_a_connection_without_an_answer_is_not(hung_url: str):
    assert not healthcheck.answers(hung_url, timeout=0.2)


@pytest.mark.req("NFR-05")
def test_nfr05_a_healthy_api_is_left_alone(monkeypatch: pytest.MonkeyPatch, kills: list[None]):
    _answers_in_turn(monkeypatch, True)
    _uptime(monkeypatch, 3600)
    assert healthcheck.main() == 0
    assert kills == []


@pytest.mark.req("NFR-05")
def test_nfr05_one_missed_answer_is_not_a_hang(monkeypatch: pytest.MonkeyPatch, kills: list[None]):
    _answers_in_turn(monkeypatch, False, True)
    _uptime(monkeypatch, 3600)
    assert healthcheck.main() == 0
    assert kills == []


@pytest.mark.req("NFR-05")
def test_nfr05_a_hung_api_is_ended(monkeypatch: pytest.MonkeyPatch, kills: list[None]):
    _answers_in_turn(monkeypatch, False, False)
    _uptime(monkeypatch, healthcheck.START_PERIOD_S + 1)
    assert healthcheck.main() == 1
    assert kills == [None]


@pytest.mark.req("NFR-05")
def test_nfr05_a_starting_api_is_not_ended(monkeypatch: pytest.MonkeyPatch, kills: list[None]):
    _answers_in_turn(monkeypatch, False, False)
    _uptime(monkeypatch, healthcheck.START_PERIOD_S - 1)
    assert healthcheck.main() == 1
    assert kills == []


@pytest.mark.req("NFR-05")
@pytest.mark.skipif(not hasattr(os, "sysconf"), reason="clock ticks are a POSIX notion")
def test_nfr05_the_uptime_counts_from_pid_1s_start(tmp_path: Path):
    """`starttime`, field 22 of /proc/1/stat, in clock ticks since boot."""
    ticks = os.sysconf("SC_CLK_TCK")
    (tmp_path / "1").mkdir()
    (tmp_path / "uptime").write_text("1000.00 3000.00\n", encoding="ascii")
    fields = ["S", *["0"] * 18, str(400 * ticks), "0"]
    (tmp_path / "1" / "stat").write_text(f"1 (docker init) {' '.join(fields)}\n", encoding="ascii")
    assert healthcheck.container_uptime_s(tmp_path) == pytest.approx(600)


def _proc(tmp_path: Path, init: str, parents: dict[int, int]) -> Path:
    """A fake /proc: PID 1 runs `init`, and each other PID has the parent `parents` names."""
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "cmdline").write_bytes(init.encode() + b"\0--\0python\0")
    (tmp_path / "1" / "stat").write_text("1 (init) S 0 1 1", encoding="ascii")
    for pid, parent in parents.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(
            f"{pid} (py thon) S {parent} 1", encoding="ascii"
        )
    (tmp_path / "self").mkdir()  # not a PID
    return tmp_path


@pytest.mark.req("NFR-05")
def test_nfr05_only_the_children_of_docker_init_are_killed(tmp_path: Path):
    """uvicorn (7) is killed; its worker (8) ends with the container; the probe (30) lives."""
    proc = _proc(tmp_path, healthcheck.DOCKER_INIT, {7: 1, 8: 7, 30: 0})
    sent: list[tuple[int, int]] = []
    killed = healthcheck.end_the_server(proc, kill=lambda pid, sig: sent.append((pid, sig)))
    assert killed == [7]
    assert sent == [(7, healthcheck.SIGKILL)]


@pytest.mark.req("NFR-05")
@pytest.mark.parametrize("init", ["/usr/lib/systemd/systemd", "python", "/sbin/tini"])
def test_nfr05_nothing_is_killed_unless_pid_1_is_docker_init(tmp_path: Path, init: str):
    """In the host's PID namespace, or with the server as PID 1, the probe kills nothing."""
    proc = _proc(tmp_path, init, {7: 1, 8: 7})
    sent: list[tuple[int, int]] = []
    assert healthcheck.end_the_server(proc, kill=lambda pid, sig: sent.append((pid, sig))) == []
    assert sent == []


def _seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+)s", value)
    assert match, f"{value!r} is not a duration in whole seconds"
    return float(match[1])


@pytest.mark.req("NFR-05")
def test_nfr05_the_image_and_compose_use_the_probes_timings():
    """One set of timings in the module, the Dockerfile and compose.yaml, within NFR-05's 30 s."""
    dockerfile = (PROJ_ROOT / "Dockerfile").read_text(encoding="utf-8")
    instruction = re.search(r"^HEALTHCHECK (.*)$", dockerfile, re.MULTILINE)
    assert instruction, "the Dockerfile has no HEALTHCHECK"
    flags = dict(re.findall(r"--([a-z-]+)=(\S+)", instruction[1]))
    compose = yaml.safe_load((PROJ_ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8"))
    probe = compose["services"]["api"]["healthcheck"]
    for interval, start, timeout in [
        (flags["interval"], flags["start-period"], flags["timeout"]),
        (probe["interval"], probe["start_period"], probe["timeout"]),
    ]:
        assert _seconds(interval) == healthcheck.INTERVAL_S
        assert _seconds(start) == healthcheck.START_PERIOD_S
        # Both attempts and a second for Python to start, or Docker ends the probe first.
        assert _seconds(timeout) >= healthcheck.ATTEMPTS * healthcheck.TIMEOUT_S + 1
    assert compose["services"]["api"]["init"] is True
    worst_case_s = (
        healthcheck.INTERVAL_S + healthcheck.ATTEMPTS * healthcheck.TIMEOUT_S + RESTART_S
    )
    assert worst_case_s <= 30
