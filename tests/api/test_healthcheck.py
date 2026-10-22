"""`python -m recommenditos.api.healthcheck`: when the probe ends a hung server (NFR-05).

The integration suite drills the whole path, a frozen server that Docker
restarts (tests/integration/test_deployed_recovery.py); these tests pin the
probe's rule on its own, with the kill replaced.
"""

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import socket
import threading

import pytest

from recommenditos.api import healthcheck


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
