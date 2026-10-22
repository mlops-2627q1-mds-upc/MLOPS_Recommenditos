"""The service log: one JSON object per line, each line of a request with its ID.

The in-process tests attach the JSON sink to loguru and read what a request
wrote. `test_fr11_the_service_logs_one_json_object_per_line` starts the service
the way the image does, `python -m recommenditos.api` with
`RECOMMENDITOS_LOG_FORMAT=json`, with one worker and with two, and reads its
stderr, so uvicorn's own lines and the logging configuration it is started
with are checked too. It stops the service only once every call's line is
written: on Windows the stop kills the process tree at once, and a line the
kill prevents is not one the service failed to write.
"""

from collections.abc import Callable, Iterator
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from fastapi.testclient import TestClient
from loguru import logger
import pytest
from tests.api.helpers import free_port, stop_service

from recommenditos.api.logs import LOG_FORMAT_ENV_VAR, json_sink, log_format, text_format
from recommenditos.api.metrics import UNMATCHED_PATH
from recommenditos.api.middleware import REQUEST_ID_HEADER
from recommenditos.api.serving import ServedModel
from recommenditos.api.settings import MODEL_DIR_ENV_VAR, PREDICTION_LOG_ENV_VAR

PREDICT = "/v1/predict"

#: The fields of the line that says how a request was answered, and no others:
#: no client address, no header, no body (NFR-08).
ANSWER_FIELDS = {
    "time",
    "level",
    "logger",
    "message",
    "request_id",
    "method",
    "path",
    "status",
    "duration_ms",
}

#: A documentation address (RFC 5737) a client claims to be, which no line may repeat.
CLIENT_ADDRESS = "203.0.113.7"

STARTUP_SECONDS = 120

#: uvicorn 0.54 binds the port in its supervisor without listening on it, and on
#: Windows each worker shares the socket into itself and calls `listen()` on it.
#: When the workers start at the same moment, about one start in twenty on CI's
#: Windows runner, one of them fails with "OSError: [WinError 10022]". The
#: supervisor replaces it, but the worker's traceback is printed by
#: `multiprocessing`, not logged, so it is not JSON. The image runs Linux, where
#: the same test with two workers runs.
UVICORN_WINDOWS_WORKER_RACE = pytest.mark.skipif(
    sys.platform == "win32",
    reason="uvicorn's workers race on listen() on Windows (WinError 10022)",
)


@pytest.fixture
def json_log() -> Iterator[Callable[[], list[dict[str, Any]]]]:
    """What loguru writes while the test runs, through the JSON sink, as parsed lines."""
    stream = io.StringIO()
    handler = logger.add(json_sink(stream), level=0, format="{message}")
    try:
        yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    finally:
        logger.remove(handler)


@pytest.mark.req("FR-11")
def test_fr11_every_line_of_a_request_carries_its_request_id(
    client: TestClient,
    listing: dict,
    json_log: Callable[[], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
):
    """A line from the route's worker thread, the 500's cause and the answer line alike."""

    def fail(*_args: Any, **_kwargs: Any) -> None:
        logger.info("valuing the listing")
        raise RuntimeError("the model broke")

    monkeypatch.setattr(ServedModel, "value", fail)

    response = client.post(PREDICT, json=listing)

    assert response.status_code == 500
    lines = json_log()
    assert {line.get("request_id") for line in lines} == {response.headers[REQUEST_ID_HEADER]}
    assert [line["message"] for line in lines][:2] == ["valuing the listing", "the request failed"]
    assert "RuntimeError: the model broke" in lines[1]["exception"]
    assert lines[-1]["status"] == 500


@pytest.mark.req("FR-11", "NFR-08")
@pytest.mark.parametrize(
    "case",
    [
        ("POST", PREDICT, PREDICT, 200),
        ("GET", "/health", "/health", 200),
        ("GET", "/nowhere/alice@example.com", UNMATCHED_PATH, 404),
    ],
    ids=["predict", "health", "unmatched"],
)
def test_fr11_a_request_ends_with_one_line_of_how_it_was_answered_and_nothing_personal(
    client: TestClient,
    listing: dict,
    json_log: Callable[[], list[dict[str, Any]]],
    case: tuple[str, str, str, int],
):
    """The path is the route template, never the one the client sent (NFR-08)."""
    method, sent, path, status = case
    body = listing if method == "POST" else None
    response = client.request(
        method,
        f"{sent}?owner=someone",
        json=body,
        headers={"X-Forwarded-For": CLIENT_ADDRESS, "User-Agent": "test-agent"},
    )

    assert response.status_code == status
    request_id = response.headers[REQUEST_ID_HEADER]
    lines = json_log()
    assert len(lines) == 1
    (answer,) = lines
    assert set(answer) == ANSWER_FIELDS
    assert answer["request_id"] == request_id
    assert (answer["method"], answer["path"], answer["status"]) == (method, path, status)
    assert answer["level"] == "INFO"
    assert answer["message"] == f"{method} {path} {status} {answer['duration_ms']} ms"
    text = json.dumps(answer)
    personal_values = (
        CLIENT_ADDRESS,
        "testclient",
        "test-agent",
        "someone",
        "alice",
        listing["make"],
    )
    for personal in personal_values:
        assert personal not in text


def test_a_text_line_names_the_request_too(client: TestClient):
    stream = io.StringIO()
    handler = logger.add(stream, level=0, format=text_format, colorize=False)
    try:
        response = client.get("/health")
    finally:
        logger.remove(handler)

    assert f"| {response.headers[REQUEST_ID_HEADER]} - GET /health 200 " in stream.getvalue()


def test_an_unknown_log_format_stops_the_service(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(LOG_FORMAT_ENV_VAR, "xml")

    with pytest.raises(ValueError, match=LOG_FORMAT_ENV_VAR):
        log_format()


def _call(url: str, method: str = "GET", body: dict | None = None) -> str:
    """Send a request and return the request ID its answer carries."""
    request = urllib.request.Request(
        url,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Forwarded-For": CLIENT_ADDRESS},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return str(response.headers[REQUEST_ID_HEADER])


def _wait_for(log: Path, condition: Callable[[list[str]], bool], what: str) -> None:
    """Wait until the lines written to `log` so far satisfy `condition`."""
    deadline = time.monotonic() + 30
    while not condition(log.read_text(encoding="utf-8").splitlines()):
        assert time.monotonic() < deadline, f"{what} in time:\n{log.read_text(encoding='utf-8')}"
        time.sleep(0.1)


def _parsed(text: str) -> list[dict[str, Any]]:
    """Every line of a JSON log, parsed; a line that is no JSON object fails with the whole log."""
    lines: list[dict[str, Any]] = []
    for line in filter(str.strip, text.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            parsed = None
        assert isinstance(parsed, dict), f"not a JSON object: {line!r}\nThe whole log:\n{text}"
        lines.append(parsed)
    return lines


@pytest.mark.req("FR-11", "NFR-08")
@pytest.mark.parametrize("workers", [1, pytest.param(2, marks=UVICORN_WINDOWS_WORKER_RACE)])
def test_fr11_the_service_logs_one_json_object_per_line(
    bundle: Path, listing: dict, tmp_path: Path, workers: int
):
    """Every line, uvicorn's and the supervisor's of several workers included, is JSON."""
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = tmp_path / "service.log"
    env = {
        **os.environ,
        LOG_FORMAT_ENV_VAR: "json",
        MODEL_DIR_ENV_VAR: str(bundle),
        PREDICTION_LOG_ENV_VAR: str(tmp_path / "prediction-log.sqlite3"),
        "MLFLOW_TRACKING_URI": "",
    }
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "recommenditos.api", "--host", "127.0.0.1"]
            + ["--port", str(port), "--workers", str(workers)],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + STARTUP_SECONDS
            while True:
                assert process.poll() is None, "the service exited during startup"
                assert time.monotonic() < deadline, "the service did not start in time"
                try:
                    _call(f"{base}/health")
                    break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.5)
            sent = {
                _call(f"{base}{PREDICT}", "POST", listing): ("POST", PREDICT),
                _call(f"{base}/health", "HEAD"): ("HEAD", "/health"),
            }
            _wait_for(
                log,
                lambda lines: all(
                    any(request_id in line and '"status"' in line for line in lines)
                    for request_id in sent
                ),
                "not every request's line was written",
            )
        finally:
            stop_service(process)

    lines = _parsed(log.read_text(encoding="utf-8"))
    assert any(line["logger"].startswith("uvicorn") for line in lines), "uvicorn's lines"
    for request_id, (method, path) in sent.items():
        answers = [line for line in lines if line.get("request_id") == request_id]
        assert len(answers) == 1, answers
        assert (answers[0]["method"], answers[0]["path"], answers[0]["status"]) == (
            method,
            path,
            200,
        )
    for line in lines:
        if "request_id" in line:
            assert "127.0.0.1" not in json.dumps(line)
            assert CLIENT_ADDRESS not in json.dumps(line)
