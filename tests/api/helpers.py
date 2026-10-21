"""Assertions, records and service helpers the API tests share."""

from datetime import UTC, datetime
from pathlib import Path
import signal
import socket
import subprocess
import sys
from typing import Any

from httpx2 import Response

from recommenditos.api.errors import PROBLEM_MEDIA_TYPE
from recommenditos.api.middleware import REQUEST_ID_HEADER
from recommenditos.api.prediction_log import AnswerRecord, SqliteStore

#: The members every problem carries (RFC 9457), and the one a 422 adds.
PROBLEM_MEMBERS = {"type", "title", "status", "detail", "instance"}


def assert_problem(response: Response, status: int) -> dict:
    """`response` is an RFC 9457 problem with `status`; returns its body.

    The same check for every error path is what FR-16's "none differ in shape"
    means in a test: the media type, the members, the status repeated in the
    body, and the request ID in `instance`.
    """
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    expected = PROBLEM_MEMBERS | ({"errors"} if status == 422 else set())
    assert set(body) == expected, body
    assert body["status"] == status
    assert body["instance"] == f"urn:uuid:{response.headers[REQUEST_ID_HEADER]}"
    assert body["title"] and body["detail"]
    return body


def pointers(problem: dict) -> list[str]:
    """The fields a 422 names, in its order."""
    return [error["pointer"] for error in problem["errors"]]


# --------------------------------------------------------------------------
# The prediction log
# --------------------------------------------------------------------------
# Here rather than in a test module, because `write_records` runs in spawned
# processes, which import the module that defines it, and this one is light.


def answer_record(request_id: str) -> AnswerRecord:
    """A typical record, about the size of a real one."""
    return AnswerRecord(
        request_id=request_id,
        endpoint="/v1/predict",
        answered_at=datetime.now(UTC),
        latency_ms=1.0,
        model_variant="lgbm-basic",
        model_version="0" * 32,
        inputs={"make": "Audi", "model": "A4", "power_kw": 110, "equipment_comfort": ["x"] * 20},
        outputs={"price_eur": 21890.0},
        warnings=[],
    )


def write_records(log_file: Path, prefix: str, count: int, start: Any) -> None:
    """One worker process: wait for the others, open the shared file, write `count` records."""
    start.wait()
    store = SqliteStore(log_file, max_bytes=10**9)
    try:
        for index in range(count):
            store.record_answer(answer_record(f"{prefix}-{index}"))
    finally:
        store.close()


def free_port() -> int:
    """A TCP port on the loopback interface that nothing listens on right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def stop_service(process: subprocess.Popen) -> None:
    """Stop a service started with `python -m recommenditos.api`, and its workers.

    SIGTERM lets uvicorn stop its workers itself. Windows has no such signal for
    a process without a console of its own, and a console event would reach the
    test runner too, so there the whole process tree is killed.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], check=False)
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
