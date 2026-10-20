"""The prediction log: one record per answer (FR-13), exactly once (NFR-03), no PII (NFR-08).

The tests read the SQLite file as the drift job will, with plain SQL, so they
check what lands on disk rather than what the store says it wrote.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from datetime import UTC, datetime, timedelta
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import httpx2
import pytest
from tests.api.conftest import API_KEY
from tests.api.helpers import answer_record, assert_problem, write_records
from tests.conftest import PII_COLUMNS

from recommenditos.api import prediction_log
from recommenditos.api.app import create_app
from recommenditos.api.prediction_log import (
    AnswerLoggedTwice,
    AnswerRecord,
    Feedback,
    PredictionLog,
    SqliteStore,
    StoreError,
)
from recommenditos.api.routes import V1_ROUTERS
from recommenditos.api.schemas.common import ModelInfo
from recommenditos.api.schemas.predict import Explanation, Valuation, ValuationRequest
from recommenditos.api.serving import Estimate, ServedModel
from recommenditos.api.settings import API_KEY_ENV_VAR, PREDICTION_LOG_ENV_VAR, Settings

PREDICT = "/v1/predict"

#: Every column of the two tables; a new one is a decision about what is logged.
PREDICTION_COLUMNS = {
    "seq",
    "request_id",
    "endpoint",
    "answered_at",
    "latency_ms",
    "model_variant",
    "model_version",
    "inputs",
    "outputs",
    "warnings",
}
FEEDBACK_COLUMNS = {"request_id", "observed_price_eur", "received_at"}


def _rows(log_file: Path, sql: str, *parameters: object) -> list[sqlite3.Row]:
    """Rows of `sql` against the log, read through a connection of its own."""
    connection = sqlite3.connect(log_file)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(sql, parameters).fetchall()
    finally:
        connection.close()


def _record_of(log_file: Path, request_id: str) -> list[sqlite3.Row]:
    return _rows(log_file, "SELECT * FROM predictions WHERE request_id = ?", request_id)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SqliteStore]:
    opened = SqliteStore(tmp_path / "log.sqlite3", max_bytes=10**9)
    yield opened
    opened.close()


# --------------------------------------------------------------------------
# FR-13: what a record holds
# --------------------------------------------------------------------------


@pytest.mark.req("FR-13", "FR-05")
def test_fr13_an_answer_is_one_record_with_its_inputs_outputs_warnings_version_time_latency(
    client: TestClient, listing: dict, log_file: Path
):
    before = datetime.now(UTC)
    started = time.perf_counter()
    response = client.post(PREDICT, json={**listing, "country_code": "ES"})
    elapsed_ms = (time.perf_counter() - started) * 1000
    body = response.json()

    assert response.status_code == 200, response.text
    records = _record_of(log_file, body["request_id"])
    assert len(records) == 1
    record = records[0]
    assert record["endpoint"] == PREDICT
    assert json.loads(record["inputs"]) == ValuationRequest(
        **{**listing, "country_code": "ES"}
    ).model_dump(mode="json", exclude_none=True)
    assert json.loads(record["outputs"]) == {
        "price_eur": body["price_eur"],
        "explanation": body["explanation"],
    }
    assert json.loads(record["warnings"]) == body["warnings"] != []
    assert (record["model_variant"], record["model_version"]) == (
        body["model"]["variant"],
        body["model"]["version"],
    )
    answered_at = datetime.fromisoformat(record["answered_at"])
    assert answered_at.utcoffset() == timedelta(0)
    assert before - timedelta(seconds=1) <= answered_at <= datetime.now(UTC)
    assert 0 < record["latency_ms"] <= elapsed_ms


def _valuation_paths() -> list[str]:
    """Every endpoint of the client contract that answers with a valuation."""
    return [
        f"/v1{route.path}"
        for router in V1_ROUTERS
        for route in router.routes
        if isinstance(route, APIRoute) and "valuation" in (route.tags or [])
    ]


@pytest.mark.req("FR-13", "NFR-03")
@pytest.mark.parametrize("path", _valuation_paths())
def test_fr13_every_valuation_endpoint_records_each_answer_once(
    path: str, client: TestClient, listing: dict, log_file: Path
):
    """Holds an endpoint added later to the log without a test of its own.

    It posts the listing every valuation endpoint accepts; an endpoint with
    another body extends this test with its own.
    """
    response = client.post(path, json=listing)

    assert response.status_code == 200, response.text
    records = _record_of(log_file, response.headers["X-Request-ID"])
    assert [record["endpoint"] for record in records] == [path]


def test_the_valuation_endpoints_include_predict():
    """Guards the test above against passing on no endpoint at all."""
    assert PREDICT in _valuation_paths()


@pytest.mark.req("FR-13")
def test_fr13_a_refused_request_is_not_an_answer_and_writes_no_record(
    client: TestClient, listing: dict, log_file: Path
):
    (before,) = _rows(log_file, "SELECT count(*) FROM predictions")[0]

    response = client.post(PREDICT, json={**listing, "power_kw": -1})

    assert_problem(response, 422)
    assert _rows(log_file, "SELECT count(*) FROM predictions")[0][0] == before


@pytest.mark.req("FR-13")
def test_fr13_startup_creates_the_log(bundle: Path, tmp_path: Path):
    log_file = tmp_path / "new" / "prediction-log.sqlite3"

    create_app(bundle, Settings(prediction_log=log_file)).state.store.close()

    tables = {row[0] for row in _rows(log_file, "SELECT name FROM sqlite_master")}
    assert {"predictions", "feedback", "labelled_predictions"} <= tables
    assert _rows(log_file, "PRAGMA journal_mode")[0][0] == "wal"


def test_a_log_with_another_schema_version_stops_the_service(tmp_path: Path):
    log_file = tmp_path / "log.sqlite3"
    connection = sqlite3.connect(log_file)
    connection.execute("PRAGMA user_version = 99")
    connection.close()

    with pytest.raises(StoreError, match="schema version 99"):
        SqliteStore(log_file, max_bytes=10**9)


# --------------------------------------------------------------------------
# NFR-08: no PII, no field for the caller's address
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-08", "FR-13")
def test_nfr08_the_log_has_no_pii_and_no_field_for_the_callers_address(
    client: TestClient, listing: dict, log_file: Path
):
    address = "203.0.113.77"
    response = client.post(
        PREDICT, json=listing, headers={"X-Forwarded-For": address, "X-Real-IP": address}
    )
    assert response.status_code == 200

    columns = {row["name"] for row in _rows(log_file, "PRAGMA table_info(predictions)")}
    feedback = {row["name"] for row in _rows(log_file, "PRAGMA table_info(feedback)")}
    assert (columns, feedback) == (PREDICTION_COLUMNS, FEEDBACK_COLUMNS)
    (record,) = _record_of(log_file, response.headers["X-Request-ID"])
    logged = set(json.loads(record["inputs"])) | set(json.loads(record["outputs"]))
    assert logged <= set(ValuationRequest.model_fields) | set(Valuation.model_fields)
    assert not logged & set(PII_COLUMNS)
    assert address not in json.dumps(dict(record))


# --------------------------------------------------------------------------
# NFR-03: exactly once, or not answered
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-03")
def test_nfr03_a_route_that_records_twice_fails(store: SqliteStore):
    log = PredictionLog(store, request_id="r-1", endpoint=PREDICT, started=time.perf_counter())
    request = ModelInfo(variant="any body", version="", trained_at=None)
    answer = Valuation(
        price_eur=1.0,
        model=ModelInfo(variant="lgbm-basic", version="0" * 32, trained_at=None),
        request_id="r-1",
        explanation=Explanation(
            base_price_eur=1.0, top_features=[], other_features_effect_pct=0.0
        ),
        warnings=[],
    )
    log.record(request, answer)

    with pytest.raises(AnswerLoggedTwice):
        log.record(request, answer)
    assert store.used_bytes() > 0
    with pytest.raises(StoreError):
        store.record_answer(answer_record("r-1"))


@pytest.mark.req("NFR-03")
def test_nfr03_an_answer_the_log_cannot_hold_is_a_503_not_an_answer(
    bundle: Path, tmp_path: Path, listing: dict
):
    app = create_app(bundle, Settings(prediction_log=tmp_path / "log.sqlite3"))

    class FullDisk(SqliteStore):
        def record_answer(self, record: AnswerRecord) -> None:
            raise StoreError("database or disk is full")

    app.state.store.close()
    app.state.store = FullDisk(tmp_path / "log.sqlite3", max_bytes=10**9)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(PREDICT, json=listing)

    body = assert_problem(response, 503)
    assert "price_eur" not in body


@pytest.mark.req("NFR-03")
def test_nfr03_two_processes_writing_one_file_lose_and_double_nothing(tmp_path: Path):
    """Two processes open and write one new file at the same moment, as two workers do."""
    log_file = tmp_path / "log.sqlite3"
    context = multiprocessing.get_context("spawn")
    start = context.Barrier(2)
    count = 300
    workers = [
        context.Process(target=write_records, args=(log_file, prefix, count, start))
        for prefix in ("a", "b")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)

    assert [worker.exitcode for worker in workers] == [0, 0]
    ids = [row[0] for row in _rows(log_file, "SELECT request_id FROM predictions")]
    assert sorted(ids) == sorted(f"{p}-{i}" for p in ("a", "b") for i in range(count))


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.req("NFR-03", "FR-13")
def test_nfr03_two_uvicorn_workers_record_every_request_exactly_once(
    bundle: Path, listing: dict, tmp_path: Path
):
    """The served app, with `--workers 2`, under concurrent requests."""
    log_file = tmp_path / "log.sqlite3"
    port = _free_port()
    environment = {
        **os.environ,
        "RECOMMENDITOS_MODEL_DIR": str(bundle),
        PREDICTION_LOG_ENV_VAR: str(log_file),
        API_KEY_ENV_VAR: API_KEY,
        "MLFLOW_TRACKING_URI": "",
    }
    server_log = tmp_path / "uvicorn.log"
    windows = sys.platform == "win32"
    with server_log.open("w", encoding="utf-8") as stderr:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "recommenditos.api.app:create_app",
                "--factory",
                "--workers",
                "2",
                "--port",
                str(port),
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
    try:
        # Each worker loads the model and opens the log on its own.
        deadline = time.monotonic() + 180
        while server_log.read_text(encoding="utf-8").count("Application startup complete") < 2:
            assert server.poll() is None, server_log.read_text(encoding="utf-8")
            assert time.monotonic() < deadline, server_log.read_text(encoding="utf-8")
            time.sleep(0.25)

        requests = 200
        with httpx2.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as http:

            def predict(_: int) -> str:
                response = http.post(PREDICT, json=listing)
                assert response.status_code == 200, response.text
                return str(response.json()["request_id"])

            with ThreadPoolExecutor(max_workers=16) as pool:
                answered = list(pool.map(predict, range(requests)))
    finally:
        if windows:
            # A console break would reach the test runner's console too, and
            # terminating the supervisor alone would leave its workers running.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(server.pid)], check=False, capture_output=True
            )
        else:
            server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    errors = server_log.read_text(encoding="utf-8")
    assert errors.count("Started server process") >= 2, errors
    logged = [row[0] for row in _rows(log_file, "SELECT request_id FROM predictions")]
    assert len(set(answered)) == requests
    assert sorted(logged) == sorted(answered)


# --------------------------------------------------------------------------
# NFR-04: rotation to the cap, oldest first
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-04")
def test_nfr04_the_log_is_pruned_to_its_cap_oldest_first_with_its_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(prediction_log, "PRUNE_EVERY", 100)
    monkeypatch.setattr(prediction_log, "PRUNE_BATCH", 50)
    cap = 256 * 1024
    store = SqliteStore(tmp_path / "log.sqlite3", max_bytes=cap)
    try:
        total = 3000
        for index in range(total):
            store.record_answer(answer_record(f"r-{index:05d}"))
            if index == 0:
                store.record_feedback(Feedback("r-00000", 9999.0, datetime.now(UTC)))
        store.prune()
        used = store.used_bytes()
    finally:
        store.close()

    kept = [
        row[0] for row in _rows(tmp_path / "log.sqlite3", "SELECT request_id FROM predictions")
    ]
    assert used <= cap
    assert 0 < len(kept) < total
    # The newest survive, in one unbroken run up to the last record.
    assert kept == [f"r-{index:05d}" for index in range(total - len(kept), total)]
    assert _rows(tmp_path / "log.sqlite3", "SELECT count(*) FROM feedback")[0][0] == 0


@pytest.mark.req("NFR-04")
def test_nfr04_nothing_is_pruned_under_the_cap(store: SqliteStore):
    for index in range(50):
        store.record_answer(answer_record(f"r-{index}"))

    assert store.prune() == 0
    assert _rows(store.path, "SELECT count(*) FROM predictions")[0][0] == 50


# --------------------------------------------------------------------------
# The app as a whole
# --------------------------------------------------------------------------


def test_the_app_holds_its_store(app: FastAPI, log_file: Path):
    assert isinstance(app.state.store, SqliteStore)
    assert app.state.store.path == log_file


@pytest.mark.req("NFR-04", "NFR-02")
def test_nfr04_a_write_during_a_prune_waits_for_one_batch_at_most(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The pruning takes the lock per batch, so a request never waits for all of it.

    Each batch is slowed by 20 ms under the lock, which stands in for a large
    batch, so the pruning takes many times longer than any one batch.
    """
    monkeypatch.setattr(prediction_log, "PRUNE_EVERY", 10**9)
    monkeypatch.setattr(prediction_log, "PRUNE_BATCH", 50)
    # The store's own pruning thread stays out of it; this test runs the one prune.
    prune = SqliteStore.prune
    monkeypatch.setattr(SqliteStore, "prune", lambda self: 0)
    store = SqliteStore(tmp_path / "log.sqlite3", max_bytes=10**9)
    for index in range(1000):
        store.record_answer(answer_record(f"old-{index}"))
    store.max_bytes = 64 * 1024

    used_bytes, delete_oldest = SqliteStore.used_bytes, SqliteStore._delete_oldest
    batches: list[float] = []

    def slow_used_bytes(self: SqliteStore) -> int:
        time.sleep(0.02)
        return used_bytes(self)

    def timed_delete_oldest(self: SqliteStore, limit: float) -> int:
        started = time.perf_counter()
        try:
            return delete_oldest(self, limit)
        finally:
            batches.append(time.perf_counter() - started)

    monkeypatch.setattr(SqliteStore, "used_bytes", slow_used_bytes)
    monkeypatch.setattr(SqliteStore, "_delete_oldest", timed_delete_oldest)
    pruner = threading.Thread(target=prune, args=(store,))
    waits: list[float] = []
    try:
        pruner.start()
        while pruner.is_alive() and len(waits) < 2000:
            started = time.perf_counter()
            store.record_answer(answer_record(f"new-{len(waits)}"))
            waits.append(time.perf_counter() - started)
            time.sleep(0.005)
        pruner.join()
    finally:
        store.close()

    assert len(batches) >= 10
    assert len(waits) >= 10
    assert max(waits) <= 2 * max(batches)
    assert sum(batches) >= 5 * max(waits)


@pytest.mark.req("NFR-04", "NFR-02")
def test_nfr04_the_pruning_runs_in_its_own_thread_not_in_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(prediction_log, "PRUNE_EVERY", 5)
    threads: list[str] = []
    checked = threading.Semaphore(0)

    def recording_prune(self: SqliteStore) -> int:
        threads.append(threading.current_thread().name)
        checked.release()
        return 0

    monkeypatch.setattr(SqliteStore, "prune", recording_prune)
    store = SqliteStore(tmp_path / "log.sqlite3", max_bytes=10**9)
    try:
        assert checked.acquire(timeout=10), "no check at startup"
        for index in range(5):
            store.record_answer(answer_record(f"r-{index}"))
        assert checked.acquire(timeout=10), "no check after PRUNE_EVERY records"
    finally:
        store.close()

    assert threads == ["prediction-log-pruner", "prediction-log-pruner"]


# --------------------------------------------------------------------------
# A price that is not a number is no answer
# --------------------------------------------------------------------------


@pytest.mark.req("FR-06", "FR-13")
@pytest.mark.parametrize("price", [math.inf, -math.inf, math.nan])
def test_fr13_a_price_that_is_not_finite_is_a_500_and_writes_no_record(
    price: float,
    client: TestClient,
    listing: dict,
    log_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    value = ServedModel.value

    def not_finite(self: ServedModel, request: ValuationRequest) -> Estimate:
        return dataclasses.replace(value(self, request), price_eur=price)

    monkeypatch.setattr(ServedModel, "value", not_finite)

    response = client.post(PREDICT, json=listing)

    body = assert_problem(response, 500)
    assert "price_eur" not in body
    assert _record_of(log_file, response.headers["X-Request-ID"]) == []


@pytest.mark.req("FR-13")
def test_fr13_the_log_refuses_a_value_json_cannot_hold(store: SqliteStore):
    record = answer_record("r-inf")
    infinite = AnswerRecord(**{**record.__dict__, "outputs": {"price_eur": math.inf}})

    with pytest.raises(StoreError):
        store.record_answer(infinite)
    assert _record_of(store.path, "r-inf") == []
