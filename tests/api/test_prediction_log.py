"""The prediction log: one record per answer (FR-13), exactly once (NFR-03), no PII (NFR-08).

The tests read the SQLite file as the drift job will, with plain SQL, so they
check what lands on disk rather than what the store says it wrote.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
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

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import httpx2
import pytest
from tests.api.conftest import API_KEY
from tests.api.helpers import (
    SKIP_UVICORN_WORKERS_ON_WINDOWS,
    answer_record,
    answers,
    assert_problem,
    write_records,
)
from tests.conftest import PII_COLUMNS

from recommenditos.api import prediction_log
from recommenditos.api.app import create_app
from recommenditos.api.comparables import DEFAULT_K
from recommenditos.api.prediction_log import (
    AnswerLoggedTwice,
    AnswerRecord,
    Feedback,
    PredictionLog,
    SqliteStore,
    StoreError,
)
from recommenditos.api.routes import V1_ROUTERS
from recommenditos.api.schemas.common import ListingFields, ModelInfo
from recommenditos.api.schemas.comparables import ComparablesRequest
from recommenditos.api.schemas.partial_listing import PartialListing
from recommenditos.api.schemas.predict import Explanation, Valuation, ValuationRequest
from recommenditos.api.serving import Estimate, ServedModel, sent_fields
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
    "client",
    "inputs_sent_only",
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
    assert json.loads(record["inputs"]) == {**listing, "country_code": "ES"}
    assert record["inputs_sent_only"] == 1
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


def test_the_valuation_endpoints_include_predict_and_price_range():
    """Guards the test above against passing on no endpoint, or on fewer than there are."""
    assert {PREDICT, "/v1/price-range"} <= set(_valuation_paths())


#: Every endpoint that logs its answers, with the body type it validates.
LOGGING_ENDPOINTS = [
    (PREDICT, ValuationRequest),
    ("/v1/price-range", PartialListing),
    ("/v1/comparables", ComparablesRequest),
]


@pytest.mark.req("FR-13")
@pytest.mark.parametrize(("endpoint", "schema"), LOGGING_ENDPOINTS)
def test_fr13_a_record_holds_only_the_fields_the_client_sent(
    client: TestClient, listing: dict, log_file: Path, endpoint: str, schema: type[ListingFields]
):
    """#191: a field left out, or sent as null, is not logged as its default.

    A field sent with its default value is logged all the same, so that the
    drift job sees a client's `false` or empty list as the value it is.
    """
    defaults_sent = {"is_rental": False, "equipment_comfort": [], "non_smoking": False}
    if endpoint == "/v1/comparables":
        defaults_sent["k"] = DEFAULT_K
    body = {**listing, **defaults_sent, "gears": None}

    response = client.post(endpoint, json=body)

    assert response.status_code == 200, response.text
    (record,) = _record_of(log_file, response.json()["request_id"])
    logged = json.loads(record["inputs"])
    assert logged == {**listing, **defaults_sent}
    assert set(logged) == sent_fields(schema.model_validate_json(json.dumps(body)))
    assert record["inputs_sent_only"] == 1


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

    with TestClient(create_app(bundle, Settings(prediction_log=log_file))):
        pass

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
    bundle: Path, tmp_path: Path, listing: dict, monkeypatch: pytest.MonkeyPatch
):
    def full_disk(self: SqliteStore, record: AnswerRecord) -> None:
        raise StoreError("database or disk is full")

    monkeypatch.setattr(SqliteStore, "record_answer", full_disk)
    app = create_app(bundle, Settings(prediction_log=tmp_path / "log.sqlite3"))
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
@SKIP_UVICORN_WORKERS_ON_WINDOWS
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
        # Each worker loads the model and opens the log on its own, then
        # listens on the port (`answers`).
        deadline = time.monotonic() + 180
        while server_log.read_text(encoding="utf-8").count(
            "Application startup complete"
        ) < 2 or not answers(f"http://127.0.0.1:{port}/health"):
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
def test_nfr04_a_cap_smaller_than_a_batch_keeps_rows_and_seq_keeps_growing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The cap holds far fewer rows than one batch: a prune must not empty the log.

    An emptied table numbers its next row 1 again, and the drift job, which
    forms its windows by `seq`, would see the log start over.
    """
    monkeypatch.setattr(prediction_log, "PRUNE_EVERY", 10**9)
    assert prediction_log.PRUNE_BATCH == 1000
    # The store's own pruning thread stays out of it; this test runs the prunes.
    prune = SqliteStore.prune
    monkeypatch.setattr(SqliteStore, "prune", lambda self: 0)
    store = SqliteStore(tmp_path / "log.sqlite3", max_bytes=10**9)
    try:
        for index in range(200):
            store.record_answer(answer_record(f"old-{index:03d}"))
        # A cap that holds about 60 of the 200 rows, far fewer than a batch.
        store.max_bytes = store.used_bytes() * 3 // 10
        deleted = prune(store)
        highest_before = _rows(store.path, "SELECT max(seq) FROM predictions")[0][0]
        for index in range(20):
            store.record_answer(answer_record(f"new-{index:03d}"))
        prune(store)
        used = store.used_bytes()
    finally:
        store.close()

    kept = _rows(store.path, "SELECT seq, request_id FROM predictions ORDER BY seq")
    assert 0 < deleted < 200
    assert 1 < len(kept) < 100
    assert used <= store.max_bytes
    assert highest_before == 200
    # `seq` goes on from 200 and never starts again at 1, and the newest survive.
    seqs = [seq for seq, _ in kept]
    assert seqs == list(range(seqs[0], 221))
    assert kept[-1][1] == "new-019"


@pytest.mark.req("NFR-04")
def test_nfr04_a_cap_below_one_row_keeps_the_newest_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(prediction_log, "PRUNE_EVERY", 10**9)
    prune = SqliteStore.prune
    monkeypatch.setattr(SqliteStore, "prune", lambda self: 0)
    store = SqliteStore(tmp_path / "log.sqlite3", max_bytes=10**9)
    try:
        for index in range(50):
            store.record_answer(answer_record(f"r-{index:02d}"))
        store.max_bytes = 1
        prune(store)
        store.record_answer(answer_record("r-50"))
        prune(store)
    finally:
        store.close()

    rows = _rows(store.path, "SELECT seq, request_id FROM predictions")
    assert [tuple(row) for row in rows] == [(51, "r-50")]


@pytest.mark.req("NFR-04")
def test_nfr04_nothing_is_pruned_under_the_cap(store: SqliteStore):
    for index in range(50):
        store.record_answer(answer_record(f"r-{index}"))

    assert store.prune() == 0
    assert _rows(store.path, "SELECT count(*) FROM predictions")[0][0] == 50


# --------------------------------------------------------------------------
# The app as a whole
# --------------------------------------------------------------------------


def test_the_app_holds_its_store(client: TestClient, log_file: Path):
    store = client.app_state["store"]
    assert isinstance(store, SqliteStore)
    assert store.path == log_file


def test_a_log_that_cannot_be_opened_stops_the_startup(bundle: Path, tmp_path: Path):
    log_file = tmp_path / "log.sqlite3"
    connection = sqlite3.connect(log_file)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    app = create_app(bundle, Settings(prediction_log=log_file))

    with pytest.raises(StoreError, match="schema version 99"), TestClient(app):
        pass


@pytest.mark.req("FR-13")
def test_an_app_started_twice_records_in_both_runs(bundle: Path, tmp_path: Path, listing: dict):
    """A second run of the same app object opens a store of its own (Schemathesis restarts it)."""
    log_file = tmp_path / "log.sqlite3"
    app = create_app(bundle, Settings(prediction_log=log_file))
    for _ in range(2):
        with TestClient(app) as client:
            assert client.post(PREDICT, json=listing).status_code == 200

    assert _rows(log_file, "SELECT count(*) FROM predictions")[0][0] == 2


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

    used_bytes, delete_oldest = SqliteStore._used_bytes, SqliteStore._delete_oldest
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

    monkeypatch.setattr(SqliteStore, "_used_bytes", slow_used_bytes)
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


# --------------------------------------------------------------------------
# #269: the kind of client, so that M6 can leave the web page's traffic out
# --------------------------------------------------------------------------

#: The layout the first deployed images wrote, without `client`.
FIRST_LAYOUT = """
CREATE TABLE predictions (
    seq INTEGER PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, endpoint TEXT NOT NULL,
    answered_at TEXT NOT NULL, latency_ms REAL NOT NULL, model_variant TEXT NOT NULL,
    model_version TEXT NOT NULL, inputs TEXT NOT NULL, outputs TEXT NOT NULL,
    warnings TEXT NOT NULL
);
CREATE TABLE feedback (
    request_id TEXT PRIMARY KEY REFERENCES predictions (request_id) ON DELETE CASCADE,
    observed_price_eur REAL NOT NULL, received_at TEXT NOT NULL
);
CREATE VIEW labelled_predictions AS
    SELECT predictions.*, feedback.observed_price_eur, feedback.received_at
    FROM predictions JOIN feedback USING (request_id);
PRAGMA user_version = 1;
"""

#: The insert of an image from before `client`, word for word.
FIRST_INSERT = (
    "INSERT INTO predictions (request_id, endpoint, answered_at, latency_ms, "
    "model_variant, model_version, inputs, outputs, warnings) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
FIRST_ROW = ("2026-11-01T00:00:00Z", 1.0, "v", "1", "{}", "{}", "[]")

#: The insert of an image from `client` (#269) to before `inputs_sent_only`
#: (#191), word for word.
SECOND_INSERT = (
    "INSERT INTO predictions (request_id, endpoint, answered_at, latency_ms, "
    "model_variant, model_version, inputs, outputs, warnings, client) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


@pytest.mark.parametrize(
    ("header", "stored"),
    [
        ("web-ui", "web-ui"),
        ("smoke", "smoke"),
        ("loadtest", "loadtest"),
        ("replay", "replay"),
        (None, None),
        ("someone@example.com", None),
        ("Web-UI", None),
    ],
)
def test_a_record_names_the_kind_of_client_only_when_it_is_a_known_one(
    client: TestClient, listing: dict, log_file: Path, header: str | None, stored: str | None
):
    headers = {"X-Client": header} if header is not None else {}
    for endpoint, body in (
        (PREDICT, listing),
        ("/v1/price-range", listing),
        ("/v1/comparables", listing),
    ):
        response = client.post(endpoint, json=body, headers=headers)
        assert response.status_code == 200, response.text
        (record,) = _record_of(log_file, response.json()["request_id"])
        assert record["client"] == stored, endpoint


def test_a_file_of_the_first_layout_gets_the_added_columns_and_keeps_its_rows(tmp_path: Path):
    path = tmp_path / "log.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(FIRST_LAYOUT)
    connection.execute(FIRST_INSERT, ("old", PREDICT, *FIRST_ROW))
    connection.commit()
    connection.close()

    store = SqliteStore(path, max_bytes=10**9)
    try:
        store.record_answer(dataclasses.replace(answer_record("new"), client="web-ui"))
    finally:
        store.close()

    rows = {
        row["request_id"]: (row["client"], row["inputs_sent_only"])
        for row in _rows(path, "SELECT * FROM predictions")
    }
    assert rows == {"old": (None, None), "new": ("web-ui", 1)}
    assert _rows(path, "PRAGMA user_version")[0][0] == prediction_log.SCHEMA_VERSION


@pytest.mark.parametrize(
    ("insert", "row"),
    [
        (FIRST_INSERT, ("older", PREDICT, *FIRST_ROW)),
        (SECOND_INSERT, ("older", PREDICT, *FIRST_ROW, "smoke")),
    ],
    ids=["before-client", "before-inputs-sent-only"],
)
def test_an_older_image_still_writes_to_the_extended_file(
    store: SqliteStore, insert: str, row: tuple
):
    """A rollback to an older image keeps the prediction log working (NFR-13).

    Its insert names its columns, so the ones it does not know get NULL: no
    `inputs_sent_only`, which is right, since its `inputs` hold the defaults.
    """
    store.record_answer(dataclasses.replace(answer_record("new"), client="web-ui"))
    older = sqlite3.connect(store.path)
    older.execute(insert, row)
    older.commit()
    older.close()
    (record,) = _record_of(store.path, "older")
    assert record["inputs_sent_only"] is None
    assert record["client"] == (row[-1] if insert == SECOND_INSERT else None)


# --------------------------------------------------------------------------
# #191: only what the client sent, and a mark that says so
# --------------------------------------------------------------------------


def _file_as_deployed_before_191(path: Path) -> None:
    """The file on the VM's volume before #191: the first layout, `client` added by #269.

    It holds two rows the old way, with the defaults of the fields left out.
    """
    connection = sqlite3.connect(path)
    connection.executescript(FIRST_LAYOUT)
    connection.execute("ALTER TABLE predictions ADD COLUMN client TEXT")
    with_defaults = json.dumps({"make": "Audi", "is_rental": False, "equipment_comfort": []})
    for request_id, client in (("old-load-test", None), ("old-smoke", "smoke")):
        connection.execute(
            SECOND_INSERT,
            (request_id, PREDICT, "2026-10-31T00:00:00Z", 1.0, "v", "1")
            + (with_defaults, "{}", "[]", client),
        )
    connection.commit()
    connection.close()


def test_the_deployed_file_gets_inputs_sent_only_in_place_and_keeps_its_rows(tmp_path: Path):
    path = tmp_path / "log.sqlite3"
    _file_as_deployed_before_191(path)
    before = {row["request_id"]: dict(row) for row in _rows(path, "SELECT * FROM predictions")}

    store = SqliteStore(path, max_bytes=10**9)
    try:
        store.record_answer(answer_record("new"))
    finally:
        store.close()

    after = {row["request_id"]: dict(row) for row in _rows(path, "SELECT * FROM predictions")}
    assert {key: after[key] for key in before} == {
        key: {**row, "inputs_sent_only": None} for key, row in before.items()
    }
    assert after["new"]["inputs_sent_only"] == 1
    assert _rows(path, "PRAGMA user_version")[0][0] == prediction_log.SCHEMA_VERSION
    # How the drift job reads the log (docs/docs/api.md#reading-the-log).
    sent_only = _rows(path, "SELECT request_id FROM predictions WHERE inputs_sent_only = 1")
    assert [row["request_id"] for row in sent_only] == ["new"]


def test_an_image_from_before_191_opens_the_extended_file_and_writes_unmarked_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A rollback image starts on the file this one extended and records there (NFR-13).

    It checks the schema version, which #191 leaves at 1, and adds only the
    columns it knows, so the extra one does not stop it; `_configure` is the
    same code in both images, with `ADDED_COLUMNS` the only difference. Its
    insert names its columns, so its rows read as not sent-only.
    """
    path = tmp_path / "log.sqlite3"
    newer = SqliteStore(path, max_bytes=10**9)
    newer.record_answer(answer_record("newer"))
    newer.close()
    monkeypatch.setattr(prediction_log, "ADDED_COLUMNS", {"client": "TEXT"})

    older = SqliteStore(path, max_bytes=10**9)
    try:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(SECOND_INSERT, ("older", PREDICT, *FIRST_ROW, "smoke"))
    finally:
        older.close()

    rows = _rows(path, "SELECT request_id, client, inputs_sent_only FROM predictions ORDER BY seq")
    assert [tuple(row) for row in rows] == [("newer", None, 1), ("older", "smoke", None)]


def test_the_labelled_view_carries_the_client(client: TestClient, listing: dict, log_file: Path):
    answer = client.post(PREDICT, json=listing, headers={"X-Client": "web-ui"}).json()
    stored = client.post(
        "/v1/feedback",
        json={"request_id": answer["request_id"], "observed_price_eur": 20000},
        headers={"X-API-Key": API_KEY},
    )
    assert stored.status_code in {200, 201}, stored.text
    (row,) = _rows(
        log_file,
        "SELECT client FROM labelled_predictions WHERE request_id = ?",
        answer["request_id"],
    )
    assert row["client"] == "web-ui"
