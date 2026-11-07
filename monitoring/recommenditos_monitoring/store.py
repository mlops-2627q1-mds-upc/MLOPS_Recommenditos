"""`monitor.db`, where the monitoring job keeps what it found per window (FR-14, EDN-101).

One SQLite file on the job's own volume, `monitor-results`, apart from the
prediction log: the API owns the log's schema and rotates it, and a second
writer there would hold up the API's workers. The tables:

- `windows`: one row per closed window of consecutive requests of the log, in
  log order. `tested` is a window of `window_size` requests that the drift test
  ran on; `partial` is one that a change of `model_version` closed early, which
  is not tested, so that no window mixes two models. Each row has the window's
  first and last `seq`, its time range, the model, its share of `ES` listings
  and, when tested, whether it was flagged and how many features drifted.
- `window_features`: per tested window, every feature's test, p-value,
  statistic, and whether it drifted.
- `window_requests`: which requests of the log make up each window, with their
  `request_id`, so that the error and coverage of a window (#229) join the
  feedback of exactly its requests. Only while the log holds them: the log's
  rotation deletes the oldest answers with their feedback, after which nothing
  can be joined to them, so a pass drops the rows of requests the log no longer
  holds (`drop_requests_before`). That bounds the table by the log's cap
  (NFR-04, docs/docs/monitoring.md#how-large-it-grows).
- `window_performance`: per closed window, its real error and the coverage of
  its price ranges, computed from the feedback of its requests (#229,
  `performance.py`), with the test values they are compared with. A pass
  recomputes a row until it is `final`.
- `model_test_values`: per model version, the test values it was evaluated
  with, recorded by the first pass that ran in a release of that model (#307),
  so that a window is compared with its own model's values after a promotion
  too, when the image holds only the new model's.
- `open_window`: the progress of the window being filled, one row.
- `runs`: the last `RUNS_KEPT` passes of the job, so that its last success is
  on record.

A later table is added with `CREATE TABLE IF NOT EXISTS` and leaves these as
they are, as `window_performance` was after #228's; `user_version` changes only
with a change a reader of these tables would notice.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from typing import Any

from recommenditos_monitoring.drift import FeatureResult

SCHEMA_VERSION = 1

TESTED, PARTIAL = "tested", "partial"

#: A figure of `window_performance` has a value (`OK`), or too few labels to have one.
OK, INSUFFICIENT_LABELS = "ok", "insufficient_labels"

#: Passes kept in `runs`: 104 days of one pass every 15 minutes.
RUNS_KEPT = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    window_id         INTEGER PRIMARY KEY,
    status            TEXT NOT NULL CHECK (status IN ('tested', 'partial')),
    first_seq         INTEGER NOT NULL UNIQUE,
    last_seq          INTEGER NOT NULL UNIQUE,
    requests          INTEGER NOT NULL,
    window_size       INTEGER NOT NULL,
    first_answered_at TEXT NOT NULL,
    last_answered_at  TEXT NOT NULL,
    model_variant     TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    es_share          REAL NOT NULL,
    flagged           INTEGER CHECK (flagged IN (0, 1)),
    drifted_features  INTEGER,
    threshold         REAL,
    reference_rows    INTEGER,
    closed_at         TEXT NOT NULL,
    CHECK ((status = 'tested') = (flagged IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS window_features (
    window_id       INTEGER NOT NULL REFERENCES windows (window_id),
    feature         TEXT NOT NULL,
    test            TEXT NOT NULL CHECK (test IN ('ks', 'chi2')),
    p_value         REAL NOT NULL,
    statistic       REAL NOT NULL,
    values_compared INTEGER NOT NULL,
    drifted         INTEGER NOT NULL CHECK (drifted IN (0, 1)),
    PRIMARY KEY (window_id, feature)
);
CREATE TABLE IF NOT EXISTS window_requests (
    seq        INTEGER PRIMARY KEY,
    window_id  INTEGER NOT NULL REFERENCES windows (window_id),
    request_id TEXT NOT NULL UNIQUE,
    endpoint   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS window_requests_by_window ON window_requests (window_id);
CREATE TABLE IF NOT EXISTS window_performance (
    window_id       INTEGER PRIMARY KEY REFERENCES windows (window_id),
    ranges          INTEGER NOT NULL,
    labelled        INTEGER NOT NULL,
    labelled_ranges INTEGER NOT NULL,
    mdape_status    TEXT NOT NULL CHECK (mdape_status IN ('ok', 'insufficient_labels')),
    mdape           REAL,
    coverage_status TEXT NOT NULL CHECK (coverage_status IN ('ok', 'insufficient_labels')),
    coverage        REAL,
    nominal_level   REAL,
    test_mdape      REAL,
    test_coverage   REAL,
    final           INTEGER NOT NULL CHECK (final IN (0, 1)),
    computed_at     TEXT NOT NULL,
    CHECK ((mdape_status = 'ok') = (mdape IS NOT NULL)),
    CHECK ((coverage_status = 'ok') = (coverage IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS model_test_values (
    model_variant TEXT NOT NULL,
    model_version TEXT NOT NULL,
    test_mdape    REAL,
    test_coverage REAL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (model_variant, model_version)
);
CREATE TABLE IF NOT EXISTS open_window (
    singleton         INTEGER PRIMARY KEY CHECK (singleton = 1),
    requests          INTEGER NOT NULL,
    window_size       INTEGER NOT NULL,
    first_seq         INTEGER,
    last_seq          INTEGER,
    model_version     TEXT,
    updated_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id         INTEGER PRIMARY KEY,
    started_at     TEXT NOT NULL,
    finished_at    TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    windows_closed INTEGER NOT NULL,
    detail         TEXT
);
"""


class StoreError(Exception):
    """`monitor.db` is not one this job can use."""


def timestamp(moment: datetime) -> str:
    """ISO 8601 in UTC to the millisecond, the prediction log's form."""
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Window:
    """A closed window, as `windows` holds it."""

    window_id: int
    status: str
    first_seq: int
    last_seq: int
    requests: int
    window_size: int
    first_answered_at: str
    last_answered_at: str
    model_variant: str
    model_version: str
    es_share: float
    flagged: bool | None
    drifted_features: int | None
    threshold: float | None
    reference_rows: int | None
    closed_at: str


@dataclass(frozen=True)
class ModelTestValues:
    """A model version's test values, as `model_test_values` holds them."""

    model_variant: str
    #: The version the API reports and logs: the hash dvc.lock records for the bundle.
    model_version: str
    #: The test MdAPE and coverage with every input given; None where the evaluation has none.
    test_mdape: float | None
    test_coverage: float | None
    recorded_at: str


@dataclass(frozen=True)
class OpenWindow:
    requests: int
    window_size: int
    first_seq: int | None
    last_seq: int | None
    model_version: str | None
    updated_at: str


@dataclass(frozen=True)
class Request:
    """One request of a window: what `window_requests` holds."""

    seq: int
    request_id: str
    endpoint: str


@dataclass(frozen=True)
class Performance:
    """The error and coverage of a closed window, as `window_performance` holds them."""

    window_id: int
    #: Answers of `/v1/price-range` among the window's requests.
    ranges: int
    #: Requests with feedback, of both endpoints; `labelled_ranges` of them ranges.
    labelled: int
    labelled_ranges: int
    mdape_status: str
    #: median(|observed - price_eur| / observed) over the labelled requests.
    mdape: float | None
    coverage_status: str
    #: The share of labelled ranges with lower_eur <= observed <= upper_eur.
    coverage: float | None
    #: The mean `confidence_level` of the labelled ranges, the coverage they promise.
    nominal_level: float | None
    #: The model's test MdAPE and coverage with every input given, when the evaluation has them.
    test_mdape: float | None
    test_coverage: float | None
    #: No later pass changes the row: its labels are in, or its requests rotated away.
    final: bool
    computed_at: str


#: Every column of `windows`, in `Window`'s order: the class's own field names.
_SELECT_WINDOWS = f"SELECT {', '.join(Window.__dataclass_fields__)} FROM windows"  # noqa: S608
_PERFORMANCE_COLUMNS = ", ".join(Performance.__dataclass_fields__)
_SELECT_PERFORMANCE = f"SELECT {_PERFORMANCE_COLUMNS} FROM window_performance"  # noqa: S608
_TEST_VALUES_COLUMNS = ", ".join(ModelTestValues.__dataclass_fields__)


class Store:
    """`monitor.db`, opened read-write; `open` creates the tables of a new file."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path) -> "Store":
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        store = cls(connection)
        with store.transaction():
            (version,) = connection.execute("PRAGMA user_version").fetchone()
            if version not in (0, SCHEMA_VERSION):
                raise StoreError(f"{path} has schema version {version}, not {SCHEMA_VERSION}")
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return store

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def last_closed_seq(self) -> int:
        """The last `seq` of the newest closed window, 0 before the first."""
        (value,) = self._connection.execute(
            "SELECT coalesce(max(last_seq), 0) FROM windows"
        ).fetchone()
        return int(value)

    def add_window(
        self,
        window: dict[str, Any],
        requests: Sequence[Request],
        features: Sequence[FeatureResult],
    ) -> int:
        """Store a closed window with its requests and, when tested, its features."""
        with self.transaction() as connection:
            columns = ", ".join(window)
            marks = ", ".join(f":{name}" for name in window)
            # The column names are the job's own keys; every value is bound.
            insert = f"INSERT INTO windows ({columns}) VALUES ({marks})"  # noqa: S608
            cursor = connection.execute(insert, window)
            window_id = cursor.lastrowid
            if window_id is None:
                raise StoreError("SQLite gave the new window no id")
            connection.executemany(
                "INSERT INTO window_requests (seq, window_id, request_id, endpoint) "
                "VALUES (?, ?, ?, ?)",
                [(r.seq, window_id, r.request_id, r.endpoint) for r in requests],
            )
            connection.executemany(
                "INSERT INTO window_features (window_id, feature, test, p_value, statistic, "
                "values_compared, drifted) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        window_id,
                        f.feature,
                        f.test,
                        f.p_value,
                        f.statistic,
                        f.n_values,
                        int(f.drifted),
                    )
                    for f in features
                ],
            )
        return window_id

    def drop_requests_before(self, seq: int) -> int:
        """Drop the requests below `seq`, the log's first; returns how many.

        The windows and their results stay: they are the evidence, and small.
        """
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM window_requests WHERE seq < ?", (seq,))
        return cursor.rowcount

    def set_open_window(self, progress: OpenWindow) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO open_window (singleton, requests, window_size, first_seq, "
                "last_seq, model_version, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?)",
                (
                    progress.requests,
                    progress.window_size,
                    progress.first_seq,
                    progress.last_seq,
                    progress.model_version,
                    progress.updated_at,
                ),
            )

    def add_run(
        self, started: datetime, finished: datetime, ok: bool, closed: int, detail: str = ""
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO runs (started_at, finished_at, status, windows_closed, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    timestamp(started),
                    timestamp(finished),
                    "ok" if ok else "failed",
                    closed,
                    detail,
                ),
            )
            connection.execute(
                "DELETE FROM runs WHERE run_id <= (SELECT max(run_id) FROM runs) - ?",
                (RUNS_KEPT,),
            )

    def windows(self) -> list[Window]:
        rows = self._connection.execute(f"{_SELECT_WINDOWS} ORDER BY window_id").fetchall()
        return [_window(row) for row in rows]

    def window(self, window_id: int) -> Window | None:
        row = self._connection.execute(
            f"{_SELECT_WINDOWS} WHERE window_id = ?", (window_id,)
        ).fetchone()
        return None if row is None else _window(row)

    def latest_tested(self) -> Window | None:
        row = self._connection.execute(
            f"{_SELECT_WINDOWS} WHERE status = 'tested' ORDER BY window_id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else _window(row)

    def features(self, window_id: int) -> list[FeatureResult]:
        rows = self._connection.execute(
            "SELECT feature, test, p_value, statistic, values_compared FROM window_features "
            "WHERE window_id = ? ORDER BY rowid",
            (window_id,),
        ).fetchall()
        return [FeatureResult(*row) for row in rows]

    def requests(self, window_id: int) -> list[Request]:
        rows = self._connection.execute(
            "SELECT seq, request_id, endpoint FROM window_requests "
            "WHERE window_id = ? ORDER BY seq",
            (window_id,),
        ).fetchall()
        return [Request(*row) for row in rows]

    def unfinished_windows(self) -> list[Window]:
        """The closed windows whose error and coverage a pass computes: none yet, or not final."""
        rows = self._connection.execute(
            f"{_SELECT_WINDOWS} WHERE window_id NOT IN "  # noqa: S608 - no value in the text
            "(SELECT window_id FROM window_performance WHERE final = 1) ORDER BY window_id"
        ).fetchall()
        return [_window(row) for row in rows]

    def set_performance(self, performance: Performance) -> None:
        values = {**performance.__dict__, "final": int(performance.final)}
        marks = ", ".join(f":{name}" for name in values)
        with self.transaction() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO window_performance ({_PERFORMANCE_COLUMNS}) "  # noqa: S608
                f"VALUES ({marks})",
                values,
            )

    def performance(self, window_id: int) -> Performance | None:
        row = self._connection.execute(
            f"{_SELECT_PERFORMANCE} WHERE window_id = ?", (window_id,)
        ).fetchone()
        return None if row is None else _performance(row)

    def performances(self) -> dict[int, Performance]:
        rows = self._connection.execute(f"{_SELECT_PERFORMANCE} ORDER BY window_id").fetchall()
        return {performance.window_id: performance for performance in map(_performance, rows)}

    def record_test_values(self, values: Sequence[ModelTestValues]) -> None:
        """Record each model version's test values, unless a pass recorded that version's before.

        The first record stays, so a window's comparison never changes once made.
        """
        marks = ", ".join("?" for _ in ModelTestValues.__dataclass_fields__)
        with self.transaction() as connection:
            connection.executemany(
                f"INSERT OR IGNORE INTO model_test_values ({_TEST_VALUES_COLUMNS}) "  # noqa: S608
                f"VALUES ({marks})",
                [tuple(value.__dict__.values()) for value in values],
            )

    def test_values(self, model_variant: str, model_version: str) -> ModelTestValues | None:
        """The test values recorded for one model version, None if no pass recorded them."""
        row = self._connection.execute(
            f"SELECT {_TEST_VALUES_COLUMNS} FROM model_test_values "  # noqa: S608
            "WHERE model_variant = ? AND model_version = ?",
            (model_variant, model_version),
        ).fetchone()
        return None if row is None else ModelTestValues(*row)

    def open_window(self) -> OpenWindow | None:
        row = self._connection.execute(
            "SELECT requests, window_size, first_seq, last_seq, model_version, updated_at "
            "FROM open_window"
        ).fetchone()
        return None if row is None else OpenWindow(*row)

    def last_success(self) -> str | None:
        (value,) = self._connection.execute(
            "SELECT max(finished_at) FROM runs WHERE status = 'ok'"
        ).fetchone()
        return value


def _window(row: Sequence[Any]) -> Window:
    window = Window(*row)
    if window.flagged is None:
        return window
    return Window(**{**window.__dict__, "flagged": bool(window.flagged)})


def _performance(row: Sequence[Any]) -> Performance:
    performance = Performance(*row)
    return Performance(**{**performance.__dict__, "final": bool(performance.final)})
