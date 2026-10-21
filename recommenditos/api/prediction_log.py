"""The prediction log and the feedback: one record per answer, one observed price per record.

Every answer of a valuation endpoint is one row of `predictions` (FR-13), and
`POST /v1/feedback` stores an observed price against its request ID in
`feedback` (FR-10). M6's drift job joins the two by request ID (FR-14); the view
`labelled_predictions` is that join.

Both live in one SQLite file in WAL mode on a volume the drift job can read
(EDN-86). `PredictionStore` is the interface the API uses, so a PostgreSQL store
would be a second implementation of it and a change in `create_app`, nothing
more. `SqliteStore` implements it:

- **Exactly once (NFR-03).** `request_id` is unique, so a second record of one
  answer fails instead of counting twice. Several worker processes write the
  same file: SQLite serialises the writers with a file lock, and a writer that
  finds it held waits up to `BUSY_TIMEOUT_S` rather than failing.
- **Rotation (NFR-04).** The live data is kept under `max_bytes`, oldest
  answers first, together with their feedback. Every `PRUNE_EVERY` records, and
  when the store opens, a background thread of the process compares the pages
  in use with the cap and deletes the oldest answers in batches of
  `PRUNE_BATCH` until the data fits 90 % of it. It takes the lock per batch,
  so a request never waits for more than one batch, and none runs the pruning
  itself. The
  file does not shrink, because SQLite reuses the freed pages for new rows
  instead, so it stays at about the cap; `PRAGMA journal_size_limit` keeps the
  WAL file small after each checkpoint.
- **No PII (NFR-08).** A row holds the validated request body, which has no
  PII field, and no column for the caller's address.

`PredictionLog`, a dependency of `recommenditos.api.dependencies`, is what a
route uses: `log.record(request, answer)` writes the answer's row once.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import enum
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel

from recommenditos.api.schemas.common import FieldWarning, ModelInfo

#: The layout of the file; a file with another one stops the service at startup.
SCHEMA_VERSION = 1

#: How long a writer waits for another process's write before it gives up.
BUSY_TIMEOUT_S = 5.0

#: Records a process writes between two checks of the size cap.
PRUNE_EVERY = 1000

#: The share of the cap the pruning brings the live data down to.
PRUNE_TARGET = 0.9

#: Answers deleted per transaction while pruning, so that a writer waits for one
#: short batch, not for the whole pruning (docs/docs/api.md#rotation).
PRUNE_BATCH = 1000

#: The pause between two batches. Python's lock is not fair, so a pruner that
#: took it again at once could keep a waiting request out for the whole pruning;
#: the pause lets it in, and a writer in another process too.
PRUNE_PAUSE_S = 0.05

#: The size the WAL file is truncated to after a checkpoint.
WAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024

_SCHEMA = """
CREATE TABLE predictions (
    seq           INTEGER PRIMARY KEY,
    request_id    TEXT NOT NULL UNIQUE,
    endpoint      TEXT NOT NULL,
    answered_at   TEXT NOT NULL,
    latency_ms    REAL NOT NULL,
    model_variant TEXT NOT NULL,
    model_version TEXT NOT NULL,
    inputs        TEXT NOT NULL,
    outputs       TEXT NOT NULL,
    warnings      TEXT NOT NULL
);
CREATE TABLE feedback (
    request_id         TEXT PRIMARY KEY
                       REFERENCES predictions (request_id) ON DELETE CASCADE,
    observed_price_eur REAL NOT NULL,
    received_at        TEXT NOT NULL
);
CREATE VIEW labelled_predictions AS
    SELECT predictions.*, feedback.observed_price_eur, feedback.received_at
    FROM predictions JOIN feedback USING (request_id);
"""


class StoreError(Exception):
    """The store could not write; the cause is chained."""


@dataclass(frozen=True)
class AnswerRecord:
    """One answer, as the prediction log holds it (FR-13)."""

    request_id: str
    endpoint: str
    answered_at: datetime
    latency_ms: float
    model_variant: str
    model_version: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    warnings: list[dict[str, Any]]


@dataclass(frozen=True)
class Feedback:
    """An observed price, as the feedback holds it (FR-10)."""

    request_id: str
    observed_price_eur: float
    received_at: datetime


class FeedbackOutcome(enum.Enum):
    #: The price is stored.
    STORED = "stored"
    #: The same price was stored for this request before; nothing changed.
    REPEATED = "repeated"
    #: Another price is stored for this request; nothing changed.
    CONFLICT = "conflict"
    #: No answer with this request ID is in the log; nothing is stored.
    UNKNOWN_REQUEST = "unknown_request"


@dataclass(frozen=True)
class FeedbackResult:
    outcome: FeedbackOutcome
    #: What the store holds for the request after the call; None for an unknown one.
    stored: Feedback | None


class PredictionStore(Protocol):
    """Where the prediction log and the feedback are written."""

    def record_answer(self, record: AnswerRecord) -> None:
        """Write one answer; raises `StoreError` if it cannot, also for a second record."""
        ...

    def record_feedback(self, feedback: Feedback) -> FeedbackResult:
        """Store an observed price unless the request is unknown or already has one."""
        ...

    def close(self) -> None: ...


class SqliteStore:
    """`PredictionStore` on one SQLite file in WAL mode, shared by every worker process."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection per process. FastAPI runs a route in a thread pool, so
        # the connection is shared between threads and `_lock` lets one thread
        # use it at a time; a statement takes well under a millisecond.
        self._connection = sqlite3.connect(
            path, timeout=BUSY_TIMEOUT_S, isolation_level=None, check_same_thread=False
        )
        self._lock = threading.Lock()
        self._since_prune = 0
        self._configure()
        # The pruning runs off the request path, in one thread per process, and
        # starts with a check at startup.
        self._prune_wanted = threading.Event()
        self._closing = False
        self._pruner = threading.Thread(
            target=self._prune_when_asked, name="prediction-log-pruner", daemon=True
        )
        self._pruner.start()
        self._prune_wanted.set()

    def _configure(self) -> None:
        with self._lock:
            connection = self._connection
            # WAL lets the drift job read while the API writes. The mode is stored
            # in the file; NORMAL syncs at checkpoints rather than every commit,
            # which WAL keeps consistent and which a lost last answer on a power
            # cut is worth.
            _when_unlocked(lambda: connection.execute("PRAGMA journal_mode = WAL"))
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA journal_size_limit = {WAL_SIZE_LIMIT_BYTES}")
            with self._transaction():
                (version,) = connection.execute("PRAGMA user_version").fetchone()
                if version == 0:
                    for statement in _SCHEMA.split(";"):
                        if statement.strip():
                            connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION:
                    raise StoreError(
                        f"{self.path} has schema version {version}; this service writes "
                        f"version {SCHEMA_VERSION}."
                    )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """A write transaction that takes the file's write lock up front.

        `BEGIN IMMEDIATE` makes a second writer wait for the lock at the start,
        where SQLite's busy timeout applies, instead of failing on the upgrade
        from a read to a write halfway through.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def record_answer(self, record: AnswerRecord) -> None:
        try:
            with self._lock, self._transaction() as connection:
                connection.execute(
                    "INSERT INTO predictions (request_id, endpoint, answered_at, latency_ms, "
                    "model_variant, model_version, inputs, outputs, warnings) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.request_id,
                        record.endpoint,
                        _timestamp(record.answered_at),
                        record.latency_ms,
                        record.model_variant,
                        record.model_version,
                        _json(record.inputs),
                        _json(record.outputs),
                        _json(record.warnings),
                    ),
                )
        except (sqlite3.Error, ValueError) as error:
            raise StoreError(f"could not record answer {record.request_id}: {error}") from error
        self._since_prune += 1
        if self._since_prune >= PRUNE_EVERY:
            self._since_prune = 0
            self._prune_wanted.set()

    def _prune_when_asked(self) -> None:
        while True:
            self._prune_wanted.wait()
            self._prune_wanted.clear()
            if self._closing:
                return
            self.prune()

    def record_feedback(self, feedback: Feedback) -> FeedbackResult:
        try:
            with self._lock, self._transaction() as connection:
                known = connection.execute(
                    "SELECT 1 FROM predictions WHERE request_id = ?", (feedback.request_id,)
                ).fetchone()
                if known is None:
                    return FeedbackResult(FeedbackOutcome.UNKNOWN_REQUEST, None)
                row = connection.execute(
                    "SELECT observed_price_eur, received_at FROM feedback WHERE request_id = ?",
                    (feedback.request_id,),
                ).fetchone()
                if row is not None:
                    stored = Feedback(feedback.request_id, row[0], datetime.fromisoformat(row[1]))
                    outcome = (
                        FeedbackOutcome.REPEATED
                        if stored.observed_price_eur == feedback.observed_price_eur
                        else FeedbackOutcome.CONFLICT
                    )
                    return FeedbackResult(outcome, stored)
                # What the row holds, to the millisecond, so that a repeat reads the same.
                stored = Feedback(
                    feedback.request_id,
                    feedback.observed_price_eur,
                    datetime.fromisoformat(_timestamp(feedback.received_at)),
                )
                connection.execute(
                    "INSERT INTO feedback (request_id, observed_price_eur, received_at) "
                    "VALUES (?, ?, ?)",
                    (stored.request_id, stored.observed_price_eur, _timestamp(stored.received_at)),
                )
                return FeedbackResult(FeedbackOutcome.STORED, stored)
        except sqlite3.Error as error:
            raise StoreError(f"could not record feedback for {feedback.request_id}") from error

    def used_bytes(self) -> int:
        """The bytes the live data takes: the file's pages minus the free ones."""
        connection = self._connection
        (page_size,) = connection.execute("PRAGMA page_size").fetchone()
        (pages,) = connection.execute("PRAGMA page_count").fetchone()
        (free,) = connection.execute("PRAGMA freelist_count").fetchone()
        return int((pages - free) * page_size)

    def prune(self) -> int:
        """Delete the oldest answers and their feedback until the data fits; returns how many.

        Nothing is deleted while the live data is under `max_bytes`. Each batch
        takes the lock on its own, so a write waits for one batch at most. A
        failure is logged and leaves the rest to the next check.
        """
        deleted = 0
        try:
            target = self.max_bytes * PRUNE_TARGET
            limit: float = self.max_bytes
            while not self._closing:
                removed = self._delete_oldest(limit)
                if not removed:
                    break
                deleted += removed
                limit = target
                time.sleep(PRUNE_PAUSE_S)
        except sqlite3.Error as error:
            logger.opt(exception=error).error(f"pruning {self.path} failed")
        if deleted:
            logger.info(f"pruned the {deleted} oldest answers from {self.path}")
        return deleted

    def _delete_oldest(self, limit: float) -> int:
        """One batch: the oldest `PRUNE_BATCH` answers, if the data is over `limit`."""
        with self._lock:
            if self.used_bytes() <= limit:
                return 0
            with self._transaction() as connection:
                return int(
                    connection.execute(
                        "DELETE FROM predictions WHERE seq IN "
                        "(SELECT seq FROM predictions ORDER BY seq LIMIT ?)",
                        (PRUNE_BATCH,),
                    ).rowcount
                )

    def close(self) -> None:
        self._closing = True
        self._prune_wanted.set()
        self._pruner.join()
        with self._lock:
            self._connection.close()


def _when_unlocked[T](action: Callable[[], T]) -> T:
    """`action()`, retried while another process holds the lock, for up to `BUSY_TIMEOUT_S`.

    Switching a new file to WAL needs it to itself, and SQLite answers that
    switch with "database is locked" at once, without its busy timeout, when a
    second worker opens the same new file at the same moment.
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_S
    while True:
        try:
            return action()
        except sqlite3.OperationalError as error:
            busy = error.sqlite_errorcode in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
            if not busy or time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def _timestamp(moment: datetime) -> str:
    """ISO 8601 in UTC, to the millisecond, so that the text sorts like the time."""
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    """Strict JSON: a NaN or an infinity is refused, not written as a bare token."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------
# What a route uses
# --------------------------------------------------------------------------


class Answer(Protocol):
    """What every valuation answer carries besides its own outputs (FR-06)."""

    @property
    def model(self) -> ModelInfo: ...

    @property
    def warnings(self) -> list[FieldWarning]: ...

    def model_dump(self, *, mode: str = ..., exclude: set[str] | None = ...) -> dict[str, Any]: ...


class AnswerLoggedTwice(RuntimeError):
    """A route recorded the same answer twice: a bug in the route."""


class PredictionLog:
    """The prediction log as one request sees it: `record` writes its answer, once."""

    def __init__(
        self, store: PredictionStore, request_id: str, endpoint: str, started: float
    ) -> None:
        self._store = store
        self._request_id = request_id
        self._endpoint = endpoint
        self._started = started
        self._recorded = False

    def record(self, request: BaseModel, answer: Answer) -> None:
        """Write the row for `answer` to the validated `request`.

        Called by the route right before it returns the answer, so that the
        client gets no answer the log does not hold (NFR-03). Raises `StoreError`
        when the store cannot write; the route lets it through, and the client
        gets a 503 instead of the answer.
        """
        if self._recorded:
            raise AnswerLoggedTwice(f"request {self._request_id} was recorded twice")
        self._recorded = True
        self._store.record_answer(
            AnswerRecord(
                request_id=self._request_id,
                endpoint=self._endpoint,
                answered_at=datetime.now(UTC),
                latency_ms=(time.perf_counter() - self._started) * 1000,
                model_variant=answer.model.variant,
                model_version=answer.model.version,
                inputs=request.model_dump(mode="json", exclude_none=True),
                outputs=answer.model_dump(
                    mode="json", exclude={"model", "request_id", "warnings"}
                ),
                warnings=[warning.model_dump(mode="json") for warning in answer.warnings],
            )
        )
