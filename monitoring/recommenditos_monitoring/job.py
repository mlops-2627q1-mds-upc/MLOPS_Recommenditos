"""One pass of the drift job, and the report of a window (FR-14, NFR-11, #228).

`run_once` reads the prediction log from where the last closed window ended,
cuts the rows it compares (`log.py`) into windows of `window_size` consecutive
requests by `seq`, and for each full window

1. computes the 23 drift features of every request, its age at
   `as_of_for(answered_at)` (the rule in `drift_features.py`), so that a
   replayed listing is as old as it was in the training snapshot;
2. runs the drift test (`drift.py`) against the reference;
3. stores the window, its requests and every feature's result in `monitor.db`
   in one transaction.

A request answered by another `model_version` than the window's first closes
the window early as `partial`, untested, so that every window and its error
(#229) belong to one model. What is left after the last full window is the
open window: only its progress is stored, and the next pass reads its rows
again. A pass never stores a window twice, because it starts after the last
closed one; a second pass while one runs finds the lock taken and leaves.
The pass ends by writing the results for node-exporter (`textfile.py`).

A pass reads the log one row at a time and holds at most one window, so a
backlog of many windows costs time, not memory.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import json
from pathlib import Path
from typing import Any

import pandas as pd

from recommenditos_monitoring.drift import (
    ALPHA,
    THRESHOLD,
    DriftTest,
    FeatureResult,
    read_reference,
)
from recommenditos_monitoring.drift_features import DRIFT_FEATURES, as_of_for, drift_features
from recommenditos_monitoring.log import LogRow, connect_read_only, last_seq, rows_after
from recommenditos_monitoring.store import (
    PARTIAL,
    TESTED,
    OpenWindow,
    Request,
    Store,
    Window,
    timestamp,
)
from recommenditos_monitoring.textfile import drift_metrics, render, write_atomically

#: NFR-11's window (EDN-14).
WINDOW_SIZE = 1000


class JobError(Exception):
    """A pass cannot run."""


@dataclass(frozen=True)
class Paths:
    log: Path
    db: Path
    data_dir: Path
    #: node-exporter's textfile directory; None writes no metrics.
    textfile_dir: Path | None


@dataclass
class Outcome:
    closed: list[Window]
    open_requests: int
    skipped: bool = False


def window_features(rows: Sequence[LogRow], reference_date: str) -> pd.DataFrame:
    """The drift features of logged requests, each one's age at `as_of_for` its answer."""
    return drift_features(
        [row.inputs for row in rows],
        [as_of_for(row.answered_at, reference_date) for row in rows],
    )


def _es_share(rows: Sequence[LogRow]) -> float:
    return sum(row.inputs.get("country_code") == "ES" for row in rows) / len(rows)


@contextmanager
def _single_pass(db: Path) -> Iterator[bool]:
    """Hold the job's lock; yields False when another pass holds it."""
    db.parent.mkdir(parents=True, exist_ok=True)
    with (db.parent / f"{db.name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_once(
    paths: Paths,
    window_size: int = WINDOW_SIZE,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Outcome:
    """One pass: test every window the log completed since the last pass."""
    if window_size < 1:
        raise JobError("a window holds at least one request")
    started = now()
    with _single_pass(paths.db) as alone:
        if not alone:
            return Outcome(closed=[], open_requests=0, skipped=True)
        store = Store.open(paths.db)
        try:
            outcome = _pass(store, paths, window_size, now)
        except Exception as error:
            store.add_run(started, now(), ok=False, closed=0, detail=f"{error}")
            store.close()
            raise
        finished = now()
        store.add_run(started, finished, ok=True, closed=len(outcome.closed))
        if paths.textfile_dir is not None:
            write_atomically(paths.textfile_dir, render(drift_metrics(store, finished)))
        store.close()
    return outcome


def _pass(store: Store, paths: Paths, window_size: int, now: Callable[[], datetime]) -> Outcome:
    reference, reference_date = read_reference(paths.data_dir)
    test: DriftTest | None = None
    closed: list[Window] = []
    after = store.last_closed_seq()
    log = connect_read_only(paths.log)
    try:
        newest = last_seq(log)
        if newest < after:
            raise JobError(
                f"the prediction log ends at seq {newest}, before the last closed window "
                f"(seq {after}): is it a new log? Move monitor.db aside to start over"
            )
        current: list[LogRow] = []
        for row in rows_after(log, after):
            if not row.has_required_fields:
                continue
            if current and row.model_version != current[0].model_version:
                closed.append(_close(store, current, window_size, now()))
                current = []
            current.append(row)
            if len(current) == window_size:
                if test is None:
                    test = DriftTest(reference)
                features = test.run(window_features(current, reference_date))
                closed.append(_close(store, current, window_size, now(), (test, features)))
                current = []
    finally:
        log.close()
    store.set_open_window(
        OpenWindow(
            requests=len(current),
            window_size=window_size,
            first_seq=current[0].seq if current else None,
            last_seq=current[-1].seq if current else None,
            model_version=current[0].model_version if current else None,
            updated_at=timestamp(now()),
        )
    )
    return Outcome(closed=closed, open_requests=len(current))


def _close(
    store: Store,
    rows: Sequence[LogRow],
    window_size: int,
    closed_at: datetime,
    tested: tuple[DriftTest, list[FeatureResult]] | None = None,
) -> Window:
    """Store `rows` as a window: `tested` with its results, or `partial` without."""
    features = tested[1] if tested is not None else []
    window: dict[str, Any] = {
        "status": PARTIAL if tested is None else TESTED,
        "first_seq": rows[0].seq,
        "last_seq": rows[-1].seq,
        "requests": len(rows),
        "window_size": window_size,
        "first_answered_at": rows[0].answered_at,
        "last_answered_at": rows[-1].answered_at,
        "model_variant": rows[0].model_variant,
        "model_version": rows[0].model_version,
        "es_share": _es_share(rows),
        "flagged": None,
        "drifted_features": None,
        "threshold": None,
        "reference_rows": None,
        "closed_at": timestamp(closed_at),
    }
    if tested is not None:
        drifted = sum(result.drifted for result in features)
        window |= {
            "flagged": int(drifted > 0),
            "drifted_features": drifted,
            "threshold": THRESHOLD,
            "reference_rows": tested[0].reference_rows,
        }
    requests = [Request(row.seq, row.request_id, row.endpoint) for row in rows]
    window_id = store.add_window(window, requests, features)
    stored = store.window(window_id)
    if stored is None:
        raise JobError(f"window {window_id} was stored and cannot be read back")
    return stored


# --------------------------------------------------------------------------
# The report of one window
# --------------------------------------------------------------------------


def _p(value: float) -> str:
    return "0" if value == 0 else f"{value:.2g}"


def finding(window: Window, features: Sequence[FeatureResult]) -> str:
    """One paragraph on `window`, for retrain.yml's `reason` and the team channel."""
    where = (
        f"window {window.window_id} (log seq {window.first_seq} to {window.last_seq}, "
        f"{window.requests} requests answered {window.first_answered_at} to "
        f"{window.last_answered_at} by {window.model_variant} {window.model_version}, "
        f"{window.es_share:.0%} ES)"
    )
    if window.status == PARTIAL:
        return (
            f"Partial {where}: the model changed before the window was full, so no drift test ran."
        )
    drifted = sorted((f for f in features if f.drifted), key=lambda f: f.p_value)
    if not drifted:
        return (
            f"No drift in {where}: none of the {len(features)} features is below "
            f"p {THRESHOLD:.3g} ({ALPHA} / {len(DRIFT_FEATURES)}, NFR-11)."
        )
    named = ", ".join(f"{f.feature} (p {_p(f.p_value)})" for f in drifted)
    return (
        f"Input drift in {where}: {len(drifted)} of {len(features)} features are below "
        f"p {THRESHOLD:.3g} ({ALPHA} / {len(DRIFT_FEATURES)}, NFR-11): {named}."
    )


def evidence(window: Window, features: Sequence[FeatureResult]) -> str:
    """The window and every feature's result as JSON, the evidence for the report."""
    return json.dumps(
        {
            "window": window.__dict__,
            "alpha": ALPHA,
            "features": [{**result.__dict__, "drifted": result.drifted} for result in features],
            "finding": finding(window, features),
        },
        indent=2,
    )


def chosen_window(store: Store, window_id: int | None) -> Window:
    """Window `window_id`, or the latest tested one."""
    window = store.latest_tested() if window_id is None else store.window(window_id)
    if window is None:
        what = "no window has been tested yet" if window_id is None else f"no window {window_id}"
        raise JobError(what)
    return window
