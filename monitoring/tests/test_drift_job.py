"""The drift job cuts the log into windows, stores each once, and exports the latest (#228).

The log is a fixture prediction log with the API's columns (samples.py), the
reference the six control listings of tests/fixtures/ ten times over, and the
windows hold a few requests, so that every case runs in well under a second
apart from Alibi Detect's import.
"""

from datetime import UTC, datetime, timedelta
import fcntl
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import pytest
from samples import PredictionLog, fixture_bodies, write_fixture_samples

from recommenditos_monitoring import __main__ as cli
from recommenditos_monitoring.drift import (
    CHI_SQUARE,
    KS,
    THRESHOLD,
    DriftTest,
    SamplesError,
    read_reference,
)
from recommenditos_monitoring.drift_features import DRIFT_FEATURES, NUMERIC
from recommenditos_monitoring.job import JobError, Paths, finding, run_once, window_features
from recommenditos_monitoring.log import REQUIRED_FIELDS, connect_read_only, rows_after
from recommenditos_monitoring.store import Store
from recommenditos_monitoring.textfile import FILE_NAME

#: 11 whole months after the samples' reference date, 2025-11-08.
START = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
STEP = timedelta(seconds=1)
WINDOW = 4


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    textfile = tmp_path / "textfile"
    textfile.mkdir()
    return Paths(
        log=tmp_path / "prediction-log.sqlite3",
        db=tmp_path / "results" / "monitor.db",
        data_dir=write_fixture_samples(tmp_path / "samples"),
        textfile_dir=textfile,
    )


@pytest.fixture
def log(paths: Paths):
    log = PredictionLog(paths.log)
    yield log
    log.close()


def control(n: int) -> list[dict[str, Any]]:
    bodies = fixture_bodies("control")
    return [bodies[i % len(bodies)] for i in range(n)]


def es(n: int) -> list[dict[str, Any]]:
    bodies = fixture_bodies("es")
    return [bodies[i % len(bodies)] for i in range(n)]


def windows(paths: Paths) -> list[tuple[Any, ...]]:
    with sqlite3.connect(paths.db) as connection:
        return connection.execute(
            "SELECT window_id, status, first_seq, last_seq, requests, flagged FROM windows "
            "ORDER BY window_id"
        ).fetchall()


@pytest.mark.req("FR-14")
def test_a_pass_tests_each_full_window_and_keeps_the_rest_open(paths, log):
    seqs = log.add_many(control(2 * WINDOW + 3), START, STEP)

    outcome = run_once(paths, window_size=WINDOW)

    assert [w.status for w in outcome.closed] == ["tested", "tested"]
    assert outcome.open_requests == 3
    assert [(w[2], w[3], w[4]) for w in windows(paths)] == [
        (seqs[0], seqs[WINDOW - 1], WINDOW),
        (seqs[WINDOW], seqs[2 * WINDOW - 1], WINDOW),
    ]
    store = Store.open(paths.db)
    progress = store.open_window()
    assert progress is not None
    assert (progress.requests, progress.window_size) == (3, WINDOW)
    assert (progress.first_seq, progress.last_seq) == (seqs[2 * WINDOW], seqs[-1])
    first = store.windows()[0]
    assert first.first_answered_at == START.isoformat(timespec="milliseconds")
    assert first.last_answered_at == (START + 3 * STEP).isoformat(timespec="milliseconds")
    assert [r.seq for r in store.requests(first.window_id)] == seqs[:WINDOW]


@pytest.mark.req("FR-14")
def test_each_tested_window_stores_every_feature_with_its_test_and_p_value(paths, log):
    log.add_many(es(WINDOW), START, STEP)

    run_once(paths, window_size=WINDOW)

    store = Store.open(paths.db)
    window = store.windows()[0]
    features = store.features(window.window_id)
    assert [f.feature for f in features] == list(DRIFT_FEATURES)
    assert {f.test for f in features} == {KS, CHI_SQUARE}
    assert all(0.0 <= f.p_value <= 1.0 for f in features)
    assert window.drifted_features == sum(f.drifted for f in features)
    assert window.threshold == pytest.approx(0.005 / 23)
    assert window.es_share == 1.0
    # The fixture reference has no ES listing, so the market alone drifts.
    assert window.flagged is True
    assert "country_code" in {f.feature for f in features if f.drifted}


def test_a_second_pass_stores_no_window_twice(paths, log):
    log.add_many(control(WINDOW + 1), START, STEP)
    run_once(paths, window_size=WINDOW)

    again = run_once(paths, window_size=WINDOW)

    assert again.closed == []
    assert again.open_requests == 1
    assert len(windows(paths)) == 1
    log.add_many(control(WINDOW - 1), START + 10 * STEP, STEP)
    third = run_once(paths, window_size=WINDOW)
    assert [w.first_seq for w in third.closed] == [WINDOW + 1]
    assert len(windows(paths)) == 2
    with sqlite3.connect(paths.db) as connection:
        assert connection.execute("SELECT count(*) FROM window_requests").fetchone() == (8,)
        assert connection.execute("SELECT count(*) FROM runs WHERE status = 'ok'").fetchone() == (
            3,
        )


def test_a_new_model_version_closes_the_open_window_as_partial(paths, log):
    first = log.add_many(control(WINDOW + 2), START, STEP, model_version="v1")
    second = log.add_many(control(WINDOW), START + 60 * STEP, STEP, model_version="v2")

    outcome = run_once(paths, window_size=WINDOW)

    assert [(w.status, w.first_seq, w.last_seq, w.model_version) for w in outcome.closed] == [
        ("tested", first[0], first[WINDOW - 1], "v1"),
        ("partial", first[WINDOW], first[-1], "v1"),
        ("tested", second[0], second[-1], "v2"),
    ]
    partial = outcome.closed[1]
    assert (partial.requests, partial.flagged, partial.drifted_features) == (2, None, None)
    store = Store.open(paths.db)
    assert store.features(partial.window_id) == []
    assert "no drift test ran" in finding(partial, [])


@pytest.mark.parametrize(
    ("change", "kwargs"),
    [
        ("an older row, with defaults in inputs", {"inputs_sent_only": None}),
        ("the load test", {"client": "loadtest"}),
        ("the smoke check", {"client": "smoke"}),
        ("the web page", {"client": "web-ui"}),
        ("an unpriced answer", {"endpoint": "/v1/comparables"}),
    ],
)
def test_test_traffic_and_old_rows_are_left_out(paths, log, change, kwargs):
    log.add_many(control(WINDOW), START, STEP, **kwargs)
    kept = log.add_many(control(1), START + 10 * STEP, STEP)

    outcome = run_once(paths, window_size=WINDOW)

    assert outcome.closed == [], f"{change} was compared"
    assert outcome.open_requests == 1
    store = Store.open(paths.db)
    progress = store.open_window()
    assert progress is not None and progress.first_seq == kept[0]


def test_a_range_request_without_every_required_field_is_left_out(paths, log):
    partial_body = {"make": "BMW", "registration_date": "2020-01", "country_code": "DE"}
    log.add_many([partial_body] * WINDOW, START, STEP, endpoint="/v1/price-range")
    log.add_many(control(1), START + 10 * STEP, STEP, endpoint="/v1/price-range")

    outcome = run_once(paths, window_size=WINDOW)

    assert (outcome.closed, outcome.open_requests) == ([], 1)
    assert len(REQUIRED_FIELDS) == 10


def test_the_age_is_computed_at_the_whole_months_of_the_answer_time(paths, log):
    """The replay shifts a listing by the months to its send time, the job takes them off."""
    body = dict(control(1)[0])
    # Sent and answered 12 months after the reference date: shifted by 12 months.
    year, month = (int(part) for part in body["registration_date"].split("-"))
    body["registration_date"] = f"{year + 1:04d}-{month:02d}"
    log.add(body, datetime(2026, 11, 8, 0, 0, 1, tzinfo=UTC))
    reference, reference_date = read_reference(paths.data_dir)

    connection = connect_read_only(paths.log)
    (row,) = list(rows_after(connection, 0))
    connection.close()
    age = window_features([row], reference_date)["age_years"].iloc[0]
    assert age == pytest.approx(reference["age_years"].iloc[0], abs=4 / 365.25)


def test_the_job_reads_the_log_without_writing_it(paths, log):
    log.add_many(control(WINDOW), START, STEP)
    before = paths.log.read_bytes()

    run_once(paths, window_size=WINDOW)

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        connect_read_only(paths.log).execute("DELETE FROM predictions")
    assert paths.log.read_bytes() == before


def test_a_log_that_restarted_below_the_last_window_fails_the_pass(paths, log, tmp_path):
    log.add_many(control(WINDOW), START, STEP)
    run_once(paths, window_size=WINDOW)
    log.close()
    paths.log.unlink()
    fresh = PredictionLog(paths.log)
    fresh.add_many(control(1), START, STEP)
    fresh.close()

    with pytest.raises(JobError, match="new log"):
        run_once(paths, window_size=WINDOW)
    with sqlite3.connect(paths.db) as connection:
        assert connection.execute("SELECT status FROM runs ORDER BY run_id").fetchall() == [
            ("ok",),
            ("failed",),
        ]


def test_no_log_fails_the_pass_with_its_cause(paths, capsys):
    code = cli.main(
        [
            "run-once",
            "--log",
            str(paths.log),
            "--db",
            str(paths.db),
            "--data-dir",
            str(paths.data_dir),
        ]
    )

    assert code == 1
    assert "there is no prediction log" in capsys.readouterr().err


def test_a_reference_of_other_features_is_refused(paths):
    manifest = json.loads((paths.data_dir / "manifest.json").read_text())
    manifest["features"]["categorical"].append("model")
    (paths.data_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SamplesError, match="other drift features"):
        read_reference(paths.data_dir)


def test_a_constant_feature_never_drifts():
    """`age_years__missing` and two more are `present` in every accepted request."""
    reference = pd.DataFrame({name: ["present"] * 50 for name in DRIFT_FEATURES})
    for name in NUMERIC:
        reference[name] = np.linspace(1, 10, 50)
    test = DriftTest(reference)

    results = {r.feature: r for r in test.run(reference.head(20))}

    constant = results["age_years__missing"]
    assert (constant.test, constant.p_value, constant.drifted) == (CHI_SQUARE, 1.0, False)


def test_a_numeric_feature_is_compared_on_its_sent_values_only():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({name: ["x"] * 400 for name in DRIFT_FEATURES})
    for name in NUMERIC:
        reference[name] = rng.normal(6, 1, 400)
    window = reference.head(200).copy()
    # Half the window did not send `gears`: the values that were sent are alike.
    window.loc[window.index[:100], "gears"] = np.nan

    result = {r.feature: r for r in DriftTest(reference).run(window)}["gears"]

    assert (result.test, result.n_values) == (KS, 100)
    assert result.p_value > THRESHOLD


def test_the_pass_writes_the_latest_tested_window_for_node_exporter(paths, log):
    log.add_many(control(WINDOW), START, STEP)
    log.add_many(es(WINDOW), START + 10 * STEP, STEP)
    log.add_many(control(1), START + 20 * STEP, STEP)

    run_once(paths, window_size=WINDOW, now=lambda: datetime(2026, 11, 9, 13, tzinfo=UTC))

    assert paths.textfile_dir is not None
    files = sorted(p.name for p in paths.textfile_dir.iterdir())
    assert files == [FILE_NAME], "a temporary file was left behind"
    prom = paths.textfile_dir / FILE_NAME
    assert prom.stat().st_mode & 0o777 == 0o644
    series = dict(
        line.rsplit(" ", 1) for line in prom.read_text().splitlines() if not line.startswith("#")
    )
    assert series["recommenditos_drift_window_id"] == "2.0"
    assert series["recommenditos_drift_window_flagged"] == "1.0"
    assert series["recommenditos_drift_window_es_share"] == "1.0"
    assert series["recommenditos_drift_open_window_requests"] == "1.0"
    assert series["recommenditos_drift_window_size"] == str(float(WINDOW))
    end = START + 10 * STEP + (WINDOW - 1) * STEP
    assert float(series["recommenditos_drift_window_end_timestamp_seconds"]) == pytest.approx(
        end.timestamp()
    )
    assert float(series["recommenditos_monitor_last_success_timestamp_seconds"]) == (
        datetime(2026, 11, 9, 13, tzinfo=UTC).timestamp()
    )
    p_values = [name for name in series if name.startswith("recommenditos_drift_feature_p_value")]
    assert len(p_values) == len(DRIFT_FEATURES)
    assert 'recommenditos_drift_feature_p_value{feature="country_code",test="chi2"}' in series
    assert len(series) == len(DRIFT_FEATURES) + 10


def test_a_failed_pass_leaves_the_last_metrics_in_place(paths, log):
    log.add_many(control(WINDOW), START, STEP)
    run_once(paths, window_size=WINDOW)
    assert paths.textfile_dir is not None
    before = (paths.textfile_dir / FILE_NAME).read_text()
    (paths.data_dir / "reference.parquet").unlink()

    with pytest.raises(FileNotFoundError):
        run_once(paths, window_size=WINDOW)

    assert (paths.textfile_dir / FILE_NAME).read_text() == before


def test_a_second_pass_while_one_runs_leaves_it_alone(paths, log):
    log.add_many(control(WINDOW), START, STEP)
    paths.db.parent.mkdir(parents=True)
    with (paths.db.parent / "monitor.db.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        outcome = run_once(paths, window_size=WINDOW)
    assert outcome.skipped
    assert not paths.db.exists()


@pytest.mark.req("FR-14")
def test_the_report_names_the_drifted_features_and_exports_the_evidence(paths, log, capsys):
    log.add_many(es(WINDOW), START, STEP)
    run_once(paths, window_size=WINDOW)
    capsys.readouterr()

    assert cli.main(["report", "--db", str(paths.db)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("Input drift in window 1 (log seq 1 to 4,")
    assert "country_code (p " in text
    assert "100% ES" in text

    assert cli.main(["report", "--db", str(paths.db), "--window", "1", "--json"]) == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["window"]["window_id"] == 1
    assert [f["feature"] for f in evidence["features"]] == list(DRIFT_FEATURES)
    assert {"p_value", "test", "statistic", "n_values", "drifted"} <= set(evidence["features"][0])
    assert evidence["finding"] == text.strip()


def test_the_report_of_an_unknown_window_fails(paths, log, capsys):
    log.add_many(control(WINDOW), START, STEP)
    run_once(paths, window_size=WINDOW)

    assert cli.main(["report", "--db", str(paths.db), "--window", "7"]) == 1
    assert "no window 7" in capsys.readouterr().err


def test_run_once_prints_what_it_closed(paths, log, capsys):
    log.add_many(control(WINDOW + 1), START, STEP)

    code = cli.main(
        [
            "run-once",
            "--log",
            str(paths.log),
            "--db",
            str(paths.db),
            "--data-dir",
            str(paths.data_dir),
            "--window-size",
            str(WINDOW),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "tested window 1: seq 1-4" in out
    assert f"open window: 1 of {WINDOW} requests" in out
