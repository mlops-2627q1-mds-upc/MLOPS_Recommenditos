"""The real error and the coverage per window, from the feedback of its requests (#229).

The fixture log holds answers with known estimates, ranges and observed prices,
so every MdAPE and coverage below is the exact value the rows define. The
windows are large enough for the real label thresholds, 200 labelled answers
and 300 labelled ranges, which the tests cross one label at a time.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest
from samples import PredictionLog, fixture_bodies, write_fixture_samples

from recommenditos_monitoring import __main__ as cli
from recommenditos_monitoring import job
from recommenditos_monitoring.job import Paths, run_once
from recommenditos_monitoring.log import Label
from recommenditos_monitoring.performance import (
    LABEL_HORIZON,
    MIN_LABELLED,
    MIN_LABELLED_RANGES,
    NO_BASELINE,
    Baseline,
    baseline,
)
from recommenditos_monitoring.store import INSUFFICIENT_LABELS, OK, Performance, Store
from recommenditos_monitoring.textfile import FILE_NAME

START = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
STEP = timedelta(seconds=1)
#: The day after the samples' reference date, 2025-11-08: an age needs no shift.
SAME_MONTH = datetime(2025, 11, 9, 12, 0, tzinfo=UTC)
#: The repository's evaluate outputs, which CD bakes into the image.
REPOSITORY = Path(__file__).parents[2]
METRICS = REPOSITORY / "metrics.json"
COVERAGE = REPOSITORY / "reports" / "metrics" / "interval-coverage-by-make.csv"
#: The fixture's test values, not the repository's: a retrain changes those.
TEST_MDAPE, TEST_COVERAGE = 0.08, 0.88
OBSERVED = 10_000.0
PREDICT, RANGE = "/v1/predict", "/v1/price-range"


def write_test_values(directory: Path) -> tuple[Path, Path]:
    """A metrics.json and a coverage-by-make table in the evaluate stage's formats.

    The table's other rows, another make and the partial-input scenario SC-05
    judges, hold other values, so a lookup of the wrong row shows.
    """
    directory.mkdir(parents=True, exist_ok=True)
    metrics = directory / "metrics.json"
    metrics.write_text(
        json.dumps({"variants": {"lgbm-basic": {"mdape": TEST_MDAPE, "sc05_coverage": 0.95}}})
    )
    coverage = directory / "interval-coverage-by-make.csv"
    coverage.write_text(
        "variant,scenario,make,n,coverage,std_error,mean_relative_width,judgement\n"
        'lgbm-basic,"(P1: make, model, registration date, mileage)",(pooled),100,0.95,0.01,0.9,'
        "in_band\n"
        "lgbm-basic,full,BMW,50,0.81,0.01,0.4,in_band\n"
        f"lgbm-basic,full,(pooled),100,{TEST_COVERAGE},0.01,0.4,in_band\n"
    )
    return metrics, coverage


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    textfile = tmp_path / "textfile"
    textfile.mkdir()
    metrics, coverage = write_test_values(tmp_path / "evaluate")
    return Paths(
        log=tmp_path / "prediction-log.sqlite3",
        db=tmp_path / "results" / "monitor.db",
        data_dir=write_fixture_samples(tmp_path / "samples"),
        textfile_dir=textfile,
        metrics=metrics,
        coverage=coverage,
    )


@pytest.fixture
def log(paths: Paths):
    log = PredictionLog(paths.log)
    yield log
    log.close()


def body(i: int, sample: str = "control") -> dict[str, Any]:
    bodies = fixture_bodies(sample)
    return bodies[i % len(bodies)]


def answer(price: float, lower: float | None = None, upper: float | None = None) -> dict:
    if lower is None:
        return {"price_eur": price, "explanation": {}}
    return {"price_eur": price, "lower_eur": lower, "upper_eur": upper, "confidence_level": 0.9}


def add_answers(  # noqa: PLR0913 - one keyword per thing a test varies
    log: PredictionLog,
    n: int,
    *,
    endpoint: str,
    start: datetime,
    error: float = 0.0,
    inside: bool = True,
    sample: str = "control",
) -> list[int]:
    """`n` answers whose estimate is `error` off OBSERVED, and whose range holds it if `inside`."""
    price = OBSERVED * (1 + error)
    if endpoint == PREDICT:
        outputs = answer(price)
    elif inside:
        outputs = answer(price, OBSERVED - 1000, OBSERVED + 1000)
    else:
        outputs = answer(price, OBSERVED + 1, OBSERVED + 2000)
    return [
        log.add(body(i, sample), start + i * STEP, endpoint=endpoint, outputs=outputs)
        for i in range(n)
    ]


def label(log: PredictionLog, seqs: list[int], at: datetime = START) -> None:
    for seq in seqs:
        log.add_feedback(seq, OBSERVED, at)


def performance(paths: Paths, window_id: int = 1) -> Performance:
    store = Store.open(paths.db)
    try:
        found = store.performance(window_id)
    finally:
        store.close()
    assert found is not None, f"window {window_id} has no error and coverage"
    return found


def series(paths: Paths) -> dict[str, float]:
    assert paths.textfile_dir is not None
    lines = (paths.textfile_dir / FILE_NAME).read_text().splitlines()
    return {
        name: float(value)
        for name, value in (line.rsplit(" ", 1) for line in lines if not line.startswith("#"))
        if name.startswith("recommenditos_performance_")
    }


def bounds(i: int) -> tuple[float, float]:
    """The range of the i-th answer: 20 of 400 miss the price, two hold it on a bound."""
    if i % 20 == 1:
        return OBSERVED + 1, OBSERVED + 9
    if i == 0:
        return OBSERVED, OBSERVED + 50
    if i == 2:
        return OBSERVED - 50, OBSERVED
    return OBSERVED - 50, OBSERVED + 50


def test_the_mdape_and_the_coverage_are_the_exact_values_of_the_labels(paths, log):
    # 400 ranges whose estimates are 0, 1, ..., 399 EUR off 10,000, under and over.
    seqs = [
        log.add(
            body(i),
            START + i * STEP,
            endpoint=RANGE,
            outputs=answer(OBSERVED + (i if i % 2 else -i), *bounds(i)),
        )
        for i in range(400)
    ]
    # 400 point estimates 30 % off, and one 90 % off without feedback.
    seqs += add_answers(log, 400, endpoint=PREDICT, start=START + 400 * STEP, error=0.3)
    add_answers(log, 1, endpoint=PREDICT, start=START + 800 * STEP, error=0.9)
    label(log, seqs)

    run_once(paths, window_size=801, now=lambda: START + timedelta(hours=1))

    found = performance(paths)
    # The 800 errors, sorted: 0 % to 3.99 % in steps of 0.01 %, then 400 of 30 %;
    # the middle two, the 400th and the 401st, are 3.99 % and 30 %.
    assert found.mdape == pytest.approx((399 / OBSERVED + 0.3) / 2, abs=1e-12)
    assert (found.mdape_status, found.labelled, found.ranges, found.labelled_ranges) == (
        OK,
        800,
        400,
        400,
    )
    assert (found.coverage_status, found.coverage) == (OK, 380 / 400)
    assert found.nominal_level == pytest.approx(0.9)
    assert not found.final


def test_the_test_values_are_the_variants_full_input_ones(paths):
    assert baseline(paths.metrics, paths.coverage, "lgbm-basic") == Baseline(
        TEST_MDAPE, TEST_COVERAGE
    )
    assert baseline(paths.metrics, paths.coverage, "no-such-variant") == NO_BASELINE
    assert baseline(None, None, "lgbm-basic") == NO_BASELINE


def test_the_repositorys_evaluate_outputs_give_the_deployed_models_test_values():
    """The files the image copies, read as the evaluate stage's own record has them."""
    variant = json.loads(METRICS.read_text())["deployable_variant"]
    record = json.loads((COVERAGE.parent / f"{variant}.json").read_text())
    full = next(
        row
        for row in record["interval_by_make"]
        if (row["scenario"], row["make"]) == ("full", "(pooled)")
    )

    assert baseline(METRICS, COVERAGE, variant) == Baseline(record["mdape"], full["coverage"])


def test_too_few_labels_give_no_figure_and_a_late_label_lands_in_its_window(paths, log):
    ranges = add_answers(log, MIN_LABELLED_RANGES, endpoint=RANGE, start=START, error=0.1)
    points = add_answers(log, MIN_LABELLED, endpoint=PREDICT, start=START + 1000 * STEP, error=0.2)
    window = len(ranges) + len(points)
    # A later window, so that the first one is closed and labels arrive after.
    add_answers(log, 1, endpoint=PREDICT, start=START + 2000 * STEP)
    label(log, points[:-1])
    hour = START + timedelta(hours=1)

    run_once(paths, window_size=window, now=lambda: hour)

    found = performance(paths)
    assert (found.labelled, found.labelled_ranges) == (MIN_LABELLED - 1, 0)
    assert (found.mdape_status, found.mdape) == (INSUFFICIENT_LABELS, None)
    assert (found.coverage_status, found.coverage) == (INSUFFICIENT_LABELS, None)
    exported = series(paths)
    assert "recommenditos_performance_mdape" not in exported
    assert "recommenditos_performance_coverage" not in exported
    assert exported['recommenditos_performance_labels_sufficient{figure="mdape"}'] == 0
    assert exported["recommenditos_performance_labelled_requests"] == MIN_LABELLED - 1

    # The 200th label arrives after the window closed.
    label(log, points[-1:], hour)
    run_once(paths, window_size=window, now=lambda: hour)
    found = performance(paths)
    assert (found.labelled, found.mdape_status) == (MIN_LABELLED, OK)
    assert found.mdape == pytest.approx(0.2, abs=1e-12)
    assert found.coverage_status == INSUFFICIENT_LABELS

    label(log, ranges[:-1], hour)
    run_once(paths, window_size=window, now=lambda: hour)
    found = performance(paths)
    assert (found.labelled_ranges, found.coverage_status) == (
        MIN_LABELLED_RANGES - 1,
        INSUFFICIENT_LABELS,
    )

    label(log, ranges[-1:], hour)
    run_once(paths, window_size=window, now=lambda: hour)
    found = performance(paths)
    assert (found.labelled_ranges, found.coverage_status, found.coverage) == (
        MIN_LABELLED_RANGES,
        OK,
        1.0,
    )
    # 300 errors of 10 % and 200 of 20 %: the median is 10 %.
    assert found.mdape == pytest.approx(0.1, abs=1e-12)
    exported = series(paths)
    assert exported["recommenditos_performance_mdape"] == pytest.approx(0.1)
    assert exported["recommenditos_performance_coverage"] == 1.0
    assert exported['recommenditos_performance_labels_sufficient{figure="coverage"}'] == 1


def test_the_figures_are_final_seven_days_after_the_last_answer(paths, log):
    seqs = add_answers(log, MIN_LABELLED + 2, endpoint=PREDICT, start=START, error=0.05)
    end = START + (len(seqs) - 1) * STEP
    label(log, seqs[:MIN_LABELLED])

    run_once(paths, window_size=len(seqs), now=lambda: end + LABEL_HORIZON - STEP)
    assert not performance(paths).final

    label(log, seqs[MIN_LABELLED : MIN_LABELLED + 1], end + LABEL_HORIZON - STEP)
    run_once(paths, window_size=len(seqs), now=lambda: end + LABEL_HORIZON)
    final = performance(paths)
    assert (final.final, final.labelled) == (True, MIN_LABELLED + 1)

    # A label later still changes nothing.
    label(log, seqs[-1:], end + LABEL_HORIZON + timedelta(days=1))
    outcome = run_once(paths, window_size=len(seqs), now=lambda: end + timedelta(days=8))
    assert outcome.scored == {}
    assert performance(paths) == final
    assert series(paths)["recommenditos_performance_final"] == 1


def test_a_window_the_log_rotated_away_keeps_its_last_figures(paths, log):
    seqs = add_answers(log, MIN_LABELLED_RANGES, endpoint=RANGE, start=START, error=0.1)
    seqs += add_answers(log, 100, endpoint=RANGE, start=START + 1000 * STEP, inside=False)
    label(log, seqs)
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=1))
    before = performance(paths)
    assert (before.labelled_ranges, before.coverage) == (400, 0.75)

    # The rotation deletes the oldest answers with their feedback.
    log.rotate(seqs[49])
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=2))

    after = performance(paths)
    assert after.final
    assert (after.labelled_ranges, after.coverage, after.mdape) == (400, 0.75, before.mdape)
    assert after.computed_at == before.computed_at


def rotating_during_the_pass(log: PredictionLog, up_to_seq: int, monkeypatch) -> None:
    """The API prunes the log after the pass judged its windows whole, before their labels."""
    original = job.labels

    def rotate_then_read(*args: Any) -> list[Label]:
        log.rotate(up_to_seq)
        return original(*args)

    monkeypatch.setattr(job, "labels", rotate_then_read)


def test_a_rotation_while_the_pass_runs_keeps_the_last_figures(paths, log, monkeypatch):
    seqs = add_answers(log, MIN_LABELLED_RANGES, endpoint=RANGE, start=START, error=0.1)
    seqs += add_answers(log, 100, endpoint=RANGE, start=START + 1000 * STEP, inside=False)
    label(log, seqs)
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=1))
    before = performance(paths)

    rotating_during_the_pass(log, seqs[149], monkeypatch)
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=2))
    monkeypatch.undo()
    during = performance(paths)
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=3))

    # Not recounted on the 250 labels the rotation left.
    assert (before.labelled_ranges, before.coverage) == (400, 0.75)
    assert during == replace(before, final=True)
    assert performance(paths) == during


def test_a_window_first_counted_while_the_log_rotates_is_final(paths, log, monkeypatch):
    seqs = add_answers(log, MIN_LABELLED + 10, endpoint=PREDICT, start=START, error=0.1)
    label(log, seqs)

    rotating_during_the_pass(log, seqs[4], monkeypatch)
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=1))

    found = performance(paths)
    assert (found.labelled, found.final) == (MIN_LABELLED + 5, True)


def test_a_window_of_an_older_monitor_db_is_counted_once_on_what_the_log_kept(paths, log):
    """A window #228's job closed before this table existed, partly rotated away since."""
    seqs = add_answers(log, MIN_LABELLED + 10, endpoint=PREDICT, start=START, error=0.1)
    label(log, seqs)
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=1))
    with sqlite3.connect(paths.db) as connection:
        connection.execute("DROP TABLE window_performance")
    log.rotate(seqs[4])

    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=2))

    found = performance(paths)
    assert (found.labelled, found.final) == (MIN_LABELLED + 5, True)


def test_only_the_windows_requests_count_and_each_once_whichever_worker_wrote_it(paths, log):
    """NFR-03: two workers write the same log; test traffic in between has feedback too."""
    other = sqlite3.connect(paths.log, isolation_level=None)
    other.execute("PRAGMA foreign_keys = ON")
    second = PredictionLog.__new__(PredictionLog)
    second.path, second._connection = paths.log, other  # a second worker
    mine: list[int] = []
    for i in range(MIN_LABELLED):
        writer = log if i % 2 else second
        mine += add_answers(writer, 1, endpoint=PREDICT, start=START + i * STEP, error=0.1)
        # The load test, whose answers can have feedback too, between them.
        loadtest = writer.add(
            body(i), START + i * STEP, client="loadtest", outputs=answer(OBSERVED * 3)
        )
        label(writer, [loadtest])
    for i, seq in enumerate(mine):
        label(log if i % 2 else second, [seq])

    run_once(paths, window_size=len(mine), now=lambda: START + timedelta(hours=1))
    other.close()

    found = performance(paths)
    assert found.labelled == MIN_LABELLED
    assert found.mdape == pytest.approx(0.1, abs=1e-12)


def test_a_partial_window_has_its_error_too(paths, log):
    seqs = add_answers(log, MIN_LABELLED, endpoint=PREDICT, start=START, error=0.1)
    log.add(body(0), START + 1000 * STEP, model_version="v2", outputs=answer(OBSERVED))
    label(log, seqs)

    outcome = run_once(paths, window_size=1000, now=lambda: START + timedelta(hours=1))

    assert [w.status for w in outcome.closed] == ["partial"]
    assert performance(paths).mdape == pytest.approx(0.1, abs=1e-12)


def _cli(paths: Paths, *args: str) -> int:
    return cli.main(["report", "--db", str(paths.db), *args])


def two_windows(paths: Paths, log: PredictionLog, size: int) -> None:
    """A control window the model prices well, then an ES window it prices badly.

    Answered in the month of the samples' reference date, so that the fixture
    listings, which are not shifted as the replay shifts them, keep their age.
    """
    half = size // 2
    control = add_answers(log, half, endpoint=PREDICT, start=SAME_MONTH, error=0.05)
    control += add_answers(log, half, endpoint=RANGE, start=SAME_MONTH + half * STEP, error=0.05)
    later = SAME_MONTH + size * STEP
    es = add_answers(log, half, endpoint=PREDICT, start=later, error=0.25, sample="es")
    # Half of its ranges miss the price.
    for start, inside in ((later + half * STEP, True), (later + 3 * half // 2 * STEP, False)):
        es += add_answers(
            log, half // 2, endpoint=RANGE, start=start, error=0.25, sample="es", inside=inside
        )
    label(log, control + es)
    run_once(paths, window_size=size, now=lambda: SAME_MONTH + timedelta(hours=1))


@pytest.mark.req("FR-14")
def test_a_pass_reports_the_drift_the_error_and_the_coverage_of_each_window(paths, log, capsys):
    two_windows(paths, log, size=2 * MIN_LABELLED_RANGES)
    capsys.readouterr()

    store = Store.open(paths.db)
    control, es = store.windows()
    drifted = {f.feature for f in store.features(es.window_id) if f.drifted}
    store.close()
    # Drift: which properties changed.
    assert (control.flagged, es.flagged) == (False, True)
    assert "country_code" in drifted
    # Error and coverage, against the test values.
    good, bad = performance(paths, control.window_id), performance(paths, es.window_id)
    assert (good.mdape, bad.mdape) == (pytest.approx(0.05), pytest.approx(0.25))
    assert (good.coverage, bad.coverage) == (1.0, 0.5)
    assert (bad.test_mdape, bad.test_coverage) == (TEST_MDAPE, TEST_COVERAGE)
    # All three in Prometheus, for the latest window.
    exported = series(paths)
    assert exported["recommenditos_performance_window_id"] == es.window_id
    assert exported["recommenditos_performance_mdape"] == pytest.approx(0.25)
    assert exported["recommenditos_performance_coverage"] == 0.5
    assert exported["recommenditos_performance_nominal_level"] == 0.9
    assert exported["recommenditos_performance_test_mdape"] == TEST_MDAPE
    assert exported["recommenditos_performance_test_coverage"] == TEST_COVERAGE
    assert exported['recommenditos_performance_min_labels{figure="coverage"}'] == 300
    # And in the report, next to the test values.
    assert _cli(paths) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"Input drift in window {es.window_id} ")
    assert "MdAPE 25.00 % over 600 labelled answers, 3.12 x the test MdAPE of 8.00 %" in text
    assert "coverage 50.00 % of 300 labelled ranges at a nominal 90.00 %" in text
    assert "against 88.00 % on the test set (labels still arriving)." in text


def test_the_report_exports_every_window_and_a_windows_evidence(paths, log, capsys):
    two_windows(paths, log, size=2 * MIN_LABELLED_RANGES)
    capsys.readouterr()

    assert _cli(paths, "--all") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "window",
        "status",
        "seq",
        "es",
        "drift",
        "labelled",
        "mdape",
        "test_mdape",
        "ranges",
        "coverage",
        "nominal",
        "final",
    ]
    assert lines[1].split()[:4] == ["1", "tested", "1-600", "0%"]
    assert lines[2].split()[-6:] == ["0.2500", "0.0800", "300", "0.5000", "0.90", "no"]

    assert _cli(paths, "--all", "--json") == 0
    every = json.loads(capsys.readouterr().out)
    assert [w["performance"]["coverage"] for w in every] == [1.0, 0.5]

    assert _cli(paths, "--window", "2", "--json") == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["performance"]["mdape"] == pytest.approx(0.25)
    assert "coverage 50.00 %" in evidence["finding"]


def test_run_once_prints_each_closed_windows_error_and_coverage(paths, log, capsys):
    seqs = add_answers(log, MIN_LABELLED, endpoint=PREDICT, start=START, error=0.1)
    label(log, seqs)

    code = cli.main(
        [
            "run-once",
            "--log",
            str(paths.log),
            "--db",
            str(paths.db),
            "--data-dir",
            str(paths.data_dir),
            "--metrics",
            str(paths.metrics),
            "--coverage",
            str(paths.coverage),
            "--window-size",
            str(MIN_LABELLED),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "  MdAPE 10.00 % over 200 labelled answers, 1.25 x the test MdAPE of 8.00 %" in out
    assert "error and coverage computed for 1 windows not yet final" in out
