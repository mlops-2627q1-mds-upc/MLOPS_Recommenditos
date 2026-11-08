"""A window's MdAPE is judged against the test MdAPE of its countries (#306).

The baseline is the per-country test MdAPE of segments.csv, weighted by the
countries of the window's labelled answers, with the test rows it rests on,
which ModelErrorHigh's ceiling needs (deploy/prometheus/rules/model-performance.yml).
"""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import sqlite3

import pytest
from samples import PredictionLog, write_fixture_samples
from test_performance import (
    COVERAGE,
    LOCK,
    METRICS,
    OBSERVED,
    PREDICT,
    START,
    STEP,
    TEST_MDAPE,
    body,
    label,
    performance,
    series,
    write_test_values,
)

from recommenditos_monitoring.job import Paths, run_once
from recommenditos_monitoring.log import Label
from recommenditos_monitoring.performance import (
    MIN_BASELINE_ROWS,
    MIN_LABELLED,
    bundle_versions,
    describe,
    market_baseline,
    release_country_values,
)
from recommenditos_monitoring.store import CountryTestValues, Store


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    """test_performance's release: a metrics.json, a coverage table and a dvc.lock, no segments."""
    textfile = tmp_path / "textfile"
    textfile.mkdir()
    metrics, coverage, lock = write_test_values(tmp_path / "evaluate")
    return Paths(
        log=tmp_path / "prediction-log.sqlite3",
        db=tmp_path / "results" / "monitor.db",
        data_dir=write_fixture_samples(tmp_path / "samples"),
        textfile_dir=textfile,
        metrics=metrics,
        coverage=coverage,
        lock=lock,
    )


@pytest.fixture
def log(paths: Paths):
    log = PredictionLog(paths.log)
    yield log
    log.close()


SEGMENTS = METRICS.parent / "reports" / "metrics" / "segments.csv"
RECORDED = "2026-11-09T12:00:00.000+00:00"
#: The fixture's countries: DE and ES with their own baseline, LU below the minimum.
COUNTRIES = {"DE": (1000, 0.06), "ES": (600, 0.09), "LU": (100, 0.05)}


def write_segments(directory: Path, version_variant: str = "lgbm-basic") -> Path:
    """A segments.csv in the evaluate stage's format, with rows a wrong lookup would take."""
    segments = directory / "segments.csv"
    header = (
        "variant,segment,level,n,mdape,within_10pct,within_20pct,mae_eur,mape,"
        "counts_toward_sc04,excluded_because\n"
    )
    rows = [
        f"{version_variant},country_code,{country},{n},{mdape},0.5,0.8,1000,0.1,True,\n"
        for country, (n, mdape) in COUNTRIES.items()
    ]
    rows += [
        f"{version_variant},make,BMW,700,0.5,0.5,0.8,1000,0.1,True,\n",
        "b1,country_code,DE,1000,0.7,0.5,0.8,1000,0.1,True,\n",
    ]
    segments.write_text(header + "".join(rows), encoding="utf-8")
    return segments


def values(version: str = "v1") -> list[CountryTestValues]:
    return [
        CountryTestValues("lgbm-basic", version, country, n, mdape, RECORDED)
        for country, (n, mdape) in sorted(COUNTRIES.items())
    ]


def labelled(country: str | None) -> Label:
    return Label("id", PREDICT, OBSERVED, OBSERVED, None, None, None, country_code=country)


def test_the_segments_give_each_bundles_test_mdape_per_country(paths):
    segments = write_segments(paths.db.parent.parent)

    found = release_country_values(segments, paths.lock, START)

    assert found == values()
    assert release_country_values(None, paths.lock, START) == []
    assert release_country_values(segments, None, START) == []


def test_the_baseline_weights_each_countrys_test_mdape_by_its_share_of_the_labels():
    """Three in four answers DE, one in four ES: 0.75 * 6 % + 0.25 * 9 %."""
    labels = [labelled("DE")] * 3 + [labelled("ES")]

    found = market_baseline(labels, TEST_MDAPE, values())

    assert found is not None
    assert found.mdape == pytest.approx(0.75 * 0.06 + 0.25 * 0.09)
    assert found.test_rows == pytest.approx(1 / (0.75**2 / 1000 + 0.25**2 / 600))


def test_a_country_with_too_few_test_rows_or_none_takes_the_overall_values():
    """LU's 100 rows, a country the test set lacks and a missing one: all test rows."""
    all_rows = sum(n for n, _ in COUNTRIES.values())
    assert COUNTRIES["LU"][0] < MIN_BASELINE_ROWS

    for country in ("LU", "PT", None):
        found = market_baseline([labelled(country)] * 2, TEST_MDAPE, values())
        assert found is not None, country
        assert (found.mdape, found.test_rows) == (TEST_MDAPE, pytest.approx(all_rows)), country


def test_without_country_values_or_an_overall_value_a_window_has_no_baseline():
    labels = [labelled("DE")]

    assert market_baseline(labels, TEST_MDAPE, []) is None
    assert market_baseline(labels, None, values()) is None
    assert market_baseline([], TEST_MDAPE, values()) is None


def es_and_de_window(
    paths: Paths, log: PredictionLog, *, es: int, de: int, error: float = 0.1
) -> None:
    """A window of `es` ES and `de` DE answers `error` off, every one labelled."""
    bodies = [{**body(i), "country_code": "ES"} for i in range(es)]
    bodies += [{**body(i), "country_code": "DE"} for i in range(de)]
    outputs = {"price_eur": OBSERVED * (1 + error), "explanation": {}}
    seqs = [log.add(b, START + i * STEP, outputs=outputs) for i, b in enumerate(bodies)]
    label(log, seqs)
    run_once(paths, window_size=len(seqs), now=lambda: START + timedelta(hours=1))


@pytest.mark.req("FR-14")
def test_a_pass_judges_the_windows_mdape_against_its_countries_and_exports_it(paths, log):
    segments = write_segments(paths.db.parent.parent)
    es_and_de_window(replace(paths, segments=segments), log, es=MIN_LABELLED, de=MIN_LABELLED)

    found = performance(paths)
    exported = series(paths)

    assert found.mdape == pytest.approx(0.1)
    assert found.baseline_mdape == pytest.approx(0.5 * 0.09 + 0.5 * 0.06)
    assert found.baseline_test_rows == pytest.approx(1 / (0.25 / 600 + 0.25 / 1000))
    assert exported["recommenditos_performance_baseline_mdape"] == pytest.approx(0.075)
    assert exported["recommenditos_performance_baseline_test_rows"] == pytest.approx(
        found.baseline_test_rows
    )
    # The overall test value stays beside it, for the report and the panels.
    assert exported["recommenditos_performance_test_mdape"] == pytest.approx(TEST_MDAPE)
    assert "1.33 x the 7.50 % of its countries on the test set" in describe(found)


def test_a_window_of_a_model_without_country_values_exports_no_baseline(paths, log):
    """Its release had no segments.csv: ModelErrorHigh, which needs the rows, cannot fire."""
    es_and_de_window(paths, log, es=MIN_LABELLED, de=0)

    found = performance(paths)
    exported = series(paths)

    assert (found.baseline_mdape, found.baseline_test_rows) == (None, None)
    assert found.test_mdape == pytest.approx(TEST_MDAPE)
    assert "recommenditos_performance_baseline_mdape" not in exported
    assert "recommenditos_performance_baseline_test_rows" not in exported


def test_the_first_record_of_a_models_country_values_stays(paths):
    store = Store.open(paths.db)
    try:
        store.record_country_values(values())
        store.record_country_values(
            [replace(value, test_mdape=0.5, recorded_at="later") for value in values()]
        )
        recorded = store.country_values("lgbm-basic", "v1")
        other = store.country_values("lgbm-basic", "v2")
    finally:
        store.close()

    assert recorded == values()
    assert other == []


def test_a_monitor_db_of_the_previous_release_gets_the_baseline_columns(paths, log):
    """Its rows stay, with no baseline, and `user_version` is unchanged."""
    es_and_de_window(paths, log, es=MIN_LABELLED, de=MIN_LABELLED)
    before = performance(paths)
    connection = sqlite3.connect(paths.db)
    for column in ("baseline_test_rows", "baseline_mdape"):
        connection.execute(f"ALTER TABLE window_performance DROP COLUMN {column}")
    (version,) = connection.execute("PRAGMA user_version").fetchone()
    connection.close()

    Store.open(paths.db).close()
    connection = sqlite3.connect(paths.db)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(window_performance)")}
    (reopened,) = connection.execute("PRAGMA user_version").fetchone()
    connection.close()

    assert {"baseline_mdape", "baseline_test_rows"} <= columns
    assert reopened == version
    assert performance(paths) == before
    assert (before.baseline_mdape, before.baseline_test_rows) == (None, None)


def test_the_repositorys_segments_give_the_deployed_models_test_mdape_per_country():
    """The file the image copies, as the evaluate stage's own record has it."""
    variant = json.loads(METRICS.read_text())["deployable_variant"]
    record = json.loads((COVERAGE.parent / f"{variant}.json").read_text())
    expected = {
        row["level"]: (row["n"], pytest.approx(row["mdape"]))
        for row in record["segments"]
        if row["segment"] == "country_code"
    }

    found = release_country_values(SEGMENTS, LOCK, START)
    version = bundle_versions(LOCK)[variant]

    assert {
        value.country_code: (value.test_rows, value.test_mdape)
        for value in found
        if (value.model_variant, value.model_version) == (variant, version)
    } == expected
    # The minimum of a country's own baseline is SC-04's, as params.yaml sets it.
    assert record["sc04_min_segment_rows"] == MIN_BASELINE_ROWS
