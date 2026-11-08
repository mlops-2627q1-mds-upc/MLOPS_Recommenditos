"""The real error and the coverage of the price ranges per window (FR-14, #229).

A window's labels are the observed prices the log's `feedback` holds for its
requests (`log.labels`), joined through `window_requests.request_id`. From them:

- MdAPE: median(|observed - price_eur| / observed) over the labelled answers of
  both endpoints. For a range request with every field `/v1/predict` needs,
  `price_eur` is the point model's (EDN-85), so both endpoints give the same
  estimate for the same listing.
- Coverage: the share of labelled `/v1/price-range` answers with
  lower_eur <= observed <= upper_eur, next to the nominal level, the mean of
  their `confidence_level` (0.9: one level, unless the configuration changed
  inside a window).

Below `MIN_LABELLED` labelled answers, or `MIN_LABELLED_RANGES` labelled ranges
for the coverage, a figure has the status `insufficient_labels` and no value, so
a quiet window is not read as a good one.

Labels can arrive after a window closed, so a pass recomputes a window's figures
until `LABEL_HORIZON` after its last request was answered, and then marks them
final. Once the log's rotation has deleted any of a window's requests, with
their feedback, the window keeps the figures it last had and is final too:
a recount would rest on fewer labels. Fewer labels than the last figures had
show such a rotation even when it runs during a pass, since labels otherwise
only grow.

The figures are compared with the model's values on the test set with every
input given (`baseline`), since a window holds only requests with all ten
required fields: the test MdAPE in `metrics.json` and the coverage of the
`full` scenario, pooled over makes, in `reports/metrics/interval-coverage-by-make.csv`
(not `sc05_coverage`, the worst scenario's, which drops inputs). Not with the
replay's control listings, which are training listings and so give an
optimistic error and coverage.

They are the values of the model that answered the window's requests
(`judged_against`, #307), not of the model running when a pass scores it. The
image holds the evaluate outputs of its own release only, so every pass records
them in `monitor.db` for each model version its `dvc.lock` names
(`release_test_values`), and a window is judged against the values recorded for
its `model_version`, after a promotion too. A window holds one model's answers:
a new `model_version` closes it (`job.py`).

The MdAPE is judged against the window's market (`market_baseline`, #306): the
test MdAPE of each country in `reports/metrics/segments.csv`, weighted by the
countries of the window's labelled answers, since a market's error differs
from the average on the test set itself (`ES` 1.12 times the overall MdAPE).
A country with fewer than `MIN_BASELINE_ROWS` test rows takes the overall test
MdAPE. The baseline comes with the test rows it rests on, so that the alert's
ceiling allows for the baseline's own sampling error
(docs/docs/monitoring.md#why-these-model-thresholds). The coverage keeps the
nominal level, the promise the ranges make in every market.
"""

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import statistics

import yaml

from recommenditos_monitoring.log import Label
from recommenditos_monitoring.store import (
    INSUFFICIENT_LABELS,
    OK,
    CountryTestValues,
    ModelTestValues,
    Performance,
    Store,
    Window,
    timestamp,
)

#: The fewest labelled answers that give an MdAPE.
MIN_LABELLED = 200
#: The fewest labelled ranges that give a coverage.
MIN_LABELLED_RANGES = 300
#: How long after a window's last answer its labels may still arrive.
LABEL_HORIZON = timedelta(days=7)

RANGE_ENDPOINT = "/v1/price-range"
#: The fewest test rows a country's own test MdAPE needs to be its baseline:
#: SC-04's minimum segment, `evaluate.success_criteria.sc04_min_segment_rows`.
MIN_BASELINE_ROWS = 500
#: The `segment` of segments.csv whose levels are the countries.
COUNTRY_SEGMENT = "country_code"


@dataclass(frozen=True)
class Baseline:
    """A model variant's test MdAPE and coverage with every input given, as evaluated."""

    mdape: float | None
    coverage: float | None


NO_BASELINE = Baseline(None, None)

#: The `scenario` and `make` of the coverage-by-make table's full-input, pooled row.
FULL_INPUTS, POOLED = "full", "(pooled)"


def tested_mdape(metrics: Path | None, variant: str) -> float | None:
    """`variant`'s test MdAPE in `metrics` (the evaluate stage's metrics.json), on full inputs."""
    if metrics is None or not metrics.is_file():
        return None
    entry = json.loads(metrics.read_text(encoding="utf-8")).get("variants", {}).get(variant)
    mdape = entry.get("mdape") if isinstance(entry, dict) else None
    return None if mdape is None else float(mdape)


def tested_coverage(coverage: Path | None, variant: str) -> float | None:
    """`variant`'s full-input coverage on the test set, pooled over makes.

    From the evaluate stage's interval-coverage-by-make.csv, which has rows only
    for the variants with a range model.
    """
    if coverage is None or not coverage.is_file():
        return None
    with coverage.open(newline="", encoding="utf-8") as table:
        for row in csv.DictReader(table):
            if (row["variant"], row["scenario"], row["make"]) == (variant, FULL_INPUTS, POOLED):
                return float(row["coverage"])
    return None


def baseline(metrics: Path | None, coverage: Path | None, variant: str) -> Baseline:
    """`variant`'s test values; a missing file or entry gives none.

    The figures then stand without the comparison.
    """
    return Baseline(
        mdape=tested_mdape(metrics, variant), coverage=tested_coverage(coverage, variant)
    )


def bundle_versions(lock: Path | None) -> dict[str, str]:
    """The model version of each variant whose bundle `lock` (dvc.lock) records.

    The hash of `models/<variant>` without DVC's `.dir`, the version the API
    computes from the bundle's files, reports in /health and logs with every
    answer (FR-12). A missing file gives none.
    """
    if lock is None or not lock.is_file():
        return {}
    stages = yaml.safe_load(lock.read_text(encoding="utf-8")).get("stages", {})
    versions = {}
    for stage in stages.values():
        for out in stage.get("outs", []):
            path = str(out.get("path", ""))
            if path.startswith("models/") and path.count("/") == 1 and out.get("md5"):
                versions[path.removeprefix("models/")] = str(out["md5"]).removesuffix(".dir")
    return versions


def release_test_values(
    metrics: Path | None, coverage: Path | None, lock: Path | None, recorded_at: datetime
) -> list[ModelTestValues]:
    """The test values the image's evaluate outputs give each model version of its dvc.lock.

    The three files are one release's, so each variant's values belong to the
    bundle that release's dvc.lock records. A variant without any value is left
    out, so that a later image with its values can still record them.
    """
    values = []
    for variant, version in sorted(bundle_versions(lock).items()):
        test = baseline(metrics, coverage, variant)
        if test != NO_BASELINE:
            values.append(
                ModelTestValues(
                    variant, version, test.mdape, test.coverage, timestamp(recorded_at)
                )
            )
    return values


def judged_against(store: Store, window: Window, previous: Performance | None) -> Baseline:
    """The test values `window`'s figures are compared with: those of the model that answered it.

    Those recorded for its `model_version`. A window of a model that no pass of
    this code ran with, one answered before #307, keeps the values its last
    scoring stored, or has none.
    """
    recorded = store.test_values(window.model_variant, window.model_version)
    if recorded is not None:
        return Baseline(recorded.test_mdape, recorded.test_coverage)
    if previous is not None:
        return Baseline(previous.test_mdape, previous.test_coverage)
    return NO_BASELINE


def release_country_values(
    segments: Path | None, lock: Path | None, recorded_at: datetime
) -> list[CountryTestValues]:
    """Each model version's test rows and test MdAPE per country, from the release's segments.csv.

    Like `release_test_values`: the evaluate stage's segments.csv and dvc.lock
    are one release's, so each variant's rows belong to the bundle dvc.lock
    records. A missing file gives none.
    """
    versions = bundle_versions(lock)
    if not versions or segments is None or not segments.is_file():
        return []
    with segments.open(newline="", encoding="utf-8") as table:
        return [
            CountryTestValues(
                row["variant"],
                versions[row["variant"]],
                row["level"],
                int(row["n"]),
                float(row["mdape"]),
                timestamp(recorded_at),
            )
            for row in csv.DictReader(table)
            if row["segment"] == COUNTRY_SEGMENT and row["variant"] in versions and row["mdape"]
        ]


@dataclass(frozen=True)
class MarketBaseline:
    """The test MdAPE a window's MdAPE is judged against, and the test rows it rests on."""

    mdape: float
    #: m in 1 / m = sum over the window's countries of w_c^2 / m_c.
    test_rows: float


def market_baseline(
    labels: Sequence[Label], overall: float | None, countries: Sequence[CountryTestValues]
) -> MarketBaseline | None:
    """The test MdAPE of the countries of `labels`, weighted by their share of the labels.

    A country with fewer than `MIN_BASELINE_ROWS` test rows, or none, takes the
    `overall` test MdAPE, which rests on all test rows. None without labels, an
    overall value, or country values: a model whose release did not record
    them, so its figures stand without the comparison.
    """
    if not labels or overall is None or not countries:
        return None
    own = {c.country_code: c for c in countries if c.test_rows >= MIN_BASELINE_ROWS}
    all_rows = sum(c.test_rows for c in countries)
    counts: dict[str | None, int] = {}
    for label in labels:
        counts[label.country_code] = counts.get(label.country_code, 0) + 1
    mdape = inverse_rows = 0.0
    for country, count in counts.items():
        share = count / len(labels)
        values = own.get(country) if country is not None else None
        mdape += share * (values.test_mdape if values else overall)
        inverse_rows += share**2 / (values.test_rows if values else all_rows)
    return MarketBaseline(mdape=mdape, test_rows=1 / inverse_rows)


def absolute_percentage_error(label: Label) -> float:
    return abs(label.observed_price_eur - label.price_eur) / label.observed_price_eur


def score(  # noqa: PLR0913 - every input of the figures is one argument
    window: Window,
    labels: Sequence[Label],
    *,
    ranges: int,
    test: Baseline,
    countries: Sequence[CountryTestValues] = (),
    computed_at: datetime,
    final: bool,
) -> Performance:
    """The figures of `window` from its `labels`; `ranges` is its number of range answers.

    `countries` are the test values per country of the model that answered it.
    """
    ranged = [label for label in labels if label.endpoint == RANGE_ENDPOINT]
    mdape = (
        statistics.median(absolute_percentage_error(label) for label in labels)
        if len(labels) >= MIN_LABELLED
        else None
    )
    coverage = (
        sum(
            label.lower_eur <= label.observed_price_eur <= label.upper_eur
            for label in ranged
            if label.lower_eur is not None and label.upper_eur is not None
        )
        / len(ranged)
        if len(ranged) >= MIN_LABELLED_RANGES
        else None
    )
    levels = [label.confidence_level for label in ranged if label.confidence_level is not None]
    market = market_baseline(labels, test.mdape, countries) if mdape is not None else None
    return Performance(
        window_id=window.window_id,
        ranges=ranges,
        labelled=len(labels),
        labelled_ranges=len(ranged),
        mdape_status=INSUFFICIENT_LABELS if mdape is None else OK,
        mdape=mdape,
        coverage_status=INSUFFICIENT_LABELS if coverage is None else OK,
        coverage=coverage,
        nominal_level=statistics.fmean(levels) if levels else None,
        test_mdape=test.mdape,
        test_coverage=test.coverage,
        final=final,
        computed_at=timestamp(computed_at),
        baseline_mdape=None if market is None else market.mdape,
        baseline_test_rows=None if market is None else market.test_rows,
    )


def labels_are_in(window: Window, now: datetime) -> bool:
    """Whether `LABEL_HORIZON` has passed since the window's last answer."""
    return now >= datetime.fromisoformat(window.last_answered_at) + LABEL_HORIZON


def _percent(value: float) -> str:
    return f"{value * 100:.2f} %"


def describe(performance: Performance | None) -> str:
    """One or two sentences on a window's error and coverage, with the test values."""
    if performance is None:
        return "Its error and coverage are not computed yet."
    state = "final" if performance.final else "labels still arriving"
    if performance.mdape is None:
        error = (
            f"MdAPE: insufficient labels ({performance.labelled} labelled answers, "
            f"fewer than {MIN_LABELLED})"
        )
    else:
        error = f"MdAPE {_percent(performance.mdape)} over {performance.labelled} labelled answers"
        if performance.test_mdape:
            error += (
                f", {performance.mdape / performance.test_mdape:.2f} x the test MdAPE of "
                f"{_percent(performance.test_mdape)}"
            )
        if performance.baseline_mdape:
            error += (
                f" and {performance.mdape / performance.baseline_mdape:.2f} x the "
                f"{_percent(performance.baseline_mdape)} of its countries on the test set"
            )
    if performance.coverage is None:
        coverage = (
            f"coverage: insufficient labels ({performance.labelled_ranges} labelled ranges, "
            f"fewer than {MIN_LABELLED_RANGES})"
        )
    else:
        coverage = (
            f"coverage {_percent(performance.coverage)} of {performance.labelled_ranges} "
            f"labelled ranges"
        )
        if performance.nominal_level is not None:
            coverage += f" at a nominal {_percent(performance.nominal_level)}"
        if performance.test_coverage is not None:
            coverage += f", against {_percent(performance.test_coverage)} on the test set"
    return f"{error}; {coverage} ({state})."
