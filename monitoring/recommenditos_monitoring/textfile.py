"""The job's results as Prometheus metrics, through node-exporter's textfile collector.

The job is no server that Prometheus could scrape, so after each pass it writes
`monitor.prom` into the directory node-exporter's textfile collector reads
(deploy/compose.yaml), and node-exporter serves its lines with its own metrics.
Only the latest tested window is exported, with the open window's progress and
the pass's time: the history is `monitor.db`'s, and Prometheus keeps the
series over time anyway. About 47 series:

    recommenditos_drift_window_flagged                    1 when the window drifted
    recommenditos_drift_window_drifted_features           how many of the 23 drifted
    recommenditos_drift_feature_p_value{feature,test}     each feature's p-value
    recommenditos_drift_threshold                         the p-value below which a feature drifted
    recommenditos_drift_window_id / _last_seq             which window it is
    recommenditos_drift_window_end_timestamp_seconds      its last answer's time
    recommenditos_drift_window_es_share                   its share of ES listings
    recommenditos_drift_open_window_requests              n of the window being filled
    recommenditos_drift_window_size                       of 1,000
    recommenditos_monitor_last_success_timestamp_seconds  the end of the last good pass

and the error and coverage of the same window (#229, `performance.py`):

    recommenditos_performance_window_id                   the window they belong to
    recommenditos_performance_requests                    its requests
    recommenditos_performance_ranges                      its range answers
    recommenditos_performance_labelled_requests           its answers with feedback
    recommenditos_performance_labelled_ranges             its range answers with feedback
    recommenditos_performance_labels_sufficient{figure}   1 when a figure has enough labels
    recommenditos_performance_min_labels{figure}          how many it needs, 200 or 300
    recommenditos_performance_mdape                       the real MdAPE
    recommenditos_performance_coverage                    the share of prices in their range
    recommenditos_performance_nominal_level               the share the ranges promise, 0.9
    recommenditos_performance_test_mdape                  the model's test MdAPE
    recommenditos_performance_test_coverage               the model's test coverage, full inputs
    recommenditos_performance_baseline_mdape              the test MdAPE of the window's countries
    recommenditos_performance_baseline_test_rows          the test rows that baseline rests on
    recommenditos_performance_final                       1 when no label can still arrive

A figure without enough labels is left out, not written as 0 or NaN, so a
panel shows no value and an alert on it cannot fire; `labels_sufficient` says
why. The same holds for a test value the evaluation does not have.

The last one, and node-exporter's own `node_textfile_mtime_seconds`, show a job
that stopped (#230's staleness alert).

The file is written under a temporary name in the same directory and renamed,
which is atomic, so a scrape reads the old file or the new one, never half of
one; it is readable to all, since node-exporter runs as `nobody`.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path

from recommenditos_monitoring.drift import THRESHOLD
from recommenditos_monitoring.performance import MIN_LABELLED, MIN_LABELLED_RANGES
from recommenditos_monitoring.store import Store

FILE_NAME = "monitor.prom"


@dataclass(frozen=True)
class Sample:
    value: float
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Metric:
    name: str
    help: str
    samples: tuple[Sample, ...]


def _seconds(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def drift_metrics(store: Store, now: datetime) -> list[Metric]:
    """The drift job's metrics from what `store` holds after a pass that ended at `now`."""
    metrics = [
        Metric(
            "recommenditos_monitor_last_success_timestamp_seconds",
            "When the last monitoring pass that succeeded ended.",
            (Sample(now.timestamp()),),
        ),
        Metric(
            "recommenditos_drift_threshold",
            "The p-value below which a feature has drifted: 0.005 / 23 (NFR-11, Bonferroni).",
            (Sample(THRESHOLD),),
        ),
    ]
    progress = store.open_window()
    if progress is not None:
        metrics += [
            Metric(
                "recommenditos_drift_open_window_requests",
                "Requests in the window being filled.",
                (Sample(progress.requests),),
            ),
            Metric(
                "recommenditos_drift_window_size",
                "Requests in a tested window.",
                (Sample(progress.window_size),),
            ),
        ]
    window = store.latest_tested()
    if window is None:
        return metrics
    if window.flagged is None or window.drifted_features is None:
        raise ValueError(f"window {window.window_id} is tested and has no verdict")
    features = store.features(window.window_id)
    return metrics + [
        Metric(
            "recommenditos_drift_window_flagged",
            "1 when the latest tested window drifted in any feature.",
            (Sample(int(window.flagged)),),
        ),
        Metric(
            "recommenditos_drift_window_drifted_features",
            "Features of the latest tested window below the threshold.",
            (Sample(window.drifted_features),),
        ),
        Metric(
            "recommenditos_drift_feature_p_value",
            "Each feature's p-value in the latest tested window.",
            tuple(
                Sample(result.p_value, {"feature": result.feature, "test": result.test})
                for result in features
            ),
        ),
        Metric(
            "recommenditos_drift_window_id",
            "The number of the latest tested window in monitor.db.",
            (Sample(window.window_id),),
        ),
        Metric(
            "recommenditos_drift_window_last_seq",
            "The prediction log's seq of the latest tested window's last request.",
            (Sample(window.last_seq),),
        ),
        Metric(
            "recommenditos_drift_window_end_timestamp_seconds",
            "When the latest tested window's last request was answered.",
            (Sample(_seconds(window.last_answered_at)),),
        ),
        Metric(
            "recommenditos_drift_window_es_share",
            "The share of ES listings in the latest tested window.",
            (Sample(window.es_share),),
        ),
    ]


def performance_metrics(store: Store) -> list[Metric]:
    """The error and coverage of the latest tested window, the one `drift_metrics` exports."""
    window = store.latest_tested()
    if window is None:
        return []
    performance = store.performance(window.window_id)
    if performance is None:
        return []

    def gauge(name: str, help_text: str, value: float | None) -> list[Metric]:
        if value is None:
            return []
        return [Metric(f"recommenditos_performance_{name}", help_text, (Sample(value),))]

    return [
        *gauge(
            "window_id",
            "The window of monitor.db the error and coverage are of.",
            window.window_id,
        ),
        *gauge("requests", "Requests in the window.", window.requests),
        *gauge("ranges", "Answers of /v1/price-range in the window.", performance.ranges),
        *gauge(
            "labelled_requests",
            "Answers in the window with an observed price (feedback).",
            performance.labelled,
        ),
        *gauge(
            "labelled_ranges",
            "Answers of /v1/price-range in the window with an observed price.",
            performance.labelled_ranges,
        ),
        Metric(
            "recommenditos_performance_labels_sufficient",
            "1 when the window has enough labels for the figure, 0 when it has too few.",
            (
                Sample(int(performance.mdape is not None), {"figure": "mdape"}),
                Sample(int(performance.coverage is not None), {"figure": "coverage"}),
            ),
        ),
        Metric(
            "recommenditos_performance_min_labels",
            "The labelled answers (mdape) or labelled ranges (coverage) a figure needs.",
            (
                Sample(MIN_LABELLED, {"figure": "mdape"}),
                Sample(MIN_LABELLED_RANGES, {"figure": "coverage"}),
            ),
        ),
        *gauge(
            "mdape",
            "Median absolute percentage error of the window's labelled answers.",
            performance.mdape,
        ),
        *gauge(
            "coverage",
            "Share of the window's labelled ranges that contain the observed price.",
            performance.coverage,
        ),
        *gauge(
            "nominal_level",
            "The share of prices the window's ranges state they contain.",
            performance.nominal_level,
        ),
        *gauge(
            "test_mdape",
            "The model's MdAPE on the test set with every input given (metrics.json).",
            performance.test_mdape,
        ),
        *gauge(
            "test_coverage",
            "The model's range coverage on the test set with every input given, all makes.",
            performance.test_coverage,
        ),
        *gauge(
            "baseline_mdape",
            "The test MdAPE of the window's countries, weighted by its labelled answers (#306).",
            performance.baseline_mdape,
        ),
        *gauge(
            "baseline_test_rows",
            "The test rows the window's baseline MdAPE rests on: 1 / sum of share^2 / rows.",
            performance.baseline_test_rows,
        ),
        *gauge(
            "final",
            "1 when the window's labels are in and its figures no longer change.",
            int(performance.final),
        ),
    ]


def _value(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return repr(float(value))


def _labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    escaped = (
        f'{name}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for name, value in sorted(labels.items())
    )
    return "{" + ",".join(escaped) + "}"


def render(metrics: Iterable[Metric]) -> str:
    """The text exposition format of `metrics`, every one a gauge."""
    lines: list[str] = []
    for metric in metrics:
        lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} gauge")
        lines += [
            f"{metric.name}{_labels(sample.labels)} {_value(sample.value)}"
            for sample in metric.samples
        ]
    return "\n".join(lines) + "\n"


def write_atomically(directory: Path, text: str, name: str = FILE_NAME) -> Path:
    """Write `text` to `directory/name` with a rename, so no reader sees half of it."""
    target = directory / name
    # A dot file without the .prom suffix, which the collector does not read.
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
