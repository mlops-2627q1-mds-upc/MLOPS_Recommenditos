"""The job's results as Prometheus metrics, through node-exporter's textfile collector.

The job is no server that Prometheus could scrape, so after each pass it writes
`monitor.prom` into the directory node-exporter's textfile collector reads
(deploy/compose.yaml), and node-exporter serves its lines with its own metrics.
Only the latest tested window is exported, with the open window's progress and
the pass's time: the history is `monitor.db`'s, and Prometheus keeps the
series over time anyway. About 30 series:

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
