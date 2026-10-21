"""Turn what a load test run collected into its figures (issue #143).

Pure functions over text the run collected - a `/metrics` scrape, a `docker
stats` line, Locust's statistics CSV, request IDs - so they are tested without a
stack. Standard library only, like `plan.py`.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

#: The latency histogram of `recommenditos/api/metrics.py`.
LATENCY_METRIC = "http_request_duration_seconds"

#: NFR-02's p95 targets in seconds, which are bucket bounds of that histogram.
TARGETS_S = (0.2, 0.3)

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)"
)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_SIZE = re.compile(r"^\s*([0-9.]+)\s*([kKMGT]?i?B)\s*$")
_UNITS = {
    "B": 1,
    "kB": 1e3,
    "KB": 1e3,
    "MB": 1e6,
    "GB": 1e9,
    "TB": 1e12,
    "KiB": 2**10,
    "MiB": 2**20,
    "GiB": 2**30,
    "TiB": 2**40,
}

#: (path, upper bound) -> cumulative count of answers, all methods and statuses.
Buckets = dict[tuple[str, float], float]


def latency_buckets(exposition: str) -> Buckets:
    """The cumulative latency bucket counts per route template of one scrape."""
    buckets: Buckets = {}
    for line in exposition.splitlines():
        match = _SAMPLE.match(line)
        if not match or match["name"] != f"{LATENCY_METRIC}_bucket":
            continue
        labels = dict(_LABEL.findall(match["labels"] or ""))
        key = (labels.get("path", ""), float(labels["le"]))
        buckets[key] = buckets.get(key, 0.0) + float(match["value"])
    return buckets


def latency_shares(before: Buckets, after: Buckets) -> dict[str, dict[str, float]]:
    """Per route: the answers between two scrapes, and the share within each target.

    The difference of two scrapes, so whatever the service answered before the
    run does not count. A route with no answer in between is left out.
    """
    shares: dict[str, dict[str, float]] = {}
    for path in sorted({path for path, _ in after}):
        count = after.get((path, float("inf")), 0.0) - before.get((path, float("inf")), 0.0)
        if count <= 0:
            continue
        figures = {"answers": count}
        for target in TARGETS_S:
            within = after.get((path, target), 0.0) - before.get((path, target), 0.0)
            figures[f"within_{round(target * 1000)}ms"] = within / count
        shares[path] = figures
    return shares


def parse_size(text: str) -> float:
    """`'157.3MiB'` or `'1.2GB'`, as `docker stats` prints sizes, in bytes."""
    match = _SIZE.match(text)
    if not match:
        raise ValueError(f"not a size: {text!r}")
    return float(match[1]) * _UNITS[match[2]]


def parse_docker_stats(line: str) -> tuple[float, float]:
    """`'157.3MiB / 15.4GiB|12.5%'` (format `{{.MemUsage}}|{{.CPUPerc}}`): bytes and CPU %."""
    memory, cpu = line.strip().split("|")
    return parse_size(memory.split("/")[0]), float(cpu.strip().rstrip("%"))


@dataclass(frozen=True)
class EndpointStats:
    requests: int
    failures: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    rate_per_s: float

    @property
    def error_rate(self) -> float:
        return self.failures / self.requests if self.requests else 0.0


def percentile(histogram: Mapping[int, int], share: float) -> float:
    """The smallest response time that `share` of the requests did not exceed.

    `histogram` maps a response time in ms, as Locust rounds it, to its count.
    """
    total = sum(histogram.values())
    if not total:
        return math.nan
    needed = math.ceil(share * total)
    seen = 0
    for milliseconds in sorted(histogram):
        seen += histogram[milliseconds]
        if seen >= needed:
            return float(milliseconds)
    return float(max(histogram))


def _stats(
    histogram: Mapping[int, int], requests: int, failures: int, seconds: float
) -> EndpointStats:
    return EndpointStats(
        requests=requests,
        failures=failures,
        p50_ms=percentile(histogram, 0.50),
        p95_ms=percentile(histogram, 0.95),
        p99_ms=percentile(histogram, 0.99),
        rate_per_s=requests / seconds,
    )


def read_locust_json(path: Path, seconds: float) -> dict[str, EndpointStats]:
    """Locust's `--json-file` output, by request name, with `Aggregated` as the total.

    Locust writes this file once, after the last request finished, unlike its
    CSV statistics, which are a snapshot taken every second. The rate is over
    the run's nominal duration.
    """
    stats = {}
    total: Counter[int] = Counter()
    requests = failures = 0
    for entry in json.loads(path.read_text(encoding="utf-8")):
        histogram = {int(ms): count for ms, count in entry["response_times"].items()}
        stats[entry["name"]] = _stats(
            histogram, entry["num_requests"], entry["num_failures"], seconds
        )
        total.update(histogram)
        requests += entry["num_requests"]
        failures += entry["num_failures"]
    stats["Aggregated"] = _stats(total, requests, failures, seconds)
    return stats


@dataclass(frozen=True)
class LogCheck:
    """The prediction log against the answers the client received (NFR-03)."""

    answered: int
    logged: int
    missing: int
    unexpected: int
    duplicated: int

    @property
    def exactly_once(self) -> bool:
        return self.answered == self.logged and not (
            self.missing or self.unexpected or self.duplicated
        )


def check_log(answered: Iterable[str], logged: Iterable[str]) -> LogCheck:
    """Compare the request IDs the client got answers for with those the log holds.

    `logged` is every record the log gained during the run. Missing: an answer
    with no record. Unexpected: a record whose answer the client did not get as
    a success. Duplicated: a request ID recorded more than once.
    """
    answered_ids = Counter(answered)
    logged_ids = Counter(logged)
    return LogCheck(
        answered=sum(answered_ids.values()),
        logged=sum(logged_ids.values()),
        missing=len(answered_ids.keys() - logged_ids.keys()),
        unexpected=len(logged_ids.keys() - answered_ids.keys()),
        duplicated=sum(1 for count in logged_ids.values() if count > 1),
    )


def markdown_table(header: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    header = list(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def percent(value: float) -> str:
    return f"{value * 100:.2f} %"


def mapping_rows(figures: Mapping[str, Mapping[str, float]]) -> list[list[object]]:
    """Rows of the `/metrics` shares table, one per route."""
    return [
        [path, int(f["answers"]), percent(f["within_200ms"]), percent(f["within_300ms"])]
        for path, f in figures.items()
    ]
