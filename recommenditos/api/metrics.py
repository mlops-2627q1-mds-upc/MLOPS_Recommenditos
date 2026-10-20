"""The service's Prometheus metrics, served at `/metrics` (FR-11).

Per endpoint, method and status code: the request count, the latency and the
error count. The endpoint is the route template (`/v1/predict`), never the raw
path, so the label set stays bounded whatever a client sends. Beside them, the
signals M6's drift monitoring reads: the distribution of the predicted price,
the warnings by field (FR-05), and the version of the served model.

The figures are correct for any number of uvicorn worker processes when
`PROMETHEUS_MULTIPROC_DIR` names a directory: every worker then writes its
values to its own files there, and a scrape, answered by any one worker, adds
them up (prometheus_client's multiprocess mode). The directory has to be
emptied once before the workers start, never by a worker, or a restarted
worker would wipe the counts of the others: `python -m recommenditos.api`
does it (`prepare_multiprocess_dir`). Without the variable, the metrics are
those of the one process, which is what the tests and `make api` run.

Multiprocess mode supports neither custom collectors nor Info metrics, so the
model version is a gauge labelled with it, always 1, aggregated with `max`
across the workers, which all serve the same bundle.
"""

from http import HTTPStatus
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from fastapi.routing import iter_route_contexts
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    # Only for the annotations: `python -m recommenditos.api` imports this module
    # in the supervising process, which needs neither pydantic nor the schemas.
    from recommenditos.api.schemas.common import FieldWarning, ModelInfo

#: The variable prometheus_client reads, at import, to switch to multiprocess mode.
MULTIPROC_DIR_ENV_VAR = "PROMETHEUS_MULTIPROC_DIR"

#: The media type of the text exposition format.
CONTENT_TYPE = CONTENT_TYPE_LATEST

#: Latency bucket bounds in seconds. 0.2 s and 0.3 s are NFR-02's p95 targets,
#: so the share of answers within each target is read from a bucket directly.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5)

#: Predicted-price bucket bounds in EUR, finer where most used cars are priced.
PRICE_BUCKETS = (
    1_000,
    2_500,
    5_000,
    7_500,
    10_000,
    12_500,
    15_000,
    20_000,
    25_000,
    30_000,
    40_000,
    50_000,
    75_000,
    100_000,
    150_000,
    250_000,
)

#: The `path` label of a request that matches no route, a 404 for any path.
UNMATCHED_PATH = "<unmatched>"

#: The methods kept as their own `method` label; any other is `OTHER`, so a
#: client cannot grow the label set by inventing methods.
METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})

_REQUEST_LABELS = ("method", "path", "status")

#: The lowest status code counted as an error: every 4xx and 5xx answer.
ERROR_STATUS = HTTPStatus.BAD_REQUEST


def multiprocess_dir() -> Path | None:
    """The multiprocess directory, or None when the metrics are per process."""
    directory = os.environ.get(MULTIPROC_DIR_ENV_VAR)
    return Path(directory) if directory else None


def prepare_multiprocess_dir() -> Path | None:
    """Create the multiprocess directory and delete the metric files in it.

    Called once, before the workers start, so a restarted service does not add
    the counts of its previous run to the new ones. Only prometheus_client's
    `*.db` files are deleted: a variable pointed at the wrong directory loses
    nothing else.
    """
    directory = multiprocess_dir()
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.db"):
        path.unlink()
    return directory


class ApiMetrics:
    """The metrics of one app, registered in a registry of their own.

    A registry per app keeps the tests' apps apart. It holds only these
    metrics, not the process and platform collectors of the default registry:
    multiprocess mode cannot serve those, and the container's CPU and memory
    come from cAdvisor or node-exporter in M6.
    """

    def __init__(self, model: "ModelInfo") -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests",
            "Requests answered, by method, route template and status code.",
            _REQUEST_LABELS,
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds",
            "Time from receiving a request to sending the last byte of its answer.",
            _REQUEST_LABELS,
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "http_request_errors",
            "Requests answered with a status code of 400 or above.",
            _REQUEST_LABELS,
            registry=self.registry,
        )
        self.predicted_price = Histogram(
            "recommenditos_predicted_price_eur",
            "Asking prices predicted by /v1/predict, in EUR.",
            buckets=PRICE_BUCKETS,
            registry=self.registry,
        )
        self.warnings = Counter(
            "recommenditos_prediction_warnings",
            "Warnings in /v1/predict answers, by the request field they point at (FR-05).",
            ("field",),
            registry=self.registry,
        )
        self.model = Gauge(
            "recommenditos_model_info",
            "The served model: 1, labelled with its variant and its DVC version (FR-12).",
            ("variant", "version"),
            multiprocess_mode="max",
            registry=self.registry,
        )
        self.model.labels(variant=model.variant, version=model.version).set(1)

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        labels = (method if method in METHODS else "OTHER", path, str(status))
        self.requests.labels(*labels).inc()
        self.latency.labels(*labels).observe(seconds)
        if status >= ERROR_STATUS:
            self.errors.labels(*labels).inc()

    def observe_valuation(self, price_eur: float, warnings: "list[FieldWarning]") -> None:
        """Record one answered valuation: its price and its warnings."""
        self.predicted_price.observe(price_eur)
        for warning in warnings:
            self.warnings.labels(field=warning.pointer.removeprefix("#/").split("/")[0]).inc()

    def exposition(self) -> bytes:
        """The metrics in the text exposition format, summed over the workers if several."""
        directory = multiprocess_dir()
        if directory is None:
            return generate_latest(self.registry)
        registry = CollectorRegistry()
        MultiProcessCollector(registry, path=str(directory))
        return generate_latest(registry)


class MetricsMiddleware:
    """Count and time every HTTP request, whatever answers it.

    Added outermost, so it sees the answer a client gets: a refused body (413)
    and a last-resort 500 included. Its latency runs from the request reaching
    the app to the last byte of the answer.
    """

    def __init__(self, app: ASGIApp, metrics: ApiMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500

        async def watching_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, watching_send)
        finally:
            self.metrics.observe_request(
                scope["method"], route_template(scope), status, time.perf_counter() - started
            )


def route_template(scope: Scope) -> str:
    """The template of the route that answers `scope`, such as `/v1/predict`.

    Matched here rather than read from the route the router records in the
    scope, for two reasons: FastAPI records the route as its router declared it,
    without the `/v1` prefix it was included under, and a request answered
    before routing, such as a body refused by its `Content-Length`, has no route
    recorded at all. A path that matches no route is `UNMATCHED_PATH`; one that
    matches a route but not its method (a 405) is counted under that route.
    """
    partial = None
    for route in iter_route_contexts(getattr(scope.get("app"), "routes", [])):
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return str(route.path)
        if match == Match.PARTIAL and partial is None:
            partial = str(route.path)
    return partial or UNMATCHED_PATH
