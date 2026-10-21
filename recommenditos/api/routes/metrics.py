"""`GET /metrics`: the service's metrics in Prometheus' text format (FR-11)."""

from fastapi import APIRouter, Response

from recommenditos.api.dependencies import Metrics
from recommenditos.api.metrics import CONTENT_TYPE

router = APIRouter(tags=["operations"])

_EXAMPLE = """\
# HELP http_requests_total Requests answered, by method, route template and status code.
# TYPE http_requests_total counter
http_requests_total{method="POST",path="/v1/predict",status="200"} 1250.0
http_requests_total{method="POST",path="/v1/predict",status="422"} 3.0
"""


@router.get(
    "/metrics",
    summary="Request, latency, error and model metrics for Prometheus",
    response_class=Response,
    responses={200: {"content": {CONTENT_TYPE: {"example": _EXAMPLE}}}},
)
def metrics(collected: Metrics) -> Response:
    """Every metric of the service, summed over its worker processes.

    For Prometheus to scrape inside the deployment: the reverse proxy refuses
    this path from outside. `docs/docs/api.md` lists the metrics.
    """
    return Response(collected.exposition(), media_type=CONTENT_TYPE)
