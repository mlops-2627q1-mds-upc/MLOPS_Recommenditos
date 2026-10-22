"""What a route can ask FastAPI for: the model, the metrics, the request ID, the log, the key.

A route declares `served: Served`, `metrics: Metrics`, `request_id: RequestId`,
`log: Log` or `store: Store` as a parameter and gets the object the app created
at startup, so no route reaches into global state, and a test can build an app
around any bundle. `RequireApiKey` in a route's `dependencies` refuses a
request without the API key (FR-10).
"""

from http import HTTPStatus
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from recommenditos.api.errors import request_id_of
from recommenditos.api.metrics import ApiMetrics
from recommenditos.api.prediction_log import PredictionLog, PredictionStore
from recommenditos.api.serving import ServedModel
from recommenditos.api.settings import Settings

API_KEY_HEADER = "X-API-Key"


def served_model(request: Request) -> ServedModel:
    """The model `create_app` loaded, from the app's state."""
    served = request.app.state.served
    if not isinstance(served, ServedModel):
        raise TypeError(f"app.state.served is a {type(served).__name__}, not a ServedModel")
    return served


def api_metrics(request: Request) -> ApiMetrics:
    """The metrics `create_app` registered, from the app's state."""
    metrics = request.app.state.metrics
    if not isinstance(metrics, ApiMetrics):
        raise TypeError(f"app.state.metrics is a {type(metrics).__name__}, not an ApiMetrics")
    return metrics


def prediction_store(request: Request) -> PredictionStore:
    """The store the app's lifespan opened for this run, from the request's state."""
    store: PredictionStore = request.state.store
    return store


def prediction_log(
    request: Request, store: Annotated[PredictionStore, Depends(prediction_store)]
) -> PredictionLog:
    """The prediction log for this request: `log.record(body, answer)` writes its row (FR-13).

    FastAPI creates it once per request, so a route that records twice fails
    instead of writing a second row.
    """
    return PredictionLog(
        store,
        request_id=request_id_of(request),
        # No endpoint takes a path parameter, so the path names the endpoint.
        endpoint=request.url.path,
        started=request.state.request_started,
    )


_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="The key the service was started with (`RECOMMENDITOS_API_KEY`).",
)


def require_api_key(
    request: Request, key: Annotated[str | None, Security(_api_key_header)]
) -> None:
    """Refuse a request whose `X-API-Key` is missing or wrong with a 401 (FR-10).

    The comparison takes the same time however much of the key matches. A
    service started without a key refuses every request.
    """
    settings = request.app.state.settings
    if not isinstance(settings, Settings):
        raise TypeError(f"app.state.settings is a {type(settings).__name__}, not a Settings")
    expected = settings.api_key
    if (
        expected is None
        or key is None
        or not secrets.compare_digest(key.encode(), expected.get_secret_value().encode())
    ):
        raise HTTPException(
            HTTPStatus.UNAUTHORIZED,
            f"This endpoint requires a valid API key in the {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": "APIKey"},
        )


Served = Annotated[ServedModel, Depends(served_model)]
Metrics = Annotated[ApiMetrics, Depends(api_metrics)]
RequestId = Annotated[str, Depends(request_id_of)]
Log = Annotated[PredictionLog, Depends(prediction_log)]
Store = Annotated[PredictionStore, Depends(prediction_store)]
RequireApiKey = Security(require_api_key)
