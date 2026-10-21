"""What a route can ask FastAPI for: the served model, the metrics and the request's ID.

A route declares `served: Served`, `metrics: Metrics` or `request_id: RequestId`
as a parameter and gets the object the app created at startup, so no route reaches into global
state, and a test can build an app around any bundle.
"""

from typing import Annotated

from fastapi import Depends, Request

from recommenditos.api.errors import request_id_of
from recommenditos.api.metrics import ApiMetrics
from recommenditos.api.serving import ServedModel


def served_model(request: Request) -> ServedModel:
    """The model `create_app` loaded, from the app's state."""
    served = request.app.state.served
    assert isinstance(served, ServedModel)
    return served


def api_metrics(request: Request) -> ApiMetrics:
    """The metrics `create_app` registered, from the app's state."""
    metrics = request.app.state.metrics
    assert isinstance(metrics, ApiMetrics)
    return metrics


Served = Annotated[ServedModel, Depends(served_model)]
Metrics = Annotated[ApiMetrics, Depends(api_metrics)]
RequestId = Annotated[str, Depends(request_id_of)]
