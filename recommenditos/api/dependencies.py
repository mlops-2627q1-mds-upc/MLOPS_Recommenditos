"""What a route can ask FastAPI for: the served model and the request's ID.

A route declares `served: Served` or `request_id: RequestId` as a parameter and
gets the object the app created at startup, so no route reaches into global
state, and a test can build an app around any bundle.
"""

from typing import Annotated

from fastapi import Depends, Request

from recommenditos.api.errors import request_id_of
from recommenditos.api.serving import ServedModel


def served_model(request: Request) -> ServedModel:
    """The model `create_app` loaded, from the app's state."""
    served = request.app.state.served
    assert isinstance(served, ServedModel)
    return served


Served = Annotated[ServedModel, Depends(served_model)]
RequestId = Annotated[str, Depends(request_id_of)]
