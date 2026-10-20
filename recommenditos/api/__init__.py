"""The HTTP API that serves the price model (M4).

`create_app` builds the FastAPI application around one model bundle. Run it
locally with `make api`; `docs/docs/api.md` explains the endpoints, the error
shape and how to add an endpoint.
"""

from recommenditos.api.app import create_app

__all__ = ["create_app"]
