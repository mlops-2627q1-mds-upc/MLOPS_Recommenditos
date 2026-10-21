"""The HTTP API that serves the price model (M4).

`recommenditos.api.app.create_app` builds the FastAPI application around one
model bundle. Run it locally with `make api`; `docs/docs/api.md` explains the
endpoints, the error shape and how to add an endpoint.

The package imports nothing itself, so that `python -m recommenditos.api`, the
process that only supervises the workers, does not load the model's libraries.
"""
