"""One module per route group, each with a `router`.

`V1_ROUTERS` are mounted under `/v1`. A new endpoint is a new module here with
its own `router`, added to the tuple in alphabetical order, one per line, so
that endpoints added in parallel do not touch the same line.
"""

from recommenditos.api.routes import health, predict

V1_ROUTERS = (
    health.router,
    predict.router,
)
