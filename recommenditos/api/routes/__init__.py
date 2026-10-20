"""One module per route group, each with a `router`.

The client contract is versioned: `V1_ROUTERS` are mounted under `/v1`, so an
incompatible change can ship as `/v2` beside them. The operational endpoints
(`/`, `/health`, and `/metrics` with #141) are not part of that contract and are
mounted at the root, where browsers, monitors and Prometheus look for them:
`OPERATIONAL_ROUTERS`.

A new endpoint is a new module here with its own `router`, added to one of the
tuples in alphabetical order, one per line, so that endpoints added in parallel
do not touch the same line.
"""

from recommenditos.api.routes import health, index, predict

V1_ROUTERS = (predict.router,)

OPERATIONAL_ROUTERS = (
    health.router,
    index.router,
)
