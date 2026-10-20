"""One module per route group, each with a `router`.

The client contract is versioned: `V1_ROUTERS` are mounted under `/v1`, so an
incompatible change can ship as `/v2` beside them. The operational endpoints
(`/`, `/health` and `/metrics`) are not part of that contract and are
mounted at the root, where browsers, monitors and Prometheus look for them:
`OPERATIONAL_ROUTERS`.

A new endpoint is a new module here with its own `router`: one line in the
import and one in the list, each in alphabetical order, so that endpoints added
in parallel do not touch the same line.
"""

from recommenditos.api.routes import (
    health,
    index,
    metrics,
    predict,
)

V1_ROUTERS = [
    predict.router,
]

OPERATIONAL_ROUTERS = [
    health.router,
    index.router,
    metrics.router,
]
