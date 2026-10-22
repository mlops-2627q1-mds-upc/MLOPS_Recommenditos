"""`GET /health`: the status, the loaded model and the comparables index (FR-11).

It answers `HEAD` too, which uptime monitors often probe with; FastAPI adds
`HEAD` to no `GET` route by itself. The `HEAD` route is left out of the OpenAPI
document, which would otherwise list a second operation for the same answer.
"""

from fastapi import APIRouter

from recommenditos.api.dependencies import Index, Served
from recommenditos.api.schemas.comparables import ComparablesIndexInfo
from recommenditos.api.schemas.health import Health

router = APIRouter(tags=["operations"])


@router.head("/health", include_in_schema=False)
@router.get("/health", summary="Service status, the served model and the comparables index")
def health(served: Served, index: Index) -> Health:
    """Answers `ok` whenever the service is up, with the model and the index it serves.

    The model and the comparables index are loaded before the service accepts
    its first request, so a service that answers here can value a listing.
    `price_range` says whether it can also answer `/v1/price-range`, which needs
    interval models the bundle may not carry, and `comparables` names the index
    `/v1/comparables` searches, or is null when the service has none.
    """
    comparables = None if index is None else ComparablesIndexInfo.of(index)
    return Health(
        status="ok",
        model=served.info,
        price_range=served.has_price_range,
        comparables=comparables,
    )
