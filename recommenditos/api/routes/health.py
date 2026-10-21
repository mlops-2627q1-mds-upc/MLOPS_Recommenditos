"""`GET /health`: the status and the loaded model (FR-11)."""

from fastapi import APIRouter

from recommenditos.api.dependencies import Served
from recommenditos.api.schemas.health import Health

router = APIRouter(tags=["operations"])


@router.get("/health", summary="Service status and the served model")
def health(served: Served) -> Health:
    """Answers `ok` whenever the service is up, with the model it serves.

    The model is loaded before the service accepts its first request, so a
    service that answers here can value a listing. `price_range` says whether
    it can also answer `/v1/price-range`, which needs interval models the
    bundle may not carry.
    """
    return Health(status="ok", model=served.info, price_range=served.has_price_range)
