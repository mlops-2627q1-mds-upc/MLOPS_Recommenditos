"""`GET /`: what this service is, and where its documentation is."""

from fastapi import APIRouter, Request

from recommenditos.api.dependencies import Served
from recommenditos.api.schemas.index import ServiceIndex

router = APIRouter(tags=["operations"])


@router.get("/", summary="Service name, served model and documentation links")
def index(request: Request, served: Served) -> ServiceIndex:
    """The landing page of the service, for a person who opens its address.

    Links are relative to the service, so they hold behind a reverse proxy.
    """
    app = request.app
    return ServiceIndex(
        service=app.title,
        api_version=app.version,
        model=served.info,
        docs=app.docs_url,
        redoc=app.redoc_url,
        openapi=app.openapi_url,
    )
