"""`GET /`: the web page for a browser, what this service is for every other client.

A browser that opens the address asks for `text/html` and gets the page of
`recommenditos.api.webui` (#269, EDN-96); curl, an HTTP library or a monitor
gets the JSON below, as before. Both answers say `Vary: Accept`, so a cache
keeps them apart. It answers `HEAD` too, for an uptime monitor that probes the
address itself (see `routes/health.py`).
"""

from fastapi import APIRouter, Request, Response

from recommenditos.api.dependencies import Served
from recommenditos.api.schemas.index import ServiceIndex
from recommenditos.api.webui import PAGE_HEADERS, WebPage, wants_html

router = APIRouter(tags=["operations"])


@router.head("/", include_in_schema=False, response_model=ServiceIndex)
@router.get(
    "/",
    response_model=ServiceIndex,
    summary="Service name, served model and documentation links; the web page for a browser",
    responses={
        200: {
            "content": {
                "text/html": {
                    "schema": {"type": "string"},
                    "example": '<!doctype html><html lang="en">...</html>',
                }
            },
            "description": "The service index as JSON, or the web page when the request "
            "asks for `text/html` before `application/json`.",
        }
    },
)
def index(request: Request, response: Response, served: Served) -> ServiceIndex | Response:
    """The landing page of the service, for a person who opens its address.

    Links are relative to the service, so they hold behind a reverse proxy.
    """
    if wants_html(request.headers.get("accept")):
        page = request.app.state.web
        if not isinstance(page, WebPage):
            raise TypeError(f"app.state.web is a {type(page).__name__}, not a WebPage")
        answer = page.page.response(request, PAGE_HEADERS)
        answer.headers["Vary"] = "Accept, Accept-Encoding"
        return answer
    response.headers["Vary"] = "Accept"
    app = request.app
    return ServiceIndex(
        service=app.title,
        api_version=app.version,
        model=served.info,
        docs=app.docs_url,
        redoc=app.redoc_url,
        openapi=app.openapi_url,
    )
