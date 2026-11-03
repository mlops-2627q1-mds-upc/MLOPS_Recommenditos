"""`GET /assets/<version>/<file>`: the scripts, styles, fonts and icon of the web page.

Only the current version's files exist; an older version or another path is a
404. Not part of the API's contract, so not in the OpenAPI document.
"""

from http import HTTPStatus

from fastapi import APIRouter, Request, Response

from recommenditos.api.errors import ProblemError
from recommenditos.api.webui import ASSET_CACHE_CONTROL, ASSETS_PREFIX, WebPage

router = APIRouter()


@router.get(ASSETS_PREFIX + "/{version}/{name:path}", include_in_schema=False)
def asset(request: Request, version: str, name: str) -> Response:
    """One file of the web page, cached for a year, since its URL names its content."""
    page = request.app.state.web
    if not isinstance(page, WebPage):
        raise TypeError(f"app.state.web is a {type(page).__name__}, not a WebPage")
    file = page.assets.get(name) if version == page.version else None
    if file is None:
        raise ProblemError(HTTPStatus.NOT_FOUND, "The web page has no such file.")
    return file.response(request, {"Cache-Control": ASSET_CACHE_CONTROL})
