"""The FastAPI application: one served model, versioned routes, one error shape.

`create_app` reads the settings from the environment (`settings.py`), loads the
bundle once, before the first request (FR-12: the bundle ships in the image and
the API never calls MLflow), opens the prediction log, wires the middlewares and
the error handlers, mounts the routers of `recommenditos.api.routes` (the client
contract under `/v1`, the operational endpoints at the root), and finishes the
OpenAPI document that `/docs` and `/redoc` render.

uvicorn calls it as a factory in every worker process, through
`python -m recommenditos.api` (`make api`, the image).
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any, get_type_hints

from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from loguru import logger

from recommenditos.api.errors import (
    PROBLEM_MEDIA_TYPE,
    STORE_UNAVAILABLE_DETAIL,
    ProblemError,
    install_handlers,
    problem_response,
    request_id_of,
)
from recommenditos.api.metrics import ApiMetrics, MetricsMiddleware
from recommenditos.api.middleware import (
    BodySizeLimitMiddleware,
    RequestIdMiddleware,
    UnexpectedErrorMiddleware,
)
from recommenditos.api.prediction_log import SqliteStore, StoreError
from recommenditos.api.routes import OPERATIONAL_ROUTERS, V1_ROUTERS
from recommenditos.api.schemas.common import FieldError, ListingFields
from recommenditos.api.serving import ServedModel
from recommenditos.api.settings import API_KEY_ENV_VAR, Settings

API_VERSION = "1.0.0"

_DESCRIPTION = """
Estimates the asking price of a used car in the eight countries of the
AutoScout24 data, with the model the service was built with.

Every error is an RFC 9457 Problem Details object (`application/problem+json`);
a 422 lists every invalid field in `errors`. See `docs/docs/api.md` in the
repository for how to run the service and how errors are shaped.
"""

_TAGS = [
    {"name": "valuation", "description": "Price estimates for a listing."},
    {
        "name": "feedback",
        "description": "Observed prices for the drift job. Internal: refused from outside the "
        "service's network, and requires the API key (FR-10).",
    },
    {"name": "operations", "description": "Service status, for monitoring."},
]


def create_app(model_dir: Path | None = None, settings: Settings | None = None) -> FastAPI:
    """The API configured by `settings`, or the environment, around the bundle in `model_dir`.

    `model_dir` overrides the settings' `model_dir`. The prediction log's schema
    is created when the file has none, so a log that cannot be opened stops the
    service at startup, like a bundle that cannot be served.
    """
    settings = settings or Settings()
    directory = Path(model_dir or settings.model_dir)
    served = ServedModel.load(directory)
    logger.info(f"serving {served.info.variant} version {served.info.version} from {directory}")
    store = SqliteStore(settings.prediction_log, max_bytes=settings.prediction_log_max_bytes)
    logger.info(f"recording answers in {settings.prediction_log}")
    if settings.api_key is None:
        logger.warning(f"{API_KEY_ENV_VAR} is not set: /v1/feedback refuses every request")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.close()

    app = FastAPI(
        title="Recommenditos used-car price API",
        version=API_VERSION,
        description=_DESCRIPTION,
        openapi_tags=_TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.served = served
    app.state.metrics = ApiMetrics(served.info)
    app.state.settings = settings
    app.state.store = store
    # The last middleware added is the outermost: every response, a refused body
    # and an unexpected failure included, carries the request ID.
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(UnexpectedErrorMiddleware)
    app.add_middleware(RequestIdMiddleware)
    # Outermost, so the metrics count and time the answer the client gets (FR-11).
    app.add_middleware(MetricsMiddleware, metrics=app.state.metrics)
    install_handlers(app, _complete_listing_errors)
    app.add_exception_handler(StoreError, _store_unavailable)
    for router in V1_ROUTERS:
        app.include_router(router, prefix="/v1")
    for router in OPERATIONAL_ROUTERS:
        app.include_router(router)
    app.openapi = _openapi(app, served)  # type: ignore[method-assign]
    return app


async def _store_unavailable(request: Request, error: Exception) -> Response:
    """A 503 instead of an answer the prediction log does not hold (NFR-03)."""
    logger.opt(exception=error).error(f"request {request_id_of(request)}: {error}")
    return problem_response(
        request_id_of(request),
        ProblemError(HTTPStatus.SERVICE_UNAVAILABLE, STORE_UNAVAILABLE_DETAIL),
    )


def _complete_listing_errors(
    request: Request, body: Any, errors: list[FieldError]
) -> list[FieldError]:
    """`errors` plus the served model's faults, if the refused body describes a listing.

    Pydantic stops at the first fault of a field but not at the first field, so
    it has reported every type, range and missing field; what it cannot know is
    whether a well-typed `make` or `fuel_category` is one the model accepts. A
    field pydantic already reported is not checked again. The result follows the
    field order of the request body, so a client reads the errors as it reads
    its own request.
    """
    endpoint = getattr(request.scope.get("route"), "endpoint", None)
    hints = get_type_hints(endpoint) if callable(endpoint) else {}
    listing = next(
        (
            hint
            for hint in hints.values()
            if isinstance(hint, type) and issubclass(hint, ListingFields)
        ),
        None,
    )
    if listing is None:
        return errors
    served = request.app.state.served
    if not isinstance(served, ServedModel):
        raise TypeError(f"app.state.served is a {type(served).__name__}, not a ServedModel")
    reported = {_field_of(error) for error in errors}
    errors = errors + [fault for fault in served.faults(body) if _field_of(fault) not in reported]
    order = {name: position for position, name in enumerate(listing.model_fields)}
    return sorted(errors, key=lambda error: order.get(_field_of(error), len(order)))


def _field_of(error: FieldError) -> str:
    """The top-level field an error points at, or `""` for the body as a whole."""
    return error.pointer.removeprefix("#").lstrip("/").split("/")[0]


def _openapi(app: FastAPI, served: ServedModel) -> Callable[[], dict[str, Any]]:
    """The OpenAPI document, finished with what FastAPI cannot know on its own.

    - An error response is documented as `application/problem+json`, the media
      type the API sends, rather than FastAPI's `application/json`.
    - A checked field of a listing body lists the values the served model
      accepts as an `enum`, so `/docs` offers them. They come from the bundle,
      like the check itself.
    """

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            document = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
                tags=app.openapi_tags,
            )
            _problems_as_problem_json(document)
            _allowed_values_as_enums(document, served)
            app.openapi_schema = document
        return app.openapi_schema

    return openapi


def _problems_as_problem_json(document: dict[str, Any]) -> None:
    for path in document.get("paths", {}).values():
        for operation in path.values():
            for status, response in operation.get("responses", {}).items():
                content = response.get("content", {})
                if int(status) >= HTTPStatus.BAD_REQUEST and "application/json" in content:
                    content[PROBLEM_MEDIA_TYPE] = content.pop("application/json")


def _allowed_values_as_enums(document: dict[str, Any], served: ServedModel) -> None:
    listings = {model.__name__ for model in _subclasses(ListingFields)}
    for name, schema in document.get("components", {}).get("schemas", {}).items():
        if name not in listings:
            continue
        for field, allowed in served.allowed.items():
            prop = schema.get("properties", {}).get(field)
            if prop is None:
                continue
            branches = prop.get("anyOf", [prop])
            for branch in branches:
                if branch.get("type") == "string":
                    branch["enum"] = list(allowed)


def _subclasses(cls: type) -> set[type]:
    found: set[type] = set(cls.__subclasses__())
    for sub in list(found):
        found |= _subclasses(sub)
    return found
