"""One shape for every error: RFC 9457 Problem Details (FR-16, EDN-84).

Every error response of the API is `application/problem+json` with the members
RFC 9457 defines (`type`, `title`, `status`, `detail`, `instance`) and, for a
request that fails validation, an `errors` list with one entry per invalid field.
That holds for FastAPI's own validation errors, the served model's checks
(FR-03, FR-04), the body-size limit (NFR-09), an unknown route and an
unexpected failure alike, because all of them leave through `problem_response`.

An endpoint added later raises `ProblemError` (or an `HTTPException`) and lists
`problem_responses(...)` in its decorator, so its errors and their documentation
take the same shape without any code of its own.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from recommenditos.api.schemas.common import FieldError, Problem

PROBLEM_MEDIA_TYPE = "application/problem+json"

#: Where the problem types are described for a human, as RFC 9457 asks of `type`.
_TYPES_DOC = (
    "https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/docs/docs/api.md"
)
VALIDATION_ERROR = f"{_TYPES_DOC}#validation-error"
PAYLOAD_TOO_LARGE = f"{_TYPES_DOC}#payload-too-large"
#: RFC 9457's type for a problem that means no more than its HTTP status.
BLANK = "about:blank"


#: Status phrases that RFC 9110 renamed and Python 3.12's `http.HTTPStatus` still
#: spells the old way.
_RFC9110_TITLES: dict[int, str] = {
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "Content Too Large",
    HTTPStatus.UNPROCESSABLE_ENTITY: "Unprocessable Content",
}


class ProblemError(Exception):
    """Raised by a route to answer with a problem; the handler renders it."""

    def __init__(
        self,
        status: int,
        detail: str,
        *,
        type_: str = BLANK,
        title: str | None = None,
        errors: Sequence[FieldError] = (),
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.type = type_
        self.title = title or _RFC9110_TITLES.get(status, HTTPStatus(status).phrase)
        self.errors = list(errors)


class PayloadTooLarge(StarletteHTTPException):
    """The body passed the size limit while it was being read (NFR-09).

    An `HTTPException`, because FastAPI turns any other exception raised while it
    reads a body into a 400 and lets an `HTTPException` through.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, too_large_detail(limit))


def too_large_detail(limit: int) -> str:
    return f"The request body is larger than the limit of {limit} bytes."


def validation_problem(errors: Sequence[FieldError]) -> ProblemError:
    """The 422 of FR-03: every invalid field of the request, in one response."""
    count = len(errors)
    return ProblemError(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        f"The request has {count} invalid field{'s' if count != 1 else ''}.",
        type_=VALIDATION_ERROR,
        title="Invalid request",
        errors=errors,
    )


def problem_response(
    request_id: str, error: ProblemError, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """The one function every error response of the API is built by."""
    problem = Problem(
        type=error.type,
        title=error.title,
        status=error.status,
        detail=error.detail,
        instance=f"urn:uuid:{request_id}",
        errors=error.errors or None,
    )
    return JSONResponse(
        problem.model_dump(exclude_none=True),
        status_code=error.status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=dict(headers or {}),
    )


# --------------------------------------------------------------------------
# Pydantic's errors as field errors
# --------------------------------------------------------------------------


def pointer(location: Iterable[str | int]) -> str:
    """A pydantic `loc` below `body` as a JSON Pointer fragment (RFC 6901)."""
    parts = [str(part).replace("~", "~0").replace("/", "~1") for part in location]
    return "#" + "".join(f"/{part}" for part in parts)


def field_errors(raw: Sequence[Mapping[str, Any]]) -> list[FieldError]:
    """FastAPI's validation errors, one entry per invalid top-level field (FR-03).

    A list field with several bad items is one invalid field, so it gets one
    entry, pointing at the first bad item.
    """
    by_field: dict[str, FieldError] = {}
    for error in raw:
        # Every endpoint takes a JSON body and no query or path parameters, so a
        # location is below `body`; the first endpoint with parameters decides how
        # its errors name them.
        location = list(error.get("loc", ()))
        if location and location[0] == "body":
            location = location[1:]
        if error.get("type") == "json_invalid":
            reason = error.get("ctx", {}).get("error", "")
            by_field.setdefault(
                "#",
                FieldError(
                    pointer="#",
                    detail=f"The body is not valid JSON ({reason}).",
                ),
            )
            continue
        key = pointer(location[:1])
        by_field.setdefault(key, FieldError(pointer=pointer(location), detail=_message(error)))
    return list(by_field.values())


def _message(error: Mapping[str, Any]) -> str:
    if error.get("type") == "missing":
        return "This field is required."
    if error.get("type") == "extra_forbidden":
        return "This field is not part of the request."
    message = str(error.get("msg", "Invalid value."))
    return message.removeprefix("Value error, ")


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def request_id_of(request: Request) -> str:
    return str(request.state.request_id)


async def _problem_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, ProblemError)
    return problem_response(request_id_of(request), error)


async def _http_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, StarletteHTTPException)
    type_ = (
        PAYLOAD_TOO_LARGE if error.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE else BLANK
    )
    detail = (
        error.detail if isinstance(error.detail, str) else HTTPStatus(error.status_code).phrase
    )
    return problem_response(
        request_id_of(request), ProblemError(error.status_code, detail, type_=type_), error.headers
    )


#: Completes the field errors of a body pydantic refused; see `install_handlers`.
ErrorCompleter = Callable[[Request, Any, list[FieldError]], list[FieldError]]


def install_handlers(app: FastAPI, complete: ErrorCompleter) -> None:
    """Route every error of `app` through `problem_response`.

    `complete(request, body, errors)` gets the field errors of a body pydantic
    refused and returns the list the response carries, so that the served
    model's own checks join them and one response lists every invalid field.
    """

    async def _validation_error(request: Request, error: Exception) -> JSONResponse:
        assert isinstance(error, RequestValidationError)
        errors = complete(request, error.body, field_errors(error.errors()))
        return await _problem_error(request, validation_problem(errors))

    app.add_exception_handler(ProblemError, _problem_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)


# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------

_EXAMPLES: dict[int, dict[str, Any]] = {
    HTTPStatus.UNPROCESSABLE_ENTITY: {
        "type": VALIDATION_ERROR,
        "title": "Invalid request",
        "status": 422,
        "detail": "The request has 2 invalid fields.",
        "instance": "urn:uuid:0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f",
        "errors": [
            {
                "pointer": "#/make",
                "detail": "'Trabant' is not a make the served model supports.",
                "allowed": ["Audi", "BMW", "Volkswagen"],
            },
            {
                "pointer": "#/power_kw",
                "detail": "Input should be less than or equal to 1200",
            },
        ],
    },
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: {
        "type": PAYLOAD_TOO_LARGE,
        "title": "Content Too Large",
        "status": 413,
        "detail": too_large_detail(10240),
        "instance": "urn:uuid:0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f",
    },
}

_DESCRIPTIONS: dict[int, str] = {
    HTTPStatus.UNPROCESSABLE_ENTITY: "The request is invalid; `errors` lists every invalid field.",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "The body is larger than 10 KB (NFR-09).",
}


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """The `responses=` of a route that can fail with `statuses`, documented as problems.

    `create_app` moves them from `application/json` to `application/problem+json`
    in the OpenAPI document, so the documentation names the media type the API sends.
    """
    return {
        status: {
            "model": Problem,
            "description": _DESCRIPTIONS.get(status, HTTPStatus(status).phrase),
            "content": {"application/json": {"example": _EXAMPLES[status]}}
            if status in _EXAMPLES
            else {},
        }
        for status in statuses
    }
