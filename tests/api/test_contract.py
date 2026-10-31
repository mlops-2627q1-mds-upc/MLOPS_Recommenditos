"""Contract tests generated from the OpenAPI document with Schemathesis (#144).

The hand-written API tests check the cases we thought of; these check that the
service answers as its OpenAPI document says, for requests generated from that
document. Schemathesis reads `/openapi.json` from the app in process (no server,
no network) and, for every operation in it, sends:

- the boundary and invalid values its coverage phase derives from the schema,
  a finite set fixed by the schema (about 370 requests for `/v1/predict`);
- up to `MAX_EXAMPLES` random requests of its fuzzing phase, valid and invalid;
- the documented example.

Every response goes through all of Schemathesis' built-in checks, among them:
no 5xx, the status code is documented, the content type and the body match the
documented response, a schema-valid request is accepted and a schema-invalid
one refused. An endpoint added to the API is covered without a change here,
because the tests come from the document. `/metrics` is included: its response
is documented as Prometheus' text format, which is the content type checked.

Generation is deterministic (Hypothesis' `derandomize`) and keeps no example
database, so every run, in CI and locally, sends the same requests, and a
failure reproduces with the `curl` line Schemathesis prints.

Known limitation: `registration_date` must not lie after the current month
(FR-03), a bound that moves with the calendar and that no static JSON Schema
can state. The schema's pattern states the lower bound, 1900-01; a generated
valid request with a later month than the current one is moved to the current
month, so it is not counted as a valid request the API refused. The upper bound
is tested by hand in `test_predict.py`.

The 10 KB limit of NFR-09 is stated as length bounds on every string and list
of a request body. A bound counts characters and the limit counts bytes, so a
body fits in the limit for ASCII text without escapes only, and a generated valid
body with multi-byte or escaped text can pass it. `accepts_valid_data_within_the_limit`
takes a 413 for valid data only when the body as sent is larger than the limit;
a 413 for a body at or under it fails the test (`test_request_bounds.py` holds the
largest valid ASCII body under the limit).

Known limitation: the response schemas allow additional properties, as
pydantic writes them, so a field an answer gains without documentation is not
caught; a documented field that changes its type or goes missing is.

Known limitation: a generated `/v1/feedback` request names a random request ID
the prediction log does not hold, so it is answered 404 or 422, and the 201, 200
and 409 of a stored price are tested by hand in `test_feedback.py`.
"""

from datetime import UTC, datetime
from http import HTTPStatus

from fastapi import FastAPI
from hypothesis import HealthCheck, given, settings
import pytest
import requests
import schemathesis
from schemathesis import GenerationMode
from schemathesis.checks import positive_data_acceptance
from schemathesis.core.failures import FailureGroup
from tests.api.conftest import API_KEY

from recommenditos.api.middleware import MAX_BODY_BYTES

#: Random requests per operation in the fuzzing phase, valid and invalid
#: together. With the coverage phase, about 970 requests and 18 s for the six
#: operations on a laptop CPU.
MAX_EXAMPLES = 50


def _sent_body_size(response: schemathesis.Response) -> int:
    """The bytes of the request body as it was sent; none is 0."""
    body = response.request.body
    if body is None:
        return 0
    return len(body.encode() if isinstance(body, str) else body)


def accepts_valid_data_within_the_limit(
    ctx: schemathesis.CheckContext, response: schemathesis.Response, case: schemathesis.Case
) -> bool | None:
    """Schemathesis' `positive_data_acceptance`, except a 413 for a body over the limit.

    The length bounds count characters and the 10 KB of NFR-09 counts bytes, so
    the schema cannot promise that every schema-valid body fits: a body of
    multi-byte or escaped text within every bound can be larger than 10 KB, and
    a generator that finds one (Hypothesis' strategies change between versions)
    would turn the test red without a change in this code. A 413 is therefore
    accepted when the body as sent really is larger than `MAX_BODY_BYTES`. A 413
    for a body at or under the limit is a defect and fails like any other
    refusal of valid data.
    """
    if (
        response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        and _sent_body_size(response) > MAX_BODY_BYTES
    ):
        return None
    return positive_data_acceptance(ctx, response, case)


def _this_month() -> str:
    now = datetime.now(UTC)
    return f"{now.year:04d}-{now.month:02d}"


@pytest.fixture(scope="session")
def api_schema(app: FastAPI) -> schemathesis.BaseSchema:
    """The app's OpenAPI document, loaded through the app itself.

    Every API-key scheme the document declares gets the key the test app was
    started with, so an operation behind one, such as `/v1/feedback` (FR-10), is
    tested past its 401; Schemathesis' `ignored_auth` check still sends requests
    without the key and expects them refused.
    """
    schemes = app.openapi().get("components", {}).get("securitySchemes", {})
    keys = {
        name: {"api_key": API_KEY}
        for name, scheme in schemes.items()
        if scheme["type"] == "apiKey"
    }
    config = schemathesis.Config.from_dict(
        {
            **({"auth": {"openapi": keys}} if keys else {}),
            "generation": {
                "mode": "all",
                "max-examples": MAX_EXAMPLES,
                "deterministic": True,
                "database": "none",
            },
        }
    )
    loaded = schemathesis.openapi.from_asgi("/openapi.json", app, config=config)

    @loaded.hook
    def map_case(context: schemathesis.HookContext, case: schemathesis.Case) -> schemathesis.Case:
        """A valid request's registration month no later than this one (see the module)."""
        body = case.body
        if (
            case.meta is not None
            and case.meta.generation.mode is GenerationMode.POSITIVE
            and isinstance(body, dict)
            and isinstance(body.get("registration_date"), str)
            and body["registration_date"] > _this_month()
        ):
            body["registration_date"] = _this_month()
        return case

    return loaded


schema = schemathesis.pytest.from_fixture("api_schema")


@pytest.mark.req("FR-16")
@schema.parametrize()
def test_every_response_matches_the_documented_contract(case: schemathesis.Case) -> None:
    case.call_and_validate(
        excluded_checks=[positive_data_acceptance],
        additional_checks=[accepts_valid_data_within_the_limit],
    )


def _positive_predict_case(api_schema: schemathesis.BaseSchema) -> schemathesis.Case:
    """One generated valid `/v1/predict` request."""
    operation = api_schema["/v1/predict"]["POST"]

    @given(case=operation.as_strategy(generation_mode=GenerationMode.POSITIVE))
    @settings(
        max_examples=1, database=None, derandomize=True, suppress_health_check=list(HealthCheck)
    )
    def draw(case: schemathesis.Case) -> None:
        drawn.append(case)

    drawn: list[schemathesis.Case] = []
    draw()
    return drawn[0]


def _answer_413(case: schemathesis.Case, body_size: int) -> schemathesis.Response:
    """A 413 as if `case` had been sent with a body of `body_size` bytes."""
    request = requests.Request("POST", "http://testserver/v1/predict", data=b" " * body_size)
    return schemathesis.Response(
        status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        headers={"content-type": ["application/problem+json"]},
        content=b"{}",
        request=request.prepare(),
        elapsed=0.0,
        verify=True,
    )


@pytest.mark.req("NFR-09")
def test_nfr09_a_413_is_accepted_for_valid_data_only_over_the_limit(
    api_schema: schemathesis.BaseSchema,
) -> None:
    case = _positive_predict_case(api_schema)
    check = [accepts_valid_data_within_the_limit]

    case.validate_response(_answer_413(case, MAX_BODY_BYTES + 1), checks=check)
    for size in (MAX_BODY_BYTES, 1):
        with pytest.raises(FailureGroup, match="Valid data should have been accepted"):
            case.validate_response(_answer_413(case, size), checks=check)
