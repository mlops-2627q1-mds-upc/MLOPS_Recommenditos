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
of a request body, so a generated valid body fits in it and a 413 for one fails
the test (`test_request_bounds.py` holds the largest valid body under the limit).

Known limitation: the response schemas allow additional properties, as
pydantic writes them, so a field an answer gains without documentation is not
caught; a documented field that changes its type or goes missing is.

Known limitation: a generated `/v1/feedback` request names a random request ID
the prediction log does not hold, so it is answered 404 or 422, and the 201, 200
and 409 of a stored price are tested by hand in `test_feedback.py`.
"""

from datetime import UTC, datetime

from fastapi import FastAPI
import pytest
import schemathesis
from schemathesis import GenerationMode
from tests.api.conftest import API_KEY

#: Random requests per operation in the fuzzing phase, valid and invalid
#: together. With the coverage phase, about 970 requests and 18 s for the six
#: operations on a laptop CPU.
MAX_EXAMPLES = 50


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
    case.call_and_validate()
