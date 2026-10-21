"""The OpenAPI document `/docs` and `/redoc` render (FR-16).

The checks walk every operation the app has, so an endpoint added later is held
to them without a test of its own: a request example where it takes a body or a
parameter, an example for every response, and every error documented as an RFC
9457 problem.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from recommenditos.api.errors import PROBLEM_MEDIA_TYPE
from recommenditos.api.serving import ServedModel

_METHODS = {"get", "put", "post", "delete", "patch"}


def _operations(app: FastAPI) -> list[tuple[str, str, dict]]:
    return [
        (method.upper(), path, operation)
        for path, item in app.openapi()["paths"].items()
        for method, operation in item.items()
        if method in _METHODS
    ]


def _resolve(app: FastAPI, schema: dict) -> dict:
    reference = schema.get("$ref")
    if reference is None:
        return schema
    return app.openapi()["components"]["schemas"][reference.rsplit("/", 1)[1]]


def _has_example(app: FastAPI, media: dict) -> bool:
    return bool(
        media.get("example")
        or media.get("examples")
        or _resolve(app, media.get("schema", {})).get("examples")
    )


@pytest.mark.req("FR-16")
def test_fr16_the_app_has_operations_to_check(app: FastAPI):
    """Guards the tests below against passing on an empty document."""
    assert {(method, path) for method, path, _ in _operations(app)} >= {
        ("POST", "/v1/predict"),
        ("POST", "/v1/price-range"),
        ("GET", "/health"),
        ("GET", "/"),
    }


@pytest.mark.req("FR-16")
def test_fr16_every_operation_has_a_request_example(app: FastAPI):
    missing = []
    for method, path, operation in _operations(app):
        body = operation.get("requestBody", {}).get("content", {})
        missing += [
            f"{method} {path} body ({media_type})"
            for media_type, media in body.items()
            if not _has_example(app, media)
        ]
        missing += [
            f"{method} {path} parameter {parameter['name']}"
            for parameter in operation.get("parameters", [])
            if not (
                parameter.get("example")
                or parameter.get("examples")
                or parameter.get("schema", {}).get("examples")
            )
        ]
    assert not missing, f"no request example for: {', '.join(missing)}"


@pytest.mark.req("FR-16")
def test_fr16_every_response_has_an_example(app: FastAPI):
    missing = [
        f"{method} {path} {status} ({media_type})"
        for method, path, operation in _operations(app)
        for status, response in operation["responses"].items()
        for media_type, media in response.get("content", {}).items()
        if not _has_example(app, media)
    ]
    assert not missing, f"no response example for: {', '.join(missing)}"


@pytest.mark.req("FR-16")
def test_fr16_every_error_response_is_documented_as_a_problem(app: FastAPI):
    wrong = []
    for method, path, operation in _operations(app):
        for status, response in operation["responses"].items():
            if int(status) < 400:
                continue
            content = response.get("content", {})
            schema = content.get(PROBLEM_MEDIA_TYPE, {}).get("schema", {})
            if set(content) != {PROBLEM_MEDIA_TYPE} or schema.get("$ref", "").split("/")[-1] != (
                "Problem"
            ):
                wrong.append(f"{method} {path} {status}")
    assert not wrong, f"not documented as application/problem+json Problem: {', '.join(wrong)}"


@pytest.mark.req("FR-16")
def test_fr16_an_operation_with_a_body_documents_its_422_and_413(app: FastAPI):
    undocumented = [
        f"{method} {path}"
        for method, path, operation in _operations(app)
        if "requestBody" in operation and not {"413", "422"} <= set(operation["responses"])
    ]
    assert not undocumented


@pytest.mark.req("FR-16")
@pytest.mark.parametrize("page", ["/docs", "/redoc"])
def test_fr16_the_documentation_pages_are_served(page: str, client: TestClient):
    response = client.get(page)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.req("FR-12", "FR-16")
def test_fr12_the_documented_values_are_the_served_models(app: FastAPI, served: ServedModel):
    """`/docs` offers the values the loaded bundle accepts, not values written in the schema."""
    properties = app.openapi()["components"]["schemas"]["ValuationRequest"]["properties"]

    assert properties["make"]["enum"] == list(served.allowed["make"])
    drive_train = next(branch for branch in properties["drive_train"]["anyOf"] if "enum" in branch)
    assert drive_train["enum"] == list(served.model.input_schema.levels["drive_train"])
