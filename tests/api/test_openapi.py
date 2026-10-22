"""The OpenAPI document `/docs` and `/redoc` render (FR-16).

The checks walk every operation the app has, so an endpoint added later is held
to them without a test of its own: a request example where it takes a body or a
parameter, an example for every response, and every error documented as an RFC
9457 problem.
"""

import copy
import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from tools.openapi_snapshot import SNAPSHOT

from recommenditos.api.app import openapi_contract
from recommenditos.api.errors import PROBLEM_MEDIA_TYPE
from recommenditos.api.schemas.common import ListingFields
from recommenditos.api.serving import ServedModel
from recommenditos.config import PROJ_ROOT

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


@pytest.mark.req("FR-16")
def test_fr16_the_committed_openapi_document_is_the_one_the_app_serves(
    app: FastAPI, served: ServedModel
):
    """docs/docs/openapi.json, which the docs site renders, is the served contract.

    The served document differs from it only by the `enum`s of the checked
    listing fields, which come from the loaded bundle, so that a retrain does
    not change the committed contract.
    """
    committed = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    served_contract = _without_allowed_values(app.openapi(), served)

    assert served_contract == openapi_contract(), (
        "the served document differs from `openapi_contract()` by more than the bundle's "
        "allowed values; finish the document in `recommenditos.api.app._contract`"
    )
    differences = _differences(committed, served_contract)
    snapshot = SNAPSHOT.relative_to(PROJ_ROOT).as_posix()
    assert not differences, (
        f"{snapshot} is not the OpenAPI document the app serves; run `make openapi` and "
        f"commit the file. It differs at the JSON pointers: {', '.join(differences[:10])}"
        + (f" and {len(differences) - 10} more" if len(differences) > 10 else "")
    )


def _without_allowed_values(document: dict[str, Any], served: ServedModel) -> dict[str, Any]:
    """`document` without the `enum`s `create_app` adds: on the string branches of a
    checked field, in the schemas of `ListingFields` and its subclasses."""
    document = copy.deepcopy(document)
    listings = {model.__name__ for model in _subclasses(ListingFields)}
    for name, schema in document["components"]["schemas"].items():
        if name not in listings:
            continue
        for field in served.allowed:
            prop = schema.get("properties", {}).get(field)
            for branch in prop.get("anyOf", [prop]) if prop else []:
                if branch.get("type") == "string":
                    branch.pop("enum", None)
    return document


def _subclasses(cls: type) -> set[type]:
    found: set[type] = set(cls.__subclasses__())
    for sub in list(found):
        found |= _subclasses(sub)
    return found


def _differences(committed: Any, served: Any, pointer: str = "") -> list[str]:
    """The JSON pointers (RFC 6901) at which `committed` and `served` differ."""
    if isinstance(committed, dict) and isinstance(served, dict):
        differences = []
        for key in sorted(committed.keys() | served.keys()):
            child = f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}"
            if key not in served:
                differences.append(f"{child} (only in the file)")
            elif key not in committed:
                differences.append(f"{child} (only in the app)")
            else:
                differences += _differences(committed[key], served[key], child)
        return differences
    return [] if committed == served else [pointer or "/"]
