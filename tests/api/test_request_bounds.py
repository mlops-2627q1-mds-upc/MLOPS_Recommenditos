"""The length bounds of the request bodies, and the 10 KB they keep an ASCII body under (NFR-09).

Every string and list of a request body has a maximum length
(`recommenditos/api/schemas/common.py`), so a body of ASCII text without
escapes within every bound fits in the 10 KB the middleware accepts, and the
OpenAPI document states the limit as bounds a client can check. A longer value
is a 422 that names the field (FR-03), not a 413 for the body as a whole. A
bound counts characters and the limit bytes, so multi-byte or escaped text can
still make a body within every bound larger than 10 KB; that is a 413.

The bodies are the ones the OpenAPI document declares for its operations, so a
request body added outside the listing models is held to the bounds too.
"""

import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from tests.api.helpers import assert_problem, pointers

from recommenditos.api.middleware import MAX_BODY_BYTES
from recommenditos.api.schemas.common import (
    MAX_EQUIPMENT_ITEM_LENGTH,
    MAX_EQUIPMENT_ITEMS,
    MAX_MODEL_VERSION_LENGTH,
    MAX_NAME_LENGTH,
)

#: The longest JSON a float in [0, 1] takes: 17 significant digits and an exponent.
LONGEST_FLOAT = 1.2345678901234567e-100

#: The free-text fields, which the served model does not check against a set.
FREE_TEXT = (
    "body_color",
    "paint_type",
    "upholstery",
    "upholstery_color",
    "envir_standard",
    "original_market",
)

#: A UUID in its canonical form, which is the only length a `uuid` format allows.
UUID = "0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f"


def _request_bodies(openapi: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The schema of every JSON request body of the document, by the schema's name."""
    schemas = openapi["components"]["schemas"]
    bodies: dict[str, dict[str, Any]] = {}
    for item in openapi["paths"].values():
        for operation in item.values():
            content = operation.get("requestBody", {}).get("content", {})
            for media_type in content.values():
                name = media_type["schema"]["$ref"].rsplit("/", 1)[1]
                bodies[name] = schemas[name]
    return bodies


def _members(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The non-null alternatives of a field's schema."""
    return [member for member in schema.get("anyOf", [schema]) if member.get("type") != "null"]


def _unbounded(schema: dict[str, Any], where: str) -> list[str]:
    """The places in `schema` where a string has no length limit or a list no item limit."""
    found: list[str] = []
    for member in _members(schema):
        match member.get("type"):
            case "string":
                if not {"maxLength", "pattern", "format", "enum"} & member.keys():
                    found.append(where)
            case "array":
                if "maxItems" not in member:
                    found.append(where)
                found += _unbounded(member["items"], f"{where}/items")
            case "object":
                for name, field in member.get("properties", {}).items():
                    found += _unbounded(field, f"{where}/{name}")
    return found


def _largest_string(schema: dict[str, Any]) -> str:
    """The longest string a string field's JSON Schema allows."""
    if "enum" in schema:
        return max(schema["enum"], key=len)
    if schema.get("format") == "uuid":
        return UUID
    # A pattern fixes the length: `YYYY-MM` is the only one so far.
    assert "maxLength" in schema or "pattern" in schema, schema
    return "2019-06" if "pattern" in schema else "x" * schema["maxLength"]


def _largest_number(schema: dict[str, Any]) -> float:
    """The float of a number field's JSON Schema whose JSON is longest."""
    if schema.get("minimum", 0) <= LONGEST_FLOAT:
        return LONGEST_FLOAT
    return schema["maximum"] - 1 / 3


def _largest_value(schema: dict[str, Any]) -> Any:
    """The value of a field's JSON Schema whose JSON is longest within its bounds."""
    (schema,) = _members(schema)
    kind = schema["type"]
    if kind == "string":
        return _largest_string(schema)
    if kind == "array":
        assert "maxItems" in schema, schema
        return [_largest_value(schema["items"])] * schema["maxItems"]
    if kind == "number":
        return _largest_number(schema)
    if kind == "integer":
        return schema["maximum"]
    assert kind == "boolean", f"no largest value for {schema}"
    return False


@pytest.mark.req("NFR-09")
def test_nfr09_every_string_and_list_of_a_request_body_is_bounded(app: FastAPI):
    bodies = _request_bodies(app.openapi())

    assert bodies
    unbounded = {name: found for name, body in bodies.items() if (found := _unbounded(body, "#"))}
    assert not unbounded


@pytest.mark.req("NFR-09")
def test_nfr09_a_body_at_every_bound_fits_in_10_kb(app: FastAPI):
    """The largest valid body of every request body is under the limit.

    It is serialized as Python's `json.dumps` writes it by default, with a space
    after every separator, as Schemathesis sends it. Its text is ASCII: a body can
    still exceed the limit with padding whitespace or multi-byte or escaped
    characters; that is a 413.
    """
    for name, body in _request_bodies(app.openapi()).items():
        largest = {field: _largest_value(schema) for field, schema in body["properties"].items()}
        size = len(json.dumps(largest).encode())

        assert size < MAX_BODY_BYTES, f"{name} is {size} bytes at its bounds"


def _at_the_bounds(listing: dict) -> dict:
    """`listing` with every free-text field, the model and the equipment at their bounds."""
    return {
        **listing,
        "model": "m" * MAX_NAME_LENGTH,
        "model_version": "v" * MAX_MODEL_VERSION_LENGTH,
        **dict.fromkeys(FREE_TEXT, "t" * MAX_NAME_LENGTH),
        **{
            name: [f"{i:02d}".ljust(MAX_EQUIPMENT_ITEM_LENGTH, "e") for i in range(items)]
            for name, items in MAX_EQUIPMENT_ITEMS.items()
        },
    }


@pytest.mark.req("NFR-09")
@pytest.mark.parametrize("path", ["/v1/predict", "/v1/price-range", "/v1/comparables"])
def test_nfr09_a_listing_at_every_length_bound_is_answered(
    client: TestClient, listing: dict, path: str
):
    response = client.post(path, json=_at_the_bounds(listing))

    assert response.status_code == 200, response.text


@pytest.mark.req("FR-03")
@pytest.mark.parametrize(
    ("field", "value", "pointer"),
    [
        ("model", "m" * (MAX_NAME_LENGTH + 1), "#/model"),
        ("body_color", "t" * (MAX_NAME_LENGTH + 1), "#/body_color"),
        ("model_version", "v" * (MAX_MODEL_VERSION_LENGTH + 1), "#/model_version"),
        (
            "equipment_entertainment",
            ["Radio"] * (MAX_EQUIPMENT_ITEMS["equipment_entertainment"] + 1),
            "#/equipment_entertainment",
        ),
        ("equipment_safety", ["e" * (MAX_EQUIPMENT_ITEM_LENGTH + 1)], "#/equipment_safety/0"),
    ],
    ids=["model", "free-text", "model-version", "equipment-items", "equipment-item-length"],
)
def test_fr03_a_value_beyond_its_length_bound_is_refused(
    client: TestClient, listing: dict, field: str, value: Any, pointer: str
):
    problem = assert_problem(client.post("/v1/predict", json={**listing, field: value}), 422)

    assert pointers(problem) == [pointer]
