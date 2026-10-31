"""The length bounds of the listing bodies, and the 10 KB they keep a body under (NFR-09).

Every string and equipment list of a listing body has a maximum length
(`recommenditos/api/schemas/common.py`), so a body within every bound fits in
the 10 KB the middleware accepts, and the OpenAPI document states the limit as
bounds a client can check. A longer value is a 422 that names the field (FR-03),
not a 413 for the body as a whole.
"""

import json
from typing import Any

from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest
from tests.api.helpers import assert_problem, pointers

# The app imports every route, so every request body is defined before the search.
import recommenditos.api.app  # noqa: F401
from recommenditos.api.middleware import MAX_BODY_BYTES
from recommenditos.api.schemas.common import (
    MAX_EQUIPMENT_ITEM_LENGTH,
    MAX_EQUIPMENT_ITEMS,
    MAX_MODEL_VERSION_LENGTH,
    MAX_NAME_LENGTH,
    ListingFields,
)

#: The longest JSON a float in [0, 1] takes: 17 significant digits and an exponent.
LONGEST_FLOAT = 1.2345678901234567e-100

#: The free-text fields, which the served model does not check against a set.
FREE_TEXT = ("body_color", "paint_type", "upholstery", "upholstery_color", "envir_standard")


def _listing_bodies() -> list[type[ListingFields]]:
    """Every request body that describes a listing, however deep it derives."""
    found: list[type[ListingFields]] = []
    pending = list(ListingFields.__subclasses__())
    while pending:
        body = pending.pop()
        found.append(body)
        pending.extend(body.__subclasses__())
    return sorted(found, key=lambda body: body.__name__)


def _largest_value(schema: dict[str, Any]) -> Any:
    """The value of a field's JSON Schema whose JSON is longest within its bounds."""
    schema = next((s for s in schema.get("anyOf", [schema]) if s.get("type") != "null"), schema)
    match schema["type"]:
        case "string":
            assert "maxLength" in schema or "pattern" in schema, schema
            # A pattern fixes the length: `YYYY-MM` is the only one so far.
            return "2019-06" if "pattern" in schema else "x" * schema["maxLength"]
        case "array":
            assert "maxItems" in schema, schema
            return [_largest_value(schema["items"])] * schema["maxItems"]
        case "number":
            if schema.get("minimum", 0) <= LONGEST_FLOAT:
                return LONGEST_FLOAT
            return schema["maximum"] - 1 / 3
        case "integer":
            return schema["maximum"]
        case "boolean":
            return False
    raise AssertionError(f"no largest value for {schema}")


@pytest.mark.req("NFR-09")
@pytest.mark.parametrize("body", _listing_bodies(), ids=lambda body: body.__name__)
def test_nfr09_a_listing_body_at_every_bound_fits_in_10_kb(body: type[BaseModel]):
    """Every string and list is bounded, and the largest valid body is under the limit.

    It is serialized as Python's `json.dumps` writes it by default, with a space
    after every separator, as Schemathesis sends it. A body can still exceed the
    limit with padding whitespace or escaped non-ASCII characters; that is a 413.
    """
    properties = body.model_json_schema()["properties"]
    largest = {name: _largest_value(schema) for name, schema in properties.items()}
    encoded = json.dumps(largest).encode()

    body.model_validate_json(encoded)
    assert len(encoded) < MAX_BODY_BYTES


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
