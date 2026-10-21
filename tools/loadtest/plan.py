"""Which valuation endpoints a load test calls, with which bodies (issue #143).

The mix is read from the target's own OpenAPI document rather than assumed: an
endpoint the target does not serve is left out, and each body is cut down to the
fields the endpoint's request schema defines, because every valuation endpoint
refuses a field it does not define. A body that lacks a field the endpoint
requires is not used for it.

Standard library only, so the Locust file and the tests can import it without
the API's dependencies.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Any

#: The valuation endpoints and their share of the requests. UC1, one seller
#: asking about one listing, is the main use case, so `/predict` gets three
#: requests for every one of the other two.
WEIGHTS: Mapping[str, int] = {
    "/v1/predict": 3,
    "/v1/price-range": 1,
    "/v1/comparables": 1,
}

Body = dict[str, Any]


@dataclass(frozen=True)
class Endpoint:
    path: str
    weight: int
    bodies: tuple[Body, ...]


@dataclass(frozen=True)
class Plan:
    """The endpoints a run calls, and why each other one is left out."""

    endpoints: tuple[Endpoint, ...]
    skipped: Mapping[str, str]

    def pick(self, rng: random.Random) -> tuple[str, Body]:
        """One request: an endpoint drawn by weight, and a body drawn for it."""
        endpoint = rng.choices(self.endpoints, weights=[e.weight for e in self.endpoints])[0]
        return endpoint.path, rng.choice(endpoint.bodies)


def _resolve(schema: Mapping[str, Any], openapi: Mapping[str, Any]) -> Mapping[str, Any]:
    """Follow a local `$ref` such as `#/components/schemas/ValuationRequest`."""
    ref = schema.get("$ref")
    if not ref:
        return schema
    node: Any = openapi
    for part in str(ref).removeprefix("#/").split("/"):
        node = node[part]
    return _resolve(node, openapi)


def request_schema(openapi: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    """The JSON request body schema of `POST path`, or None if the target has none."""
    operation = openapi.get("paths", {}).get(path, {}).get("post")
    if operation is None:
        return None
    content = operation.get("requestBody", {}).get("content", {})
    schema = content.get("application/json", {}).get("schema")
    return None if schema is None else _resolve(schema, openapi)


def project(body: Mapping[str, Any], schema: Mapping[str, Any]) -> Body | None:
    """`body` cut down to the fields `schema` defines, or None if one it requires is missing."""
    properties = schema.get("properties", {})
    projected = {key: value for key, value in body.items() if key in properties}
    if any(field not in projected for field in schema.get("required", [])):
        return None
    return projected


def build_plan(openapi: Mapping[str, Any], bodies: Sequence[Mapping[str, Any]]) -> Plan:
    """The endpoints of `WEIGHTS` the target serves, each with the bodies it accepts."""
    endpoints = []
    skipped = {}
    for path, weight in WEIGHTS.items():
        schema = request_schema(openapi, path)
        if schema is None:
            skipped[path] = "not in the target's OpenAPI document"
            continue
        usable = tuple(p for p in (project(body, schema) for body in bodies) if p is not None)
        if not usable:
            skipped[path] = "no body has every field its request schema requires"
            continue
        endpoints.append(Endpoint(path, weight, usable))
    if not endpoints:
        raise ValueError(f"the target serves none of the valuation endpoints: {skipped}")
    return Plan(tuple(endpoints), skipped)
