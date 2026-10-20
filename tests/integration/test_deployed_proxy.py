"""What every client meets at the proxy: health, the service pages, errors, limits, refusals."""

import httpx2
import pytest
from tests.api.helpers import assert_problem

from recommenditos.api.errors import PROBLEM_MEDIA_TYPE
from recommenditos.api.middleware import MAX_BODY_BYTES

pytestmark = pytest.mark.integration

#: The paths the proxy refuses, with the methods a client might try (FR-10,
#: EDN-13). The API itself would answer 404, 405 or 401, so only the proxy rule
#: passes. `/v1/feedback` because a client endpoint is versioned and the feedback
#: route may sit under `/v1`; `//feedback` because nginx matches the normalised
#: path, and a rule written for the raw one would let it through.
REFUSED = (
    ("GET", "/feedback"),
    ("POST", "/feedback"),
    ("POST", "/v1/feedback"),
    ("POST", "//feedback"),
    ("GET", "/metrics"),
    ("GET", "/v1/metrics"),
)

#: nginx's own limit in deploy/nginx.conf, above the API's 10 KB (NFR-09).
PROXY_MAX_BODY_BYTES = 16 * 1024

JSON = {"content-type": "application/json"}


def assert_proxy_problem(response: httpx2.Response, status: int) -> None:
    """An RFC 9457 problem that nginx answered itself, without the API's request ID."""
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["status"] == status
    assert body["title"] and body["detail"]


@pytest.mark.req("FR-12", "NFR-12")
def test_health_reports_the_bundle_in_the_image(health, expected_version):
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["status"] == "ok"
    assert body["model"]["version"] == expected_version, (
        f"the image serves {body['model']['variant']} version {body['model']['version']}, "
        f"expected {expected_version}"
    )


def test_the_index_and_the_documentation_answer(client, health):
    index = client.get("/")
    assert index.status_code == 200, index.text
    assert index.json()["model"] == health.json()["model"]

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "swagger-ui" in docs.text.lower()
    # Swagger UI loads the document from the same origin, so it has to pass too.
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.req("NFR-09")
def test_the_proxy_sets_security_headers_and_hides_its_version(health):
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert not any(character.isdigit() for character in health.headers.get("server", ""))
    # Plain HTTP, no TLS (EDN-13): HSTS would lock browsers out.
    assert "strict-transport-security" not in health.headers


def test_an_invalid_request_is_an_rfc_9457_problem(client, listing):
    del listing["make"]
    listing["power_kw"] = -1
    problem = assert_problem(client.post("/v1/predict", json=listing), 422)
    assert {error["pointer"] for error in problem["errors"]} >= {"#/make", "#/power_kw"}


def test_an_unknown_route_is_an_rfc_9457_problem(client):
    assert_problem(client.get("/v1/no-such-route"), 404)


@pytest.mark.req("NFR-09")
def test_a_body_over_10_kb_is_refused_by_the_api(client):
    body = b'{"make": "' + b"x" * MAX_BODY_BYTES + b'"}'
    assert len(body) < PROXY_MAX_BODY_BYTES, "this case is about the API's limit, not nginx's"
    response = client.post(
        "/v1/predict", content=body, headers={"content-type": "application/json"}
    )
    assert_problem(response, 413)


@pytest.mark.req("NFR-09")
def test_a_body_over_the_proxy_limit_is_refused_by_nginx(client):
    body = b'{"make": "' + b"x" * PROXY_MAX_BODY_BYTES + b'"}'
    response = client.post(
        "/v1/predict", content=body, headers={"content-type": "application/json"}
    )
    assert_proxy_problem(response, 413)


@pytest.mark.req("FR-10", "NFR-09")
@pytest.mark.parametrize(("method", "path"), REFUSED, ids=[f"{m} {p}" for m, p in REFUSED])
def test_internal_endpoints_are_refused_from_outside(client, base_url, method, path):
    # The absolute URL, because the client would read a path such as `//feedback`
    # as a host name and send the request to `/`.
    response = client.request(method, base_url + path, json={} if method == "POST" else None)
    assert_proxy_problem(response, 403)
