"""`POST /v1/predict` through the proxy, from the image's own bundle.

The example for a new endpoint's case: one valid request through the proxy, and
the documented shape of its answer, read from the served OpenAPI document so the
case follows the contract instead of restating it.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.req("NFR-12")
def test_a_valid_listing_is_valued(client, listing, openapi, health):
    response = client.post("/v1/predict", json=listing)

    assert response.status_code == 200, response.text
    body = response.json()
    required = openapi["components"]["schemas"]["Valuation"]["required"]
    assert set(required) <= set(body), f"missing {set(required) - set(body)}"
    assert body["price_eur"] > 0
    assert body["model"] == health.json()["model"]
    assert body["request_id"] == response.headers["x-request-id"]
