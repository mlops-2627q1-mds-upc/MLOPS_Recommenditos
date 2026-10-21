"""`POST /v1/price-range` through the proxy, from the image's own bundle."""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.req("NFR-12", "FR-02", "FR-07")
def test_a_car_named_by_its_make_alone_gets_a_price_range(client, listing, openapi, health):
    response = client.post("/v1/price-range", json={"make": listing["make"]})

    assert response.status_code == 200, response.text
    body = response.json()
    required = openapi["components"]["schemas"]["PriceRange"]["required"]
    assert set(required) <= set(body), f"missing {set(required) - set(body)}"
    assert 0 < body["lower_eur"] <= body["price_eur"] <= body["upper_eur"]
    assert body["model"] == health.json()["model"]
    assert body["request_id"] == response.headers["x-request-id"]
