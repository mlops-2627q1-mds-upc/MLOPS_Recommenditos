"""`POST /v1/comparables` through the proxy, from the image's own index."""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.req("FR-12")
def test_health_reports_the_comparables_index_in_the_image(health, expected_comparables_version):
    comparables = health.json()["comparables"]

    assert comparables is not None, "the image carries no comparables index"
    assert comparables["version"] == expected_comparables_version
    assert comparables["listings"] > 0


@pytest.mark.req("NFR-12", "FR-09")
def test_a_car_named_by_its_make_alone_gets_listings_of_that_make(
    client, listing, openapi, health
):
    response = client.post("/v1/comparables", json={"make": listing["make"]})

    assert response.status_code == 200, response.text
    body = response.json()
    required = openapi["components"]["schemas"]["Comparables"]["required"]
    assert set(required) <= set(body), f"missing {set(required) - set(body)}"
    assert 0 < len(body["listings"]) <= 5
    assert {each["make"] for each in body["listings"]} == {listing["make"]}
    assert body["index"] == health.json()["comparables"]
    assert body["request_id"] == response.headers["x-request-id"]
