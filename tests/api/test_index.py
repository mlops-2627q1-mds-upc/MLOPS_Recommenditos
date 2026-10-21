"""`GET /`: the landing page of the service."""

from fastapi.testclient import TestClient
import pytest

from recommenditos.api.serving import ServedModel


@pytest.mark.req("FR-16")
def test_fr16_the_root_names_the_service_and_links_its_documentation(
    client: TestClient, served: ServedModel
):
    """Someone who opens the service's address finds the model and the docs, not a 404."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == served.info.model_dump()
    for link in ("docs", "redoc", "openapi"):
        assert client.get(body[link]).status_code == 200, link


@pytest.mark.req("FR-16")
def test_the_root_answers_head_for_uptime_monitors(client: TestClient):
    head = client.head("/")

    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-type"] == "application/json"
