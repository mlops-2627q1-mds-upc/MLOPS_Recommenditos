"""One request in nginx's access log and in the API's JSON log, joined by its ID (FR-11).

The API's log is read from its container, so this needs the Docker host.
"""

import json
import re
import uuid

import httpx2
import pytest
from tests.integration.conftest import Container

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]


@pytest.mark.req("FR-11", "NFR-08")
def test_fr11_nginx_and_the_api_log_one_request_under_the_same_id(
    client: httpx2.Client, api_container: Container, proxy_container: Container
):
    agent = f"log-join-{uuid.uuid4()}"
    response = client.get("/health", headers={"User-Agent": agent})
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert response.headers.get_list("X-Request-ID") == [request_id], "nginx added its own"

    proxy = [line for line in proxy_container.log_lines() if agent in line]
    assert len(proxy) == 1, proxy
    assert proxy[0].endswith(f"request_id={request_id}")

    lines = [json.loads(line) for line in api_container.log_lines() if line.strip()]
    (answer,) = [line for line in lines if line.get("request_id") == request_id]
    assert (answer["method"], answer["path"], answer["status"]) == ("GET", "/health", 200)
    assert agent not in json.dumps(answer)


@pytest.mark.req("FR-16")
def test_an_answer_of_nginx_itself_carries_its_own_request_id(
    client: httpx2.Client, proxy_container: Container
):
    """A 403 never reaches the API, so nginx names it, in the header and in its log."""
    agent = f"log-own-{uuid.uuid4()}"
    response = client.get("/metrics", headers={"User-Agent": agent})
    assert response.status_code == 403
    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id), request_id
    assert response.headers["X-Content-Type-Options"] == "nosniff"

    proxy = [line for line in proxy_container.log_lines() if agent in line]
    assert len(proxy) == 1, proxy
    assert proxy[0].endswith(f"request_id={request_id}")
