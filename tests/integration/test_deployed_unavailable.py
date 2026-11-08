"""What a client meets at the proxy while the API is down, as in a deploy (#273).

The API container is stopped, so the proxy has nothing to forward to, as in a
restart. A browser, which asks for `text/html`, gets a page that says so and
tries again by itself; every other client keeps the RFC 9457 problem. The API
is started again and waited for, so the tests after this file find it healthy.
"""

import base64
from collections.abc import Iterator
import hashlib
import re
import time

import httpx2
import pytest
from tests.integration.conftest import Container, docker

from recommenditos.api.errors import PROBLEM_MEDIA_TYPE

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: What a browser sends when a person opens the address.
BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

#: deploy/nginx.conf's answer for everything but a browser, byte for byte.
PROBLEM = (
    '{"type":"about:blank","title":"Service Unavailable","status":503,'
    '"detail":"The API is not answering; try again in a few seconds."}\n'
)

#: Long enough for the API to load its model and turn healthy again.
RESTART_S = 60


@pytest.fixture(scope="module")
def api_down(client: httpx2.Client, api_container: Container) -> Iterator[None]:
    """The API container stopped, and started again once the module's tests are done."""
    docker("stop", "--time", "5", api_container.id)
    try:
        yield
    finally:
        docker("start", api_container.id)
        deadline = time.monotonic() + RESTART_S
        while True:
            try:
                if client.get("/health", timeout=2).status_code == 200:
                    break
            except httpx2.TransportError:
                pass
            assert time.monotonic() < deadline, "the API did not come back after the test"
            time.sleep(0.5)


@pytest.mark.usefixtures("api_down")
def test_a_browser_gets_a_page_that_retries_while_the_api_is_down(client):
    response = client.get("/", headers={"Accept": BROWSER_ACCEPT})
    assert response.status_code == 503, response.text
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["retry-after"] == "5"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"]
    # It says what is going on, tries again by itself, and asks for nothing else:
    # the API's fonts and scripts are down with the API.
    page = response.text
    assert "The price service is restarting" in page
    assert '<meta http-equiv="refresh" content="5">' in page
    assert "src=" not in page.replace("<script>", "")
    assert 'rel="stylesheet"' not in page

    # The CSP lets exactly the page's one inline script run, by its hash, and the
    # empty icon load, so that the console stays clean and /favicon.ico is not asked.
    (script,) = re.findall(r"<script>(.*?)</script>", page, flags=re.DOTALL)
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    policy = response.headers["content-security-policy"]
    assert f"script-src 'sha256-{digest}'" in policy, policy
    assert "'unsafe-inline'" not in policy.split("style-src")[0], policy
    assert "img-src data:" in policy, policy
    assert "frame-ancestors 'none'" in policy, policy


@pytest.mark.usefixtures("api_down")
@pytest.mark.parametrize(
    "accept",
    [None, "application/json", "*/*", "text/plain"],
    ids=["no Accept", "json", "any", "plain text"],
)
def test_every_other_client_keeps_the_json_problem_while_the_api_is_down(client, accept):
    headers = {} if accept is None else {"Accept": accept}
    response = client.get("/v1/predict", headers=headers)
    assert response.status_code == 503, response.text
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.text == PROBLEM
    assert "retry-after" not in response.headers
