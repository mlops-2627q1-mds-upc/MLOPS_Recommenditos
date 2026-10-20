"""`GET /metrics` inside the Compose network, where Prometheus scrapes it.

From outside the proxy refuses it (`test_deployed_proxy.py`); this is the other
half of that rule: the endpoint exists and answers on `api:8000`.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]


def test_metrics_answer_inside_the_network(proxy_container):
    # From the proxy's container, a sibling of the API like Prometheus will be;
    # busybox `wget` is what the nginx image has.
    exposition = proxy_container.exec("wget", "-q", "-O", "-", "http://api:8000/metrics")
    assert "# TYPE http_requests_total counter" in exposition, exposition[:500]
