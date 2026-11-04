"""Prometheus in the stack: what it scrapes, what it keeps, and the figures the
dashboards and alerts of M6 read (docs/docs/monitoring.md#resource-monitoring).

Prometheus publishes no port, so every query goes through its own container,
whose busybox `wget` asks the HTTP API on 127.0.0.1:9090.
"""

from collections.abc import Callable
import json
import time
from typing import Any
from urllib.parse import quote

import pytest
from tests.integration.conftest import COMPOSE_PROJECT, Container

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: The scrape jobs of deploy/prometheus/prometheus.yml.
JOBS = {"prometheus", "api", "cadvisor", "node", "alertmanager", "grafana"}

#: The first scrape of a target comes up to one scrape interval (15 s) after
#: Prometheus starts, and cAdvisor sees a new container at its next housekeeping
#: (30 s): long enough for both, short enough that a target that never comes up
#: fails the run instead of stalling it.
DEADLINE_S = 75
POLL_S = 2

#: The series of one service of this stack, as cAdvisor labels it.
SERVICE = (
    'container_label_com_docker_compose_project="{project}",'
    'container_label_com_docker_compose_service="{service}"'
)


def api_get(prometheus: Container, path: str) -> Any:
    """`GET` a path of Prometheus' HTTP API from inside its container, as JSON."""
    body = json.loads(prometheus.exec("wget", "-q", "-O", "-", f"http://127.0.0.1:9090{path}"))
    assert body["status"] == "success", body
    return body["data"]


def query(prometheus: Container, promql: str) -> list[dict[str, Any]]:
    """The instant vector of `promql`."""
    return api_get(prometheus, f"/api/v1/query?query={quote(promql)}")["result"]


def eventually[T](probe: Callable[[], T | None], what: str) -> T:
    """The first result of `probe` that is not None, within the deadline."""
    deadline = time.monotonic() + DEADLINE_S
    while True:
        result = probe()
        if result is not None:
            return result
        if time.monotonic() > deadline:
            pytest.fail(f"{what} within {DEADLINE_S} s")
        time.sleep(POLL_S)


def service_series(service: str) -> str:
    return SERVICE.format(project=COMPOSE_PROJECT, service=service)


@pytest.mark.req("FR-11", "NFR-04")
def test_every_target_is_up(prometheus_container):
    """The API, both exporters and Prometheus itself are scraped, inside the network."""
    last: dict[str, str] = {}

    def all_up() -> dict[str, str] | None:
        targets = api_get(prometheus_container, "/api/v1/targets?state=active")["activeTargets"]
        last.clear()
        last.update(
            {t["labels"]["job"]: f"{t['health']} {t['lastError']}".strip() for t in targets}
        )
        healthy = set(last) == JOBS and all(state == "up" for state in last.values())
        return dict(last) if healthy else None

    eventually(all_up, f"not every target was up; the last states: {last}")


@pytest.mark.req("NFR-04")
def test_prometheus_keeps_at_most_2_gb_for_60_days_in_a_named_volume(prometheus_container):
    """The retention caps are configured, not just intended (NFR-04's deployment check)."""
    flags = api_get(prometheus_container, "/api/v1/status/flags")
    # Prometheus reads a size's units as powers of two, so compose.yaml's 2GB is
    # 2 GiB, 2.15 GB, and the flag API writes it back so.
    assert flags["storage.tsdb.retention.size"] == "2GiB", flags["storage.tsdb.retention.size"]
    assert flags["storage.tsdb.retention.time"] == "60d", flags["storage.tsdb.retention.time"]

    path = flags["storage.tsdb.path"]
    mounts = {m["Destination"]: m for m in prometheus_container.inspect()["Mounts"]}
    assert path in mounts, f"the TSDB at {path} is in the container's own layer, lost with it"
    assert mounts[path]["Type"] == "volume", f"the TSDB is on a {mounts[path]['Type']} mount"
    assert mounts[path]["Name"].endswith("_prometheus-data"), mounts[path]["Name"]


#: The figures the dashboards and alerts of M6 are built on, each with what it says.
RESOURCE_QUERIES = {
    # The API's memory against its limit, and its start time, whose change is a restart.
    "api memory": f"container_memory_working_set_bytes{{{service_series('api')}}}",
    "api memory limit": f"container_spec_memory_limit_bytes{{{service_series('api')}}}",
    "api start time": f"container_start_time_seconds{{{service_series('api')}}}",
    "proxy memory": f"container_memory_working_set_bytes{{{service_series('proxy')}}}",
    # The host: memory left and the root file system's free bytes (NFR-04's 4 GB floor).
    "host memory available": "node_memory_MemAvailable_bytes",
    "host root free": 'node_filesystem_avail_bytes{mountpoint="/"}',
    "host root size": 'node_filesystem_size_bytes{mountpoint="/"}',
    # The API's own figures, through Prometheus.
    "api requests": "sum(http_requests_total)",
}


@pytest.mark.req("FR-11", "NFR-04")
@pytest.mark.parametrize("name", RESOURCE_QUERIES)
def test_the_resource_figures_reach_prometheus(name, prometheus_container):
    """Each figure has exactly one series with a positive value."""
    promql = RESOURCE_QUERIES[name]

    def one_series() -> list[dict[str, Any]] | None:
        result = query(prometheus_container, promql)
        return result or None

    result = eventually(one_series, f"{promql} returned no series")
    assert len(result) == 1, f"{promql} returned {len(result)} series: {result}"
    assert float(result[0]["value"][1]) > 0, f"{promql} = {result[0]['value'][1]}"


@pytest.mark.req("NFR-04")
def test_the_api_memory_limit_is_the_one_compose_sets(prometheus_container, api_container):
    """cAdvisor reads the limit from the container's cgroup, the figure an alert compares with."""
    limit = api_container.inspect()["HostConfig"]["Memory"]
    promql = RESOURCE_QUERIES["api memory limit"]
    result = eventually(lambda: query(prometheus_container, promql) or None, f"{promql} empty")
    assert float(result[0]["value"][1]) == limit
