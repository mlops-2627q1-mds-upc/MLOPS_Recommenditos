"""Grafana and the alerting in the stack (docs/docs/monitoring.md#grafana, #alerts).

A fresh stack shows the provisioned data source and dashboard without a click,
Grafana answers only on the host's loopback, an anonymous visitor can look but
not change anything, and Prometheus has every alerting rule loaded and hands
what fires to Alertmanager, which is configured with the Discord receiver.

Grafana is asked from inside its own container, with its busybox `wget`, as
Prometheus is in test_deployed_prometheus.py; the admin password is read from
the container's environment, where Compose put the stack's `.env` value.
"""

import base64
import json
import re
from typing import Any

import pytest
from tests.integration.conftest import Container, docker
from tests.integration.test_deployed_prometheus import api_get, eventually, query
from tests.test_monitoring_config import ALERTS, dashboards, panels
import yaml

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

GRAFANA = "http://127.0.0.1:3000"


def wget(container: Container, url: str, *options: str) -> str:
    return container.exec("wget", "-q", "-O", "-", *options, url)


def grafana_get(grafana: Container, path: str, *, admin: bool = False) -> Any:
    options: list[str] = []
    if admin:
        credentials = base64.b64encode(f"admin:{admin_password(grafana)}".encode()).decode()
        options = ["--header", f"Authorization: Basic {credentials}"]
    return json.loads(wget(grafana, f"{GRAFANA}{path}", *options))


def admin_password(grafana: Container) -> str:
    environment = dict(
        variable.split("=", 1)
        for variable in grafana.inspect()["Config"]["Env"]
        if "=" in variable
    )
    password = environment.get("GF_SECURITY_ADMIN_PASSWORD", "")
    assert password, "Grafana runs without an admin password from .env"
    return password


@pytest.fixture(scope="module")
def grafana(stack_container) -> Container:
    return stack_container("grafana")


@pytest.fixture(scope="module")
def alertmanager(stack_container) -> Container:
    return stack_container("alertmanager")


@pytest.mark.req("FR-11")
def test_grafana_provisions_the_prometheus_data_source_and_reaches_it(grafana):
    source = grafana_get(grafana, "/api/datasources/uid/prometheus", admin=True)
    assert (source["type"], source["url"], source["readOnly"]) == (
        "prometheus",
        "http://prometheus:9090",
        True,
    ), source

    health = grafana_get(grafana, "/api/datasources/uid/prometheus/health", admin=True)
    assert health["status"] == "OK", health


@pytest.mark.req("FR-11")
@pytest.mark.parametrize("uid", sorted(dashboards()))
def test_an_anonymous_viewer_sees_the_provisioned_dashboard(uid, grafana):
    """Without logging in, as through the SSH tunnel, and as the file in the repository has it."""
    found = grafana_get(grafana, f"/api/dashboards/uid/{uid}")

    assert found["meta"]["provisioned"] is True
    assert found["meta"]["canSave"] is False and found["meta"]["canEdit"] is False
    served = {panel["id"]: panel["title"] for panel in found["dashboard"]["panels"]}
    assert served == {panel["id"]: panel["title"] for panel in dashboards()[uid]["panels"]}


#: Grafana's own variables, which it replaces before it sends a query.
GRAFANA_VARIABLES = {"$__range": "1h", "$__rate_interval": "1m", "$__interval": "15s"}


def panel_queries() -> list[tuple[str, str, str]]:
    return [
        (uid, panel["title"], target["expr"])
        for uid, dashboard in sorted(dashboards().items())
        for panel in panels(dashboard)
        for target in panel.get("targets", [])
    ]


@pytest.mark.req("FR-11", "FR-14")
@pytest.mark.parametrize(
    ("uid", "title", "expr"), panel_queries(), ids=[f"{u}:{t}" for u, t, _ in panel_queries()]
)
def test_prometheus_accepts_every_panel_query(uid, title, expr, prometheus_container):
    """A query Prometheus cannot parse would leave its panel blank, and no other test sees it."""
    for variable, value in GRAFANA_VARIABLES.items():
        expr = expr.replace(variable, value)
    query(prometheus_container, expr)


@pytest.mark.req("NFR-09")
def test_an_anonymous_visitor_cannot_change_grafana(grafana):
    """Anyone with the tunnel is a Viewer: a write without the admin password is refused."""
    result = docker(
        "exec",
        grafana.id,
        "sh",
        "-c",
        "wget -q -S -O /dev/null --header 'Content-Type: application/json' "
        f'--post-data \'{{"dashboard": {{"title": "x"}}}}\' {GRAFANA}/api/dashboards/db '
        "2>&1; true",
    )
    assert re.search(r"HTTP/1\.1 40[13] ", result), f"an anonymous write was not refused: {result}"


@pytest.mark.req("NFR-09")
def test_grafana_is_published_on_the_hosts_loopback_only(grafana):
    ports = grafana.inspect()["NetworkSettings"]["Ports"]
    bindings = [binding for port in ports.values() if port for binding in port]
    assert bindings, "Grafana publishes no port, so no tunnel reaches it"
    assert {binding["HostIp"] for binding in bindings} == {"127.0.0.1"}, bindings


@pytest.mark.req("NFR-09")
def test_grafana_runs_no_bundled_data_source_but_prometheus(grafana):
    """`.trivyignore` accepts the findings of Grafana's bundled data source plugins because
    `GF_PLUGINS_DISABLE_PLUGINS` keeps every one but Prometheus from loading.

    A bundled plugin is a binary of its own under `plugins-bundled`, which Grafana starts as a
    process; the core data sources (signature `internal`) are part of Grafana's own binary. A
    Grafana release that renames a plugin's ID leaves it out of the list, so it loads again.
    """
    loaded = grafana_get(grafana, "/api/plugins?type=datasource", admin=True)
    bundled = sorted(plugin["id"] for plugin in loaded if plugin["signature"] != "internal")
    assert bundled == ["prometheus"], (
        f"Grafana loads the bundled data sources {bundled}; add every one but prometheus "
        "to GF_PLUGINS_DISABLE_PLUGINS in deploy/compose.yaml, or re-check .trivyignore"
    )

    processes = grafana.exec("ps", "-o", "args").splitlines()
    plugins = sorted(line for line in processes if "/plugins-bundled/" in line)
    assert len(plugins) == 1 and "/gpx_grafana-prometheus-datasource_" in plugins[0], (
        f"Grafana runs these plugin processes: {plugins}"
    )


@pytest.mark.req("FR-11")
def test_prometheus_has_every_alerting_rule_loaded(prometheus_container):
    groups = api_get(prometheus_container, "/api/v1/rules?type=alert")["groups"]
    loaded = {rule["name"]: rule["health"] for group in groups for rule in group["rules"]}
    assert loaded.keys() == ALERTS.keys(), sorted(loaded.keys() ^ ALERTS.keys())
    assert set(loaded.values()) <= {"ok", "unknown"}, loaded


@pytest.mark.req("FR-11")
def test_prometheus_hands_alerts_to_alertmanager(prometheus_container):
    def active() -> list[dict[str, Any]] | None:
        found = api_get(prometheus_container, "/api/v1/alertmanagers")["activeAlertmanagers"]
        return found or None

    found = eventually(active, "Prometheus found no Alertmanager")
    assert [manager["url"] for manager in found] == ["http://alertmanager:9093/api/v2/alerts"]


@pytest.mark.req("FR-11", "NFR-15")
def test_alertmanager_runs_the_discord_receiver_with_the_webhook_from_env(alertmanager):
    status = json.loads(wget(alertmanager, "http://127.0.0.1:9093/api/v2/status"))
    config = yaml.safe_load(status["config"]["original"])
    (discord,) = [receiver for receiver in config["receivers"] if receiver.get("discord_configs")]
    webhook_file = discord["discord_configs"][0]["webhook_url_file"]

    # The URL is in a file only the service user reads, not in the configuration.
    assert "webhook_url:" not in status["config"]["original"]
    mode, owner = alertmanager.exec("stat", "-c", "%a %u", webhook_file).split()
    assert (mode, owner) == ("600", alertmanager.exec("id", "-u")), (mode, owner)
    assert int(alertmanager.exec("stat", "-c", "%s", webhook_file)) > 0, (
        "the webhook file is empty"
    )
    # ... and not in Alertmanager's own environment.
    environment = alertmanager.exec("cat", "/proc/1/environ")
    assert "DISCORD_WEBHOOK_URL" not in environment
