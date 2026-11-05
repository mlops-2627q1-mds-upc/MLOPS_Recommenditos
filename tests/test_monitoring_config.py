"""The monitoring configuration in deploy/ hangs together (docs/docs/monitoring.md).

`promtool test rules` checks what each alert does (make prometheus-check);
these tests check what promtool cannot see: that every alert tells a person
what to do and links a runbook section and a dashboard panel that exist, that
every alert has a unit test, that the dashboards draw an alert's threshold
from the rule that records it, and that the dashboard JSON is reviewable, with
stable uids and no ids that Grafana assigns.
"""

from itertools import pairwise
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from recommenditos.config import PROJ_ROOT

DEPLOY = PROJ_ROOT / "deploy"
RULES = sorted((DEPLOY / "prometheus" / "rules").glob("*.yml"))
RULE_TESTS = sorted((DEPLOY / "prometheus" / "tests").glob("*.yml"))
DASHBOARDS = sorted((DEPLOY / "grafana" / "dashboards").glob("*.json"))
DATASOURCES = DEPLOY / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
RUNBOOK = PROJ_ROOT / "docs" / "docs" / "monitoring.md"
COMPOSE = DEPLOY / "compose.yaml"

RUNBOOK_URL = (
    "https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/docs/docs/monitoring.md"
)
#: Grafana as the SSH tunnel of docs/docs/monitoring.md#grafana shows it.
GRAFANA_URL = "http://localhost:3000"

#: Keys that Grafana assigns or bumps when it saves a dashboard: in a
#: provisioned file they would only show as noise in a diff.
VOLATILE_KEYS = {"id", "version", "iteration", "gnetId"}


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def alerting_rules() -> list[dict[str, Any]]:
    return [
        rule
        for path in RULES
        for group in load_yaml(path)["groups"]
        for rule in group["rules"]
        if "alert" in rule
    ]


ALERTS = {rule["alert"]: rule for rule in alerting_rules()}
RECORDED = {
    rule["record"]: rule["expr"]
    for path in RULES
    for group in load_yaml(path)["groups"]
    for rule in group["rules"]
    if "record" in rule
}


def dashboards() -> dict[str, dict[str, Any]]:
    return {
        (dashboard := json.loads(path.read_text(encoding="utf-8")))["uid"]: dashboard
        for path in DASHBOARDS
    }


def slug(heading: str) -> str:
    """The anchor GitHub and MkDocs give a heading."""
    return re.sub(r"[\s]+", "-", re.sub(r"[^\w\s-]", "", heading.strip().lower()))


def runbook_anchors() -> set[str]:
    headings = re.findall(r"^#+\s+(.+)$", RUNBOOK.read_text(encoding="utf-8"), re.MULTILINE)
    return {slug(heading) for heading in headings}


def panels(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    """Every panel, those inside a collapsed row too."""
    found = []
    for panel in dashboard["panels"]:
        found.append(panel)
        found.extend(panel.get("panels", []))
    return found


def test_there_are_alerts_and_a_dashboard():
    assert len(ALERTS) >= 9, sorted(ALERTS)
    assert dashboards(), "no dashboard in deploy/grafana/dashboards/"


@pytest.mark.req("FR-11")
@pytest.mark.parametrize("name", sorted(ALERTS))
def test_each_alert_says_what_to_do_and_links_its_runbook_and_panel(name):
    annotations = ALERTS[name]["annotations"]
    assert {"summary", "description", "runbook_url", "dashboard_url"} <= annotations.keys()
    assert ALERTS[name]["labels"]["severity"] in {"warning", "critical"}

    runbook = urlsplit(annotations["runbook_url"])
    assert annotations["runbook_url"].startswith(RUNBOOK_URL + "#"), annotations["runbook_url"]
    assert runbook.fragment in runbook_anchors(), (
        f"{name} links #{runbook.fragment}, which docs/docs/monitoring.md lacks"
    )

    dashboard_url = urlsplit(annotations["dashboard_url"])
    assert annotations["dashboard_url"].startswith(GRAFANA_URL + "/d/"), annotations[
        "dashboard_url"
    ]
    uid = dashboard_url.path.split("/")[2]
    assert uid in dashboards(), f"{name} links the dashboard {uid}, which is not provisioned"
    (view_panel,) = parse_qs(dashboard_url.query)["viewPanel"]
    ids = {f"panel-{panel['id']}" for panel in panels(dashboards()[uid]) if panel["type"] != "row"}
    assert view_panel in ids, f"{name} links {view_panel}, which the dashboard {uid} lacks"


@pytest.mark.req("FR-11")
def test_each_alert_has_a_unit_test_where_it_fires():
    fired = {
        case["alertname"]
        for path in RULE_TESTS
        for test in load_yaml(path)["tests"]
        for case in test.get("alert_rule_test", [])
        if case.get("exp_alerts")
    }
    assert sorted(ALERTS.keys() - fired) == [], "alerts without a test in deploy/prometheus/tests/"


def test_the_dashboards_draw_each_recorded_threshold_from_its_rule():
    """A threshold recorded for an alert is drawn by name, so it has one source.

    A panel that copied a constant of the rule (a z of 2.576, say) would keep
    drawing the old line after the rule changed.
    """
    queries = " ".join(
        target["expr"]
        for dashboard in dashboards().values()
        for panel in panels(dashboard)
        for target in panel.get("targets", [])
    )
    assert RECORDED, "no recording rules in deploy/prometheus/rules/"
    for name, expr in RECORDED.items():
        assert name in queries, f"no panel draws {name}"
        for constant in re.findall(r"\d+\.\d{2,}", expr):
            assert not re.search(rf"(?<![\w.]){re.escape(constant)}(?![\w])", queries), (
                f"a panel copies {constant} from {name}; draw {name} instead"
            )


def volatile_paths(node: Any, path: str = "") -> list[str]:
    """Where `node` holds a key Grafana assigns, other than a panel's own id.

    An override's matcher and properties, and a transformation, name what they
    match, set or do in an `id` of their own ("byName", "color", "organize"), a
    string, where the ids Grafana assigns are numbers.
    """
    found = []
    if isinstance(node, dict):
        is_panel = "type" in node and "gridPos" in node
        for key, value in node.items():
            schema_id = key == "id" and isinstance(value, str)
            if key in VOLATILE_KEYS and not (is_panel and key == "id") and not schema_id:
                found.append(f"{path}/{key}")
            found.extend(volatile_paths(value, f"{path}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(volatile_paths(value, f"{path}/{index}"))
    return found


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda path: path.name)
def test_the_dashboard_is_reviewable(path):
    """A stable uid, unique panel ids, no ids Grafana assigns, only the provisioned data source."""
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    datasource_uids = {source["uid"] for source in load_yaml(DATASOURCES)["datasources"]}

    assert dashboard["uid"] == path.stem, f"{path.name} has the uid {dashboard['uid']}"
    assert dashboard["editable"] is False
    assert volatile_paths(dashboard) == []
    ids = [panel["id"] for panel in panels(dashboard)]
    assert len(ids) == len(set(ids)), f"panel ids repeat: {sorted(ids)}"
    for panel in panels(dashboard):
        for target in panel.get("targets", []):
            assert target["datasource"]["uid"] in datasource_uids, (panel["title"], target)
            assert target["expr"].strip(), panel["title"]


def lightness(hex_colour: str) -> float:
    """CIE L* of a `#rrggbb` colour: what remains of it on a greyscale print."""
    red, green, blue = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
    return 116 * luminance ** (1 / 3) - 16 if luminance > 0.008856 else 903.3 * luminance


def fixed_colours(panel: dict[str, Any]) -> dict[str, str]:
    """The colour each series of `panel` is given by name, from its overrides."""
    return {
        override["matcher"]["options"]: prop["value"]["fixedColor"]
        for override in panel["fieldConfig"]["overrides"]
        if override["matcher"]["id"] == "byName"
        for prop in override["properties"]
        if prop["id"] == "color" and prop["value"]["mode"] == "fixed"
    }


def container_panels() -> list[dict[str, Any]]:
    """The panels that draw one series per compose service."""
    return [
        panel
        for panel in panels(dashboards()["service-health"])
        if any(
            "compose_service}}" in target.get("legendFormat", "")
            for target in panel.get("targets", [])
        )
    ]


def test_each_service_has_one_fixed_colour_that_survives_greyscale():
    """The default palette gave api and proxy the same yellow; every service now has its own.

    The colours differ in lightness, not only in hue, so that a greyscale
    print or a colour-blind reader still tells the lines apart.
    """
    services = set(load_yaml(COMPOSE)["services"])
    found = container_panels()
    assert len(found) >= 3, "the memory, CPU and restart panels name containers"
    colours = [fixed_colours(panel) for panel in found]
    for panel, by_name in zip(found, colours, strict=True):
        assert by_name.keys() == services, f"{panel['title']} lacks a colour for a service"
        assert by_name == colours[0], f"{panel['title']} colours a service differently"
    levels = sorted(lightness(colour) for colour in colours[0].values())
    gaps = [later - earlier for earlier, later in pairwise(levels)]
    assert min(gaps) >= 6, f"two services are {min(gaps):.1f} L* apart in greyscale"


def test_threshold_lines_are_dashed():
    """A solid threshold line reads like a gridline, so it is dashed, like the memory floors."""
    for dashboard in dashboards().values():
        for panel in panels(dashboard):
            style = panel.get("fieldConfig", {}).get("defaults", {}).get("custom", {})
            mode = style.get("thresholdsStyle", {}).get("mode", "off")
            assert mode in {"off", "dashed", "dashed+area"}, (
                f"{panel['title']} draws its thresholds with {mode!r}"
            )


def test_the_p_value_table_has_room_for_every_feature():
    """The table showed 16 of its 23 features and cut the last row of them.

    Grafana's grid is 30 px per unit and 8 px between panels; a panel spends
    about 40 px on its title, and the table 36 px on its header and on each
    row. The number of features is the one in the title (`0.005 / 23`).
    """
    (table,) = (
        panel
        for panel in panels(dashboards()["model-performance"])
        if panel["type"] == "table" and "p-value of each feature" in panel["title"]
    )
    found = re.search(r"/ (\d+)", table["title"])
    assert found, table["title"]
    features = int(found.group(1))
    units = table["gridPos"]["h"]
    inside = units * 30 + (units - 1) * 8 - 40 - 36
    assert inside >= features * 36, f"{inside} px hold {inside // 36} of {features} rows"


def test_the_predicted_price_percentiles_need_a_minimum_of_requests():
    """After an API restart three requests put all three percentiles at one bucket's edge.

    Each query is drawn only where `/v1/predict` answered at least this many
    requests in the same 30 minutes, and the description says so.
    """
    (panel,) = (
        panel
        for panel in panels(dashboards()["model-performance"])
        if panel["title"] == "Predicted prices of /v1/predict"
    )
    assert len(panel["targets"]) == 3
    for target in panel["targets"]:
        guard = re.search(
            r"and on\(\) \(sum\(increase\(recommenditos_predicted_price_eur_count\[30m\]\)\)"
            r" >= (\d+)\)",
            target["expr"],
        )
        assert guard, f"{target['legendFormat']} has no minimum of requests: {target['expr']}"
        assert int(guard.group(1)) >= 30
    assert "at least 30 requests" in panel["description"]


def test_alertmanager_reads_the_webhook_file_the_entrypoint_writes():
    """The webhook URL goes from `.env` to a file only the alertmanager service writes."""
    config = load_yaml(DEPLOY / "alertmanager" / "alertmanager.yml")
    files = {
        receiver_config["webhook_url_file"]
        for receiver in config["receivers"]
        for receiver_config in receiver.get("discord_configs", [])
    }
    service = load_yaml(DEPLOY / "compose.yaml")["services"]["alertmanager"]
    script = service["entrypoint"][-1]

    assert len(files) == 1, files
    (webhook_file,) = files
    assert f"> {webhook_file}\n" in script, script
    assert any(
        mount.startswith(str(PurePosixPath(webhook_file).parent) + ":")
        for mount in service["tmpfs"]
    )
    assert "webhook_url:" not in (DEPLOY / "alertmanager" / "alertmanager.yml").read_text(
        encoding="utf-8"
    ), "a webhook URL in the committed configuration"
