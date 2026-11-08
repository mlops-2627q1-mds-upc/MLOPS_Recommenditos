"""The drift job in the stack: a loop of passes, its limits, and its results in Prometheus.

The `monitor` service (deploy/compose.yaml, docs/docs/monitoring.md#drift-job)
runs one pass of `run-once` at its start and then every 15 minutes; between
passes no Python process runs. Its results reach Prometheus through
node-exporter's textfile collector, from a volume that is new in every CI run,
so these tests also show that the job, which is not root, can write it.

A test pass runs on a copy of the four requests the test sends, each with an
observed price, with windows of 4 and a monitor.db of its own in the
container's /tmp, so that it tests exactly one window however many requests
the stack's log holds; it rewrites the metrics, which the next scheduled pass
rewrites again.
"""

import csv
import hashlib
import json
from pathlib import Path
import time

import pytest
from tests.integration.conftest import Container, docker
from tests.integration.test_deployed_prometheus import eventually, query

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: Docker's init, the shell's loop and its sleep: what runs between two passes.
BETWEEN_PASSES = {"/sbin/docker-init", "sh", "sleep"}
WINDOW = 4
REPOSITORY = Path(__file__).parents[2]


def processes(container: Container) -> list[str]:
    """The program of every process of the container, as the host's `ps` sees them."""
    # `docker top` needs the PID column to map the host's processes to the container.
    listing = docker("top", container.id, "-o", "pid,args").splitlines()[1:]
    return [line.split()[1] for line in listing if line.strip()]


#: Copies the requests a test sent into a log of their own, with the API's tables,
#: so that the test pass sees exactly one window, however busy the stack's log is,
#: and gives each an observed price 10 % above its estimate, as feedback would.
COPY_REQUESTS = """
import json, sqlite3, sys
ids = json.loads(sys.argv[1])
source = sqlite3.connect("file:/var/lib/recommenditos/prediction-log.sqlite3?mode=ro", uri=True)
target = sqlite3.connect("/tmp/integration-log.sqlite3")
target.execute("DROP TABLE IF EXISTS feedback")
target.execute("DROP TABLE IF EXISTS predictions")
for table in ("predictions", "feedback"):
    (schema,) = source.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    target.execute(schema)
marks = ", ".join("?" for _ in ids)
rows = source.execute(f"SELECT * FROM predictions WHERE request_id IN ({marks})", ids).fetchall()
target.executemany(f"INSERT INTO predictions VALUES ({', '.join('?' for _ in rows[0])})", rows)
target.execute(
    "INSERT INTO feedback SELECT request_id, json_extract(outputs, '$.price_eur') * 1.1, "
    "answered_at FROM predictions"
)
target.commit()
print(len(rows))
"""


#: The lock the test pass reads: the image's own, when it names the served
#: bundle's version, as in CD, where the stack serves the real bundle; otherwise
#: one that names it, written into the container's /tmp, so that the pass still
#: records the image's test values for the served model: in CI the stack serves
#: a synthetic bundle, which the image's dvc.lock does not name. Prints its path.
LOCK_FOR_SERVED_MODEL = """
import sys
from pathlib import Path
from recommenditos_monitoring.performance import bundle_versions
variant, version = sys.argv[1:]
image = Path("/app/dvc.lock")
if bundle_versions(image).get(variant) == version:
    print(image)
else:
    written = Path("/tmp/integration-dvc.lock")
    written.write_text(
        f"stages:\\n  train@{variant}:\\n    outs:\\n"
        f"    - path: models/{variant}\\n      md5: {version}.dir\\n",
        encoding="utf-8",
    )
    print(written)
"""


def deployed_test_values() -> tuple[float, float]:
    """The deployed model's test MdAPE and full-input coverage, as the evaluate stage wrote them.

    Read at test time, since a retrain commits new ones (retrain.yml).
    """
    metrics = json.loads((REPOSITORY / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["deployable_variant"]
    table = REPOSITORY / "reports" / "metrics" / "interval-coverage-by-make.csv"
    with table.open(newline="", encoding="utf-8") as rows:
        (coverage,) = (
            float(row["coverage"])
            for row in csv.DictReader(rows)
            if (row["variant"], row["scenario"], row["make"]) == (variant, "full", "(pooled)")
        )
    return float(metrics["variants"][variant]["mdape"]), coverage


@pytest.fixture(scope="module")
def monitor(stack_container) -> Container:
    return stack_container("monitor")


@pytest.mark.req("NFR-04")
def test_no_python_process_runs_between_two_passes(monitor):
    # The pass at the container's start may still be running.
    programs = eventually(
        lambda: found if "python" not in (found := processes(monitor)) else None,
        "the first pass ended",
    )

    assert set(programs) <= BETWEEN_PASSES, programs


@pytest.mark.req("NFR-04")
def test_the_drift_job_is_limited_and_has_no_healthcheck(monitor):
    inspected = monitor.inspect()
    host = inspected["HostConfig"]

    assert host["Memory"] == 512 * 1024 * 1024
    assert host["NanoCpus"] == 500_000_000
    assert host["RestartPolicy"]["Name"] == "unless-stopped"
    # With no healthcheck, `up --wait` waits only for the container to run, so a
    # failing pass cannot fail a deploy.
    assert inspected["Config"].get("Healthcheck") is None
    assert "Health" not in inspected["State"]


@pytest.mark.req("NFR-09")
def test_the_monitoring_image_has_no_pillow(monitor):
    """monitoring/Dockerfile removes it: Alibi Detect's Pillow<11 has fixed HIGH findings."""
    found = monitor.exec(
        "python", "-c", "import importlib.util; print(importlib.util.find_spec('PIL'))"
    )
    assert found == "None"


def test_the_monitoring_image_has_the_releases_dvc_lock(monitor):
    """It names the model version of each variant's test values (#307)."""
    found = monitor.exec(
        "python",
        "-c",
        "import hashlib; print(hashlib.md5(open('/app/dvc.lock', 'rb').read()).hexdigest())",
    )
    expected = hashlib.md5((REPOSITORY / "dvc.lock").read_bytes(), usedforsecurity=False)
    assert found == expected.hexdigest()


def test_the_monitoring_image_has_the_releases_segments(monitor):
    """They hold the test MdAPE per country, a window's market baseline (#306)."""
    found = monitor.exec(
        "python",
        "-c",
        "import hashlib; print(hashlib.md5(open('/app/segments.csv', 'rb').read()).hexdigest())",
    )
    segments = REPOSITORY / "reports" / "metrics" / "segments.csv"
    expected = hashlib.md5(segments.read_bytes(), usedforsecurity=False)
    assert found == expected.hexdigest()


@pytest.mark.req("FR-14")
def test_the_pass_at_the_start_succeeded(monitor):
    eventually(
        lambda: True if any("open window:" in line for line in monitor.log_lines()) else None,
        "the monitor's log shows a pass that ended",
    )
    assert not [line for line in monitor.log_lines() if "the drift pass failed" in line]


@pytest.mark.req("FR-14")
def test_a_pass_reaches_prometheus_through_node_exporter(
    monitor, prometheus_container, client, listing, health
):
    ids = []
    for _ in range(WINDOW):
        answer = client.post("/v1/predict", json=listing, headers={"X-Client": "replay"})
        assert answer.status_code == 200, answer.text
        ids.append(answer.json()["request_id"])
    assert monitor.exec("python", "-c", COPY_REQUESTS, json.dumps(ids)) == str(WINDOW)
    served = health.json()["model"]
    lock = monitor.exec(
        "python", "-c", LOCK_FOR_SERVED_MODEL, served["variant"], served["version"]
    )
    started = time.time()

    out = monitor.exec(
        "python",
        "-m",
        "recommenditos_monitoring",
        "run-once",
        "--log",
        "/tmp/integration-log.sqlite3",  # noqa: S108 - the container's own tmpfs
        "--db",
        "/tmp/integration-monitor.db",  # noqa: S108 - the container's own tmpfs
        "--lock",
        lock,
        "--window-size",
        str(WINDOW),
    )
    assert "tested window 1: seq" in out, out
    assert f"open window: 0 of {WINDOW}" in out, out

    p_values = eventually(
        lambda: (
            result
            if len(result := query(prometheus_container, "recommenditos_drift_feature_p_value"))
            == 23
            else None
        ),
        "23 p-values in Prometheus",
    )
    assert {series["metric"]["job"] for series in p_values} == {"node"}
    success = eventually(
        lambda: (
            result
            if (
                result := query(
                    prometheus_container, "recommenditos_monitor_last_success_timestamp_seconds"
                )
            )
            and float(result[0]["value"][1]) >= started
            else None
        ),
        "the test pass's success in Prometheus",
    )
    assert len(success) == 1
    errors = query(prometheus_container, "node_textfile_scrape_error")
    assert [float(series["value"][1]) for series in errors] == [0.0]

    # The window's error and coverage beside its drift (#229), every series of
    # them: four labels of four point estimates are too few for a figure, which
    # is then left out, and the test values are the image's copies of the
    # evaluate outputs, the repository's, recorded for the served model.
    mdape, coverage = deployed_test_values()
    exported = {
        (series["metric"]["__name__"], series["metric"].get("figure")): float(series["value"][1])
        for series in query(prometheus_container, '{__name__=~"recommenditos_performance_.+"}')
    }
    assert exported == {
        ("recommenditos_performance_window_id", None): 1,
        ("recommenditos_performance_requests", None): WINDOW,
        ("recommenditos_performance_ranges", None): 0,
        ("recommenditos_performance_labelled_requests", None): WINDOW,
        ("recommenditos_performance_labelled_ranges", None): 0,
        ("recommenditos_performance_labels_sufficient", "mdape"): 0,
        ("recommenditos_performance_labels_sufficient", "coverage"): 0,
        ("recommenditos_performance_min_labels", "mdape"): 200,
        ("recommenditos_performance_min_labels", "coverage"): 300,
        ("recommenditos_performance_test_mdape", None): pytest.approx(mdape),
        ("recommenditos_performance_test_coverage", None): pytest.approx(coverage),
        ("recommenditos_performance_final", None): 0,
    }
    # Left out without labels: the two figures, the level the ranges state, and
    # the market baseline of the MdAPE.
    absent = {"mdape", "coverage", "nominal_level", "baseline_mdape", "baseline_test_rows"}
    assert not {name for name, _ in exported} & {f"recommenditos_performance_{n}" for n in absent}
