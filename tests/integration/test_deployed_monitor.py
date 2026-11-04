"""The drift job in the stack: a loop of passes, its limits, and its results in Prometheus.

The `monitor` service (deploy/compose.yaml, docs/docs/monitoring.md#drift-job)
runs one pass of `run-once` at its start and then every 15 minutes; between
passes no Python process runs. Its results reach Prometheus through
node-exporter's textfile collector, from a volume that is new in every CI run,
so these tests also show that the job, which is not root, can write it.

A test pass runs on a copy of the four requests the test sends, with windows of
4 and a monitor.db of its own in the container's /tmp, so that it tests exactly
one window however many requests the stack's log holds; it rewrites the
metrics, which the next scheduled pass rewrites again.
"""

import json
import time

import pytest
from tests.integration.conftest import Container, docker
from tests.integration.test_deployed_prometheus import eventually, query

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: Docker's init, the shell's loop and its sleep: what runs between two passes.
BETWEEN_PASSES = {"/sbin/docker-init", "sh", "sleep"}
WINDOW = 4


def processes(container: Container) -> list[str]:
    """The program of every process of the container, as the host's `ps` sees them."""
    # `docker top` needs the PID column to map the host's processes to the container.
    listing = docker("top", container.id, "-o", "pid,args").splitlines()[1:]
    return [line.split()[1] for line in listing if line.strip()]


#: Copies the requests a test sent into a log of their own, with the API's schema,
#: so that the test pass sees exactly one window, however busy the stack's log is.
COPY_REQUESTS = """
import json, sqlite3, sys
ids = json.loads(sys.argv[1])
source = sqlite3.connect("file:/var/lib/recommenditos/prediction-log.sqlite3?mode=ro", uri=True)
target = sqlite3.connect("/tmp/integration-log.sqlite3")
(schema,) = source.execute("SELECT sql FROM sqlite_master WHERE name = 'predictions'").fetchone()
target.execute("DROP TABLE IF EXISTS predictions")
target.execute(schema)
marks = ", ".join("?" for _ in ids)
rows = source.execute(f"SELECT * FROM predictions WHERE request_id IN ({marks})", ids).fetchall()
target.executemany(f"INSERT INTO predictions VALUES ({', '.join('?' for _ in rows[0])})", rows)
target.commit()
print(len(rows))
"""


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


@pytest.mark.req("FR-14")
def test_the_pass_at_the_start_succeeded(monitor):
    eventually(
        lambda: True if any("open window:" in line for line in monitor.log_lines()) else None,
        "the monitor's log shows a pass that ended",
    )
    assert not [line for line in monitor.log_lines() if "the drift pass failed" in line]


@pytest.mark.req("FR-14")
def test_a_pass_reaches_prometheus_through_node_exporter(
    monitor, prometheus_container, client, listing
):
    ids = []
    for _ in range(WINDOW):
        answer = client.post("/v1/predict", json=listing, headers={"X-Client": "replay"})
        assert answer.status_code == 200, answer.text
        ids.append(answer.json()["request_id"])
    assert monitor.exec("python", "-c", COPY_REQUESTS, json.dumps(ids)) == str(WINDOW)
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
