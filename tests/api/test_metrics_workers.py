"""`/metrics` adds up every worker's figures (FR-11, NFR-03).

Starts the service the way the image does, `python -m recommenditos.api`, with
two uvicorn worker processes and a multiprocess directory, sends it valuations
until both workers have answered some, and checks that a scrape, which one
worker answers, reports every request.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from prometheus_client.parser import text_string_to_metric_families
import pytest
from tests.api.helpers import free_port, stop_service

from recommenditos.api.metrics import MULTIPROC_DIR_ENV_VAR
from recommenditos.api.settings import MODEL_DIR_ENV_VAR

WORKERS = 2
STARTUP_SECONDS = 120
#: Valuations sent per round; rounds repeat until every worker has answered.
ROUND = 40
MAX_ROUNDS = 10


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode()


def _post(url: str, body: dict) -> int:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.status)


def _pids(directory: Path, kind: str) -> set[str]:
    """The PIDs of the workers that wrote a `<kind>_<pid>.db` file (prometheus_client's names)."""
    return {path.stem.rsplit("_", 1)[1] for path in directory.glob(f"{kind}_*.db")}


@pytest.fixture
def service(bundle: Path, tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """The base URL of the service with two workers, and its multiprocess directory."""
    directory = tmp_path / "prometheus"
    directory.mkdir()
    # A file left by an earlier run: the launcher must delete it before the workers start.
    (directory / "counter_1.db").write_bytes(b"stale")
    port = free_port()
    env = {
        **os.environ,
        MULTIPROC_DIR_ENV_VAR: str(directory),
        MODEL_DIR_ENV_VAR: str(bundle),
        "MLFLOW_TRACKING_URI": "",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "recommenditos.api", "--host", "127.0.0.1"]
        + ["--port", str(port), "--workers", str(WORKERS), "--log-level", "warning"],
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_SECONDS
        while len(_pids(directory, "gauge_max")) < WORKERS:
            assert process.poll() is None, "the service exited during startup"
            assert time.monotonic() < deadline, "the workers did not start in time"
            time.sleep(0.5)
        _get(f"{base}/health")
        yield base, directory
    finally:
        stop_service(process)


@pytest.mark.req("FR-11", "NFR-03")
def test_fr11_metrics_add_up_across_two_worker_processes(service: tuple[str, Path], listing: dict):
    base, directory = service
    assert not (directory / "counter_1.db").exists()

    sent = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(MAX_ROUNDS):
            statuses = list(pool.map(lambda _: _post(f"{base}/v1/predict", listing), range(ROUND)))
            assert statuses == [200] * ROUND
            sent += ROUND
            if len(_pids(directory, "counter")) >= WORKERS:
                break
    assert len(_pids(directory, "counter")) == WORKERS, "not every worker answered a request"

    scrapes = [
        {
            (sample.name, frozenset(sample.labels.items())): sample.value
            for family in text_string_to_metric_families(_get(f"{base}/metrics"))
            for sample in family.samples
        }
        for _ in range(4)
    ]

    ok = frozenset({"method": "POST", "path": "/v1/predict", "status": "200"}.items())
    for samples in scrapes:
        assert samples[("http_requests_total", ok)] == sent
        assert samples[("http_request_duration_seconds_count", ok)] == sent
        assert samples[("recommenditos_predicted_price_eur_count", frozenset())] == sent
        model = [
            value for (name, _), value in samples.items() if name == "recommenditos_model_info"
        ]
        assert model == [1]
