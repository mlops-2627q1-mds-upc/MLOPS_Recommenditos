"""The integration suite: the API as deployed, the built image behind nginx, over HTTP.

    make test-integration                                   # the local stack (make stack)
    make smoke SMOKE_URL=http://<vm-host>:<port>            # after a deployment

The in-process tests under `tests/api/` run the app with `TestClient`; this
suite sends real requests through the proxy, so it sees what they cannot: the
image as built (runtime dependencies only, non-root, the baked-in bundle), the
nginx rules, the proxy's body limit beside the middleware's, and the Compose
healthcheck and restart policy (docs/docs/deployment.md).

Every test here is marked `integration`, which `make test` deselects. The base
URL comes from `RECOMMENDITOS_STACK_URL` and defaults to the local stack. Tests
marked `docker_host` inspect or kill containers, so they need the machine that
runs the stack: they run when the base URL is a loopback address and are skipped
otherwise. `make smoke`, the post-deployment check, deselects them whatever the
URL, so it never kills or inspects a container (docs/docs/deployment.md).

The model version `/health` reports is compared with `dvc.lock`, as FR-12 asks,
unless `RECOMMENDITOS_EXPECTED_MODEL_VERSION` names another one: CI builds the
image around a synthetic bundle, whose version `tests/fixtures/synthetic_bundle.py`
prints.

A new endpoint adds one file here, `test_deployed_<endpoint>.py`, with one valid
request through the proxy and its documented shape; `test_deployed_predict.py`
is the example to copy. Error shapes, limits and proxy rules are already covered
for every route by `test_deployed_proxy.py`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from urllib.parse import urlsplit

import httpx2
import pytest
import yaml

from recommenditos.config import PROJ_ROOT

STACK_URL_ENV_VAR = "RECOMMENDITOS_STACK_URL"
EXPECTED_VERSION_ENV_VAR = "RECOMMENDITOS_EXPECTED_MODEL_VERSION"
DEFAULT_STACK_URL = "http://localhost:8080"

#: The Compose project of deploy/compose.yaml (`name:`), on the laptop, in CI and
#: on the VM alike, so its containers are found by label without the file.
COMPOSE_PROJECT = "recommenditos"

#: Long enough for a valuation on a cold API, short enough that a hung proxy
#: fails the run instead of stalling it.
TIMEOUT_S = 10

#: A listing with the ten fields FR-01 requires. The categoricals the bundle
#: restricts are replaced by values it accepts (see `listing`), so the same
#: listing works with the real bundle and with CI's synthetic one.
LISTING = {
    "make": "Volkswagen",
    "model": "Golf",
    "registration_date": "2019-06",
    "mileage_km_raw": 45000,
    "power_kw": 110,
    "fuel_category": "Gasoline",
    "transmission": "Manual",
    "country_code": "DE",
    "body_type": "Compact",
    "seller_type": "Dealer",
}


def stack_url() -> str:
    return os.environ.get(STACK_URL_ENV_VAR, DEFAULT_STACK_URL).rstrip("/")


def is_local(url: str) -> bool:
    """Whether `url` points at this machine, where the stack's containers are."""
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if is_local(stack_url()):
        return
    skip = pytest.mark.skip(reason=f"needs the Docker host; {stack_url()} is remote")
    for item in items:
        if "docker_host" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_url() -> str:
    return stack_url()


@pytest.fixture(scope="session")
def client(base_url: str) -> Iterator[httpx2.Client]:
    """A client of the proxy, which every request of the suite goes through."""
    with httpx2.Client(base_url=base_url, timeout=TIMEOUT_S) as session:
        yield session


@pytest.fixture(scope="session")
def health(client: httpx2.Client) -> httpx2.Response:
    """`GET /health`, asked once; the first failure says when nothing answers."""
    try:
        return client.get("/health")
    except httpx2.TransportError as error:
        pytest.fail(f"no answer from {client.base_url}: {error!r}; is the stack up?")


@pytest.fixture(scope="session")
def expected_version(health: httpx2.Response) -> str:
    """The model version the image has to report: the override, or dvc.lock's hash."""
    if override := os.environ.get(EXPECTED_VERSION_ENV_VAR):
        return override
    variant = health.json()["model"]["variant"]
    version = bundle_hash(PROJ_ROOT / "dvc.lock", variant)
    assert version is not None, f"dvc.lock records no models/{variant}"
    return version


def bundle_hash(lock: Path, variant: str) -> str | None:
    """The MD5 `dvc.lock` records for `models/<variant>`, without DVC's `.dir` suffix."""
    stages = yaml.safe_load(lock.read_text(encoding="utf-8"))["stages"]
    for stage in stages.values():
        for out in stage.get("outs", []):
            if out["path"] == f"models/{variant}":
                return str(out["md5"]).removesuffix(".dir")
    return None


@pytest.fixture(scope="session")
def openapi(client: httpx2.Client) -> dict[str, Any]:
    response = client.get("/openapi.json")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def listing(openapi: dict[str, Any]) -> dict[str, Any]:
    """A valid `/v1/predict` body for the bundle the image serves.

    The values a categorical accepts are the bundle's (FR-03, FR-04), which the
    served OpenAPI document lists as enums; a field of `LISTING` whose value the
    bundle does not accept takes the first one it does. A fresh dict per test.
    """
    properties = openapi["components"]["schemas"]["ValuationRequest"]["properties"]
    body = dict(LISTING)
    for field, value in LISTING.items():
        branches = properties[field].get("anyOf", [properties[field]])
        allowed = next((branch["enum"] for branch in branches if "enum" in branch), None)
        if allowed is not None and value not in allowed:
            body[field] = allowed[0]
    return body


@dataclass(frozen=True)
class Container:
    """One running container of the stack, driven with the Docker CLI."""

    id: str

    def inspect(self) -> dict[str, Any]:
        return json.loads(docker("inspect", self.id))[0]

    def exec(self, *command: str) -> str:
        return docker("exec", self.id, *command)

    def log_lines(self) -> list[str]:
        """The container's log, stdout and stderr, which `docker logs` passes on apart."""
        result = subprocess.run(
            [_docker_executable(), "logs", self.id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"docker logs {self.id} failed: {result.stderr.strip()}")
        return (result.stdout + result.stderr).splitlines()


def docker(*arguments: str) -> str:
    """Run the Docker CLI; a failure fails the test with what Docker said."""
    result = subprocess.run(
        [_docker_executable(), *arguments], capture_output=True, text=True, timeout=60, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"docker {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _docker_executable() -> str:
    executable = shutil.which("docker")
    if executable is None:
        pytest.fail("the docker CLI is not on PATH, and docker_host tests need it")
    return executable


def service_container(service: str) -> Container:
    ids = docker(
        "ps",
        "--quiet",
        "--filter",
        f"label=com.docker.compose.project={COMPOSE_PROJECT}",
        "--filter",
        f"label=com.docker.compose.service={service}",
    ).split()
    assert len(ids) == 1, (
        f"expected one running {service!r} container of the {COMPOSE_PROJECT!r} stack, "
        f"found {len(ids)}; is the stack up (make stack)?"
    )
    return Container(ids[0])


@pytest.fixture(scope="session")
def api_container() -> Container:
    return service_container("api")


@pytest.fixture(scope="session")
def proxy_container() -> Container:
    return service_container("proxy")
