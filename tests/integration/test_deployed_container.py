"""The running containers, inspected on the Docker host: user, packages, published ports."""

import hashlib
import json
from pathlib import Path

import pytest
from tests.lockfile import PROJECT, normalise, read_lock, runtime_closure

from recommenditos.config import PROJ_ROOT

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: Prints the distributions the API's interpreter can import, which is the
#: image's virtual environment: it is created without the system site-packages.
LIST_DISTRIBUTIONS = (
    "import importlib.metadata, json; "
    "print(json.dumps(sorted({d.metadata['Name'] for d in importlib.metadata.distributions()})))"
)

#: Prints the SHA-256 of every module of the installed `recommenditos` package,
#: by its path inside the package, without importing it.
PACKAGE_DIGESTS = (
    "import hashlib, importlib.util, json, pathlib; "
    "spec = importlib.util.find_spec('recommenditos'); "
    "root = pathlib.Path(spec.submodule_search_locations[0]); "
    "print(json.dumps({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() "
    "for p in root.rglob('*.py')}))"
)


def package_digests(root: Path) -> dict[str, str]:
    """The same digests as `PACKAGE_DIGESTS`, of the package in the checkout."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py")
    }


@pytest.mark.req("NFR-09")
@pytest.mark.parametrize("service", ["api", "proxy"])
def test_the_container_runs_as_a_non_root_user(service, request):
    container = request.getfixturevalue(f"{service}_container")
    uid = container.exec("id", "-u")
    assert uid.isdigit() and int(uid) != 0, f"{service} runs as uid {uid}"


@pytest.mark.req("NFR-09")
def test_the_image_holds_the_serving_runtime_and_no_dependency_group(api_container):
    """Every installed distribution is in the closure of `[project] dependencies` (EDN-47)."""
    installed = {
        normalise(name)
        for name in json.loads(api_container.exec("python", "-c", LIST_DISTRIBUTIONS))
    }
    runtime = runtime_closure(read_lock())

    assert {"fastapi", "lightgbm", "uvicorn"} <= installed, "the listing found no runtime"
    extra = sorted(installed - runtime - {PROJECT})
    assert not extra, f"the image installs packages outside the serving runtime: {extra}"


@pytest.mark.req("FR-10", "NFR-09")
def test_the_api_port_is_not_published_on_the_host(api_container):
    """Only nginx is reachable from outside; the API only inside the Compose network."""
    ports = api_container.inspect()["NetworkSettings"]["Ports"]
    published = {port: bindings for port, bindings in ports.items() if bindings}
    assert not published, f"the API container publishes {published}"


@pytest.mark.req("NFR-13")
def test_the_image_holds_the_package_of_the_checkout(api_container):
    """The suite tests the checked-out code, not a package built from another checkout.

    The image could install a stale wheel of the project from the shared uv cache
    of a laptop's earlier build (#180); this compares every module.
    """
    installed = json.loads(api_container.exec("python", "-c", PACKAGE_DIGESTS))
    checkout = package_digests(PROJ_ROOT / "recommenditos")

    assert "api/app.py" in checkout, "the checkout's package was not found"
    differ = sorted(
        path
        for path in installed.keys() | checkout.keys()
        if installed.get(path) != checkout.get(path)
    )
    assert not differ, (
        f"the image's recommenditos package differs from the checkout in {differ}: "
        "rebuild the image from this checkout (make stack)"
    )
