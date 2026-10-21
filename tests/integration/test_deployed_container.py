"""The running containers, inspected on the Docker host: user, packages, published ports."""

import json

import pytest
from tests.lockfile import PROJECT, normalise, read_lock, runtime_closure

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: Prints the distributions the API's interpreter can import, which is the
#: image's virtual environment: it is created without the system site-packages.
LIST_DISTRIBUTIONS = (
    "import importlib.metadata, json; "
    "print(json.dumps(sorted({d.metadata['Name'] for d in importlib.metadata.distributions()})))"
)


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
