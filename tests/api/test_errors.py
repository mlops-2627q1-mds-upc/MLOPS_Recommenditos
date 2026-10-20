"""One error shape for every error path (FR-16), the body-size limit among them (NFR-09)."""

import asyncio
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from tests.api.helpers import assert_problem

from recommenditos.api.errors import PAYLOAD_TOO_LARGE, VALIDATION_ERROR
from recommenditos.api.middleware import MAX_BODY_BYTES
from recommenditos.api.serving import ServedModel

PREDICT = "/v1/predict"


# --------------------------------------------------------------------------
# NFR-09: the body-size limit
# --------------------------------------------------------------------------


def _call(app: FastAPI, headers: list[tuple[bytes, bytes]]) -> tuple[list[dict], int]:
    """Send a request straight to the ASGI app; return the messages sent and the body reads."""
    reads = 0
    sent: list[dict] = []

    async def receive() -> dict:
        nonlocal reads
        reads += 1
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": PREDICT,
        "raw_path": PREDICT.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    return sent, reads


@pytest.mark.req("NFR-09", "FR-16")
def test_nfr09_a_declared_body_over_10_kb_is_refused_before_it_is_read(app: FastAPI):
    sent, reads = _call(
        app,
        [
            (b"content-type", b"application/json"),
            (b"content-length", str(MAX_BODY_BYTES + 1).encode()),
        ],
    )

    assert reads == 0
    assert sent[0]["status"] == 413


@pytest.mark.req("NFR-09", "FR-16")
def test_nfr09_a_body_over_10_kb_is_refused_in_the_shared_shape(client: TestClient):
    response = client.post(
        PREDICT,
        content=b" " * (MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )

    assert assert_problem(response, 413)["type"] == PAYLOAD_TOO_LARGE


@pytest.mark.req("NFR-09")
def test_nfr09_a_chunked_body_is_cut_off_once_it_passes_10_kb(client: TestClient):
    """Without `Content-Length` the body is counted as it arrives."""

    def chunks():
        for _ in range(8):
            yield b" " * 2048

    response = client.post(PREDICT, content=chunks(), headers={"content-type": "application/json"})

    assert response.request.headers.get("transfer-encoding") == "chunked"
    assert assert_problem(response, 413)["type"] == PAYLOAD_TOO_LARGE


@pytest.mark.req("NFR-09")
def test_nfr09_a_body_of_exactly_10_kb_is_read(client: TestClient, listing: dict):
    body = __import__("json").dumps(listing).encode()
    padded = body + b" " * (MAX_BODY_BYTES - len(body))

    response = client.post(PREDICT, content=padded, headers={"content-type": "application/json"})

    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------
# FR-16: every error path, one shape
# --------------------------------------------------------------------------


def _missing_field(client: TestClient, listing: dict):
    return client.post(
        PREDICT, json={key: value for key, value in listing.items() if key != "make"}
    )


def _unsupported_make(client: TestClient, listing: dict):
    return client.post(PREDICT, json={**listing, "make": "Trabant"})


def _not_json(client: TestClient, listing: dict):
    return client.post(PREDICT, content=b"{make:", headers={"content-type": "application/json"})


def _not_an_object(client: TestClient, listing: dict):
    return client.post(PREDICT, json=[listing])


def _too_large(client: TestClient, listing: dict):
    return client.post(
        PREDICT, content=b" " * 20_000, headers={"content-type": "application/json"}
    )


def _unknown_route(client: TestClient, listing: dict):
    return client.get("/v1/nothing-here")


def _wrong_method(client: TestClient, listing: dict):
    return client.get(PREDICT)


_PATHS: list = [
    pytest.param(_missing_field, 422, id="missing-field"),
    pytest.param(_unsupported_make, 422, id="unsupported-make"),
    pytest.param(_not_json, 422, id="body-not-json"),
    pytest.param(_not_an_object, 422, id="body-not-an-object"),
    pytest.param(_too_large, 413, id="body-too-large"),
    pytest.param(_unknown_route, 404, id="unknown-route"),
    pytest.param(_wrong_method, 405, id="wrong-method"),
]


@pytest.mark.req("FR-16")
@pytest.mark.parametrize(("request_it", "status"), _PATHS)
def test_fr16_every_error_path_returns_the_same_shape(
    request_it: Callable, status: int, client: TestClient, listing: dict
):
    problem = assert_problem(request_it(client, listing), status)

    if status == 422:
        assert problem["type"] == VALIDATION_ERROR
        assert problem["errors"]


@pytest.mark.req("FR-16")
def test_fr16_an_unexpected_failure_returns_the_same_shape_and_hides_the_cause(
    client: TestClient, listing: dict, served: ServedModel, monkeypatch: pytest.MonkeyPatch
):
    def fail(*_args, **_kwargs):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(ServedModel, "value", fail)

    problem = assert_problem(client.post(PREDICT, json=listing), 500)

    assert "secret" not in problem["detail"]
