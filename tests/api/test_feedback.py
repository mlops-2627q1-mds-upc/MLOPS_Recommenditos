"""`POST /v1/feedback`: an observed price against a request ID (FR-10), and its key (NFR-15)."""

import io
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import SecretStr, ValidationError
import pytest
from tests.api.conftest import API_KEY
from tests.api.helpers import assert_problem

from recommenditos.api.app import create_app
from recommenditos.api.settings import API_KEY_ENV_VAR, MIN_API_KEY_LENGTH, Settings
from recommenditos.config import PROJ_ROOT

FEEDBACK = "/v1/feedback"
PREDICT = "/v1/predict"
KEY = {"X-API-Key": API_KEY}


def _answer(client: TestClient, listing: dict) -> dict:
    response = client.post(PREDICT, json=listing)
    assert response.status_code == 200, response.text
    return dict(response.json())


def _feedback(request_id: str, price: float) -> dict:
    return {"request_id": request_id, "observed_price_eur": price}


# --------------------------------------------------------------------------
# FR-10: the API key
# --------------------------------------------------------------------------


@pytest.mark.req("FR-10", "FR-16")
def test_fr10_rejects_missing_api_key(client: TestClient, listing: dict):
    answer = _answer(client, listing)

    response = client.post(FEEDBACK, json=_feedback(answer["request_id"], 20000.0))

    assert_problem(response, 401)
    assert response.headers["WWW-Authenticate"] == "APIKey"


@pytest.mark.req("FR-10", "FR-16")
@pytest.mark.parametrize("key", ["wrong-key-0123456789abcdef", API_KEY[:-1], API_KEY + "x", ""])
def test_fr10_rejects_a_wrong_api_key(key: str, client: TestClient, listing: dict):
    answer = _answer(client, listing)

    response = client.post(
        FEEDBACK,
        json=_feedback(answer["request_id"], 20000.0),
        headers={"X-API-Key": key},
    )

    assert_problem(response, 401)


@pytest.mark.req("FR-10")
def test_fr10_the_key_is_checked_before_the_body_is_validated(client: TestClient):
    """A caller without the key learns nothing about the fields the endpoint accepts.

    FastAPI parses the JSON before the dependencies run, so a body that is not
    JSON is still a 422, and the size limit's 413 comes first of all.
    """
    response = client.post(FEEDBACK, json={"nonsense": True})

    assert_problem(response, 401)


@pytest.mark.req("FR-10")
def test_fr10_a_service_started_without_a_key_refuses_every_request(bundle: Path, tmp_path: Path):
    app = create_app(bundle, Settings(prediction_log=tmp_path / "log.sqlite3", api_key=None))

    with TestClient(app) as client:
        for headers in ({}, {"X-API-Key": ""}, KEY):
            response = client.post(
                FEEDBACK, json=_feedback(str(uuid.uuid4()), 1.0), headers=headers
            )
            assert_problem(response, 401)


@pytest.mark.req("FR-10", "FR-16")
def test_fr10_the_openapi_document_declares_the_key(app: FastAPI):
    document = app.openapi()
    schemes = document["components"]["securitySchemes"]
    (name,) = [name for name, scheme in schemes.items() if scheme["type"] == "apiKey"]

    assert schemes[name]["in"] == "header"
    assert schemes[name]["name"] == "X-API-Key"
    assert document["paths"][FEEDBACK]["post"]["security"] == [{name: []}]
    assert "security" not in document["paths"][PREDICT]["post"]


# --------------------------------------------------------------------------
# FR-10: storing a price and joining it (FR-14)
# --------------------------------------------------------------------------


@pytest.mark.req("FR-10", "FR-14")
def test_fr10_a_stored_price_joins_its_answer_by_request_id(
    client: TestClient, listing: dict, log_file: Path
):
    answer = _answer(client, listing)

    response = client.post(FEEDBACK, json=_feedback(answer["request_id"], 20500.0), headers=KEY)

    assert response.status_code == 201, response.text
    receipt = response.json()
    assert (receipt["request_id"], receipt["observed_price_eur"]) == (
        answer["request_id"],
        20500.0,
    )
    # The join M6's drift job runs, through the view and as plain SQL.
    connection = sqlite3.connect(log_file)
    try:
        joined = connection.execute(
            "SELECT json_extract(p.outputs, '$.price_eur'), f.observed_price_eur, "
            "p.model_version FROM predictions AS p JOIN feedback AS f USING (request_id) "
            "WHERE p.request_id = ?",
            (answer["request_id"],),
        ).fetchall()
        viewed = connection.execute(
            "SELECT observed_price_eur FROM labelled_predictions WHERE request_id = ?",
            (answer["request_id"],),
        ).fetchall()
    finally:
        connection.close()
    assert joined == [(answer["price_eur"], 20500.0, answer["model"]["version"])]
    assert viewed == [(20500.0,)]


@pytest.mark.req("FR-10")
def test_fr10_the_same_price_again_changes_nothing(client: TestClient, listing: dict):
    """The replay job may retry a request whose answer it lost."""
    answer = _answer(client, listing)
    body = _feedback(answer["request_id"], 18000.0)
    first = client.post(FEEDBACK, json=body, headers=KEY)

    again = client.post(FEEDBACK, json=body, headers=KEY)

    assert (first.status_code, again.status_code) == (201, 200)
    assert again.json() == first.json()


@pytest.mark.req("FR-10")
def test_fr10_another_price_for_the_same_answer_is_refused(client: TestClient, listing: dict):
    answer = _answer(client, listing)
    first = client.post(FEEDBACK, json=_feedback(answer["request_id"], 18000.0), headers=KEY)

    response = client.post(FEEDBACK, json=_feedback(answer["request_id"], 18500.0), headers=KEY)

    assert first.status_code == 201
    problem = assert_problem(response, 409)
    assert "18000.0" in problem["detail"]


@pytest.mark.req("FR-10")
def test_fr10_a_price_for_an_unknown_request_id_is_not_stored(client: TestClient, log_file: Path):
    request_id = str(uuid.uuid4())

    response = client.post(FEEDBACK, json=_feedback(request_id, 18000.0), headers=KEY)

    assert_problem(response, 404)
    connection = sqlite3.connect(log_file)
    try:
        stored = connection.execute(
            "SELECT count(*) FROM feedback WHERE request_id = ?", (request_id,)
        ).fetchone()
    finally:
        connection.close()
    assert stored == (0,)


@pytest.mark.req("FR-10", "FR-03")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_price_eur", 0),
        ("observed_price_eur", -5.0),
        ("observed_price_eur", "NaN"),
        ("request_id", "not-a-uuid"),
    ],
)
def test_fr10_an_invalid_body_is_a_422(field: str, value: object, client: TestClient):
    body = {"request_id": str(uuid.uuid4()), "observed_price_eur": 18000.0, field: value}

    response = client.post(FEEDBACK, json=body, headers=KEY)

    problem = assert_problem(response, 422)
    assert [error["pointer"] for error in problem["errors"]] == [f"#/{field}"]


def test_feedback_is_not_an_answer_and_writes_no_record(
    client: TestClient, listing: dict, log_file: Path
):
    answer = _answer(client, listing)
    connection = sqlite3.connect(log_file)
    try:
        (before,) = connection.execute("SELECT count(*) FROM predictions").fetchone()
        client.post(FEEDBACK, json=_feedback(answer["request_id"], 1.0), headers=KEY)
        (after,) = connection.execute("SELECT count(*) FROM predictions").fetchone()
    finally:
        connection.close()
    assert after == before


# --------------------------------------------------------------------------
# NFR-15: the key is a secret
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-15")
def test_nfr15_the_key_never_appears_in_the_settings_repr_or_the_startup_log(
    bundle: Path, tmp_path: Path
):
    secret = "s3cr3t-" + "k" * MIN_API_KEY_LENGTH
    settings = Settings(prediction_log=tmp_path / "log.sqlite3", api_key=SecretStr(secret))
    captured = io.StringIO()
    sink = logger.add(captured, level="DEBUG")
    try:
        create_app(bundle, settings).state.store.close()
    finally:
        logger.remove(sink)

    for text in (repr(settings), str(settings), settings.model_dump_json(), captured.getvalue()):
        assert secret not in text


@pytest.mark.req("NFR-15")
def test_nfr15_the_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, API_KEY)

    key = Settings().api_key

    assert key is not None
    assert key.get_secret_value() == API_KEY


@pytest.mark.req("NFR-15")
def test_nfr15_the_template_lists_the_key_without_a_value():
    lines = (PROJ_ROOT / ".env.template").read_text(encoding="utf-8").splitlines()

    assert f"{API_KEY_ENV_VAR}=" in [line.strip() for line in lines]


@pytest.mark.req("FR-10")
def test_fr10_an_empty_key_means_no_key_and_a_short_one_stops_the_service(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(API_KEY_ENV_VAR, "")
    assert Settings().api_key is None

    monkeypatch.setenv(API_KEY_ENV_VAR, "changeme")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.req("NFR-15")
def test_nfr15_a_refused_key_is_not_repeated_in_the_error_or_the_startup_output(
    monkeypatch: pytest.MonkeyPatch,
):
    refused = "short-leak-xyz"
    monkeypatch.setenv(API_KEY_ENV_VAR, refused)

    with pytest.raises(ValidationError) as error:
        Settings()
    startup = subprocess.run(
        [sys.executable, "-c", "from recommenditos.api.settings import Settings; Settings()"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, API_KEY_ENV_VAR: refused, "MLFLOW_TRACKING_URI": ""},
    )

    assert startup.returncode != 0
    assert "api_key" in startup.stderr
    for text in (str(error.value), repr(error.value), startup.stdout, startup.stderr):
        assert refused not in text
