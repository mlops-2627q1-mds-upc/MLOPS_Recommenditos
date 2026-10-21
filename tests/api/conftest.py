"""Fixtures for the API tests: an app around a synthetic bundle, and a valid listing.

The bundle is the `lgbm-basic` variant that `tests/conftest.py`'s `trained`
fixture fits on the synthetic fixture, because CI cannot `dvc pull` the real one
(see `tests/test_serving.py`). A test of a new endpoint takes `client` and
`listing` and needs nothing else; `assert_problem` in `tests/api/helpers.py`
checks the error shape every error response shares (FR-16). The app writes its
prediction log to `log_file`, in a temporary directory, and its `/v1/feedback`
takes `API_KEY`.
"""

from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from recommenditos.api.app import create_app
from recommenditos.api.serving import ServedModel
from recommenditos.api.settings import Settings

#: The variant the API serves by default (FR-12).
SERVED_VARIANT = "lgbm-basic"

#: The key the test app's `/v1/feedback` requires; a test value, not a secret.
API_KEY = "test" + "-key" * 5


@pytest.fixture(scope="session")
def bundle(trained: dict) -> Path:
    """The served variant's bundle directory, fitted on the synthetic fixture."""
    return Path(trained["dir"]) / SERVED_VARIANT


@pytest.fixture(scope="session")
def log_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The SQLite file the session's app records its answers and the feedback in."""
    return tmp_path_factory.mktemp("prediction-log") / "prediction-log.sqlite3"


@pytest.fixture(scope="session")
def app(bundle: Path, log_file: Path) -> FastAPI:
    return create_app(bundle, Settings(prediction_log=log_file, api_key=SecretStr(API_KEY)))


@pytest.fixture(scope="session")
def served(app: FastAPI) -> ServedModel:
    """The model the app loaded, to read what it accepts."""
    model = app.state.served
    assert isinstance(model, ServedModel)
    return model


@pytest.fixture(scope="session")
def client(app: FastAPI) -> Iterator[TestClient]:
    """A client that turns an unexpected exception into the 500 a real client gets."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def listing(served: ServedModel) -> dict:
    """A valid `/v1/predict` body with the ten required fields of FR-01.

    Every categorical is a value the synthetic bundle knows, taken from the
    bundle rather than written down, so the fixture follows the generator.
    A fresh dict per test, so a test can change it freely.
    """
    levels = served.model.input_schema.levels
    return {
        "make": served.allowed["make"][0],
        "model": levels["model"][0],
        "registration_date": "2019-06",
        "mileage_km_raw": 45000,
        "power_kw": 110,
        "fuel_category": served.allowed["fuel_category"][0],
        "transmission": served.allowed["transmission"][0],
        "country_code": levels["country_code"][0],
        "body_type": served.allowed["body_type"][0],
        "seller_type": served.allowed["seller_type"][0],
    }
