"""`GET /v1/makes`: the makes and the model names the web form suggests (FR-05)."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from tests.api.helpers import assert_problem

from recommenditos.api.app import create_app
from recommenditos.api.comparables import ComparablesIndex
from recommenditos.api.serving import ServedModel
from recommenditos.api.settings import Settings
from recommenditos.data.build_comparables import INDEX_FILE

MAKES = "/v1/makes"


@pytest.mark.req("FR-05")
def test_fr05_every_accepted_make_is_listed_with_models_of_both_index_and_model(
    client: TestClient, served: ServedModel, index: ComparablesIndex
):
    answer = client.get(MAKES).json()

    known = set(served.model.input_schema.levels["model"])
    listings = index.listings
    assert [entry["make"] for entry in answer["makes"]] == list(served.allowed["make"])
    for entry in answer["makes"]:
        in_index = set(listings.loc[listings["make"] == entry["make"], "model"].dropna())
        assert entry["models"] == sorted(in_index & known)
    assert any(entry["models"] for entry in answer["makes"])
    assert answer["model"] == served.info.model_dump(mode="json")


@pytest.mark.req("FR-05")
def test_fr05_a_listed_model_reaches_the_model_without_a_warning(
    client: TestClient, listing: dict
):
    entry = next(e for e in client.get(MAKES).json()["makes"] if e["models"])
    body = {**listing, "make": entry["make"], "model": entry["models"][0]}

    warnings = client.post("/v1/predict", json=body).json()["warnings"]

    assert "/model" not in [warning["pointer"] for warning in warnings]


@pytest.mark.req("FR-16")
def test_a_service_without_an_index_answers_503(bundle: Path, tmp_path: Path):
    settings = Settings(
        prediction_log=tmp_path / "log.sqlite3", comparables_file=tmp_path / INDEX_FILE
    )

    with TestClient(create_app(bundle, settings), raise_server_exceptions=False) as api:
        assert_problem(api.get(MAKES), 503)
