"""`GET /v1/makes`: the makes and the model names the web form suggests (FR-05)."""

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from tests.api.helpers import assert_problem

from recommenditos.api.app import create_app
from recommenditos.api.comparables import ComparablesIndex
from recommenditos.api.errors import pointer
from recommenditos.api.schemas.makes import Makes
from recommenditos.api.serving import ServedModel
from recommenditos.api.settings import Settings
from recommenditos.data.build_comparables import INDEX_FILE

MAKES = "/v1/makes"


@pytest.mark.req("FR-05")
def test_fr05_every_accepted_make_is_listed_with_the_models_it_knows(
    client: TestClient, served: ServedModel
):
    answer = client.get(MAKES).json()

    assert [entry["make"] for entry in answer["makes"]] == list(served.allowed["make"])
    models = {entry["make"]: entry["models"] for entry in answer["makes"]}
    # The synthetic fixture's BMW names, sorted regardless of case.
    assert models["BMW"] == ["1 Series", "3 Series", "5 Series", "i4", "X3", "X3 M", "X5"]
    assert answer["model"] == served.info.model_dump(mode="json")


@pytest.mark.req("FR-05")
def test_fr05_a_name_the_model_never_saw_is_not_suggested(
    served: ServedModel, index: ComparablesIndex
):
    # In the fixture every indexed name is also a model level, so one is added that is not.
    unseen = "Isetta"
    assert unseen not in served.model.input_schema.levels["model"]
    bmw = (*index.models_by_make["BMW"], unseen)
    extended = replace(index, models_by_make={**index.models_by_make, "BMW": bmw})

    models = {entry.make: entry.models for entry in Makes.of(served, extended).makes}

    assert unseen not in models["BMW"]
    assert models["BMW"] == list(index.models_by_make["BMW"])


@pytest.mark.req("FR-05")
def test_fr05_every_listed_model_reaches_the_model_without_a_warning(
    client: TestClient, listing: dict
):
    for entry in client.get(MAKES).json()["makes"]:
        for model in entry["models"]:
            body = {**listing, "make": entry["make"], "model": model}

            warnings = client.post("/v1/predict", json=body).json()["warnings"]

            assert pointer(["model"]) not in [warning["pointer"] for warning in warnings]


@pytest.mark.req("FR-05")
def test_fr05_a_model_off_the_list_is_answered_with_a_warning(client: TestClient, listing: dict):
    body = {**listing, "make": "BMW", "model": "3 series"}

    warnings = client.post("/v1/predict", json=body).json()["warnings"]

    assert pointer(["model"]) in [warning["pointer"] for warning in warnings]


@pytest.mark.req("FR-16")
def test_a_service_without_an_index_answers_503(bundle: Path, tmp_path: Path):
    settings = Settings(
        prediction_log=tmp_path / "log.sqlite3", comparables_file=tmp_path / INDEX_FILE
    )

    with TestClient(create_app(bundle, settings), raise_server_exceptions=False) as api:
        assert_problem(api.get(MAKES), 503)
