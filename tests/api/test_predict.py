"""`POST /v1/predict`: the valuation (FR-06), its inputs (FR-01, FR-03 to FR-05)."""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import pytest
from tests.api.helpers import assert_problem, pointers

from recommenditos.api import create_app
from recommenditos.api.schemas import ValuationRequest
from recommenditos.api.serving import COUNTRIES, REQUEST_CONTRACT, ServedModel
from recommenditos.modeling.model import MODEL_FILE, RequestContract

PREDICT = "/v1/predict"


def _this_month() -> str:
    now = datetime.now(UTC)
    return f"{now.year:04d}-{now.month:02d}"


def _next_month() -> str:
    now = datetime.now(UTC)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return f"{year:04d}-{month:02d}"


# --------------------------------------------------------------------------
# FR-06: the answer
# --------------------------------------------------------------------------


@pytest.mark.req("FR-06")
def test_fr06_answers_a_price_in_eur_the_model_version_and_a_request_id(
    client: TestClient, listing: dict, served: ServedModel
):
    response = client.post(PREDICT, json=listing)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"price_eur", "model", "request_id", "warnings"}
    training = served.model.metadata["training"]
    assert training["price_min_eur"] <= body["price_eur"] <= training["price_max_eur"]
    assert body["model"] == served.info.model_dump()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["warnings"] == []


@pytest.mark.req("FR-06")
def test_fr06_every_request_gets_its_own_id(client: TestClient, listing: dict):
    first, second = (client.post(PREDICT, json=listing).json()["request_id"] for _ in range(2))

    assert first != second


@pytest.mark.req("FR-06")
def test_fr06_the_price_is_the_models_own_for_the_same_listing(
    client: TestClient, listing: dict, served: ServedModel
):
    """The route adds nothing to the model's number but the rounding to cents."""
    price, _ = served.value(ValuationRequest(**listing))

    assert client.post(PREDICT, json=listing).json()["price_eur"] == round(price, 2)


# --------------------------------------------------------------------------
# FR-01: required and optional fields
# --------------------------------------------------------------------------


@pytest.mark.req("FR-01")
@pytest.mark.parametrize("field", REQUEST_CONTRACT.required_fields)
def test_fr01_rejects_a_request_missing(field: str, client: TestClient, listing: dict):
    """One case per required field, e.g. `[power_kw]`: the 422 names exactly that field."""
    del listing[field]

    problem = assert_problem(client.post(PREDICT, json=listing), 422)

    assert pointers(problem) == [f"#/{field}"]
    assert problem["errors"][0]["detail"] == "This field is required."


@pytest.mark.req("FR-01")
def test_fr01_the_required_fields_are_the_pipelines(params: dict):
    """The API's contract is the one `params.yaml` gives SC-06 and the blanked copies."""
    pipeline = RequestContract.from_params(params)

    assert set(REQUEST_CONTRACT.required_fields) == set(pipeline.required_fields)
    assert dict(REQUEST_CONTRACT.field_to_feature) == dict(pipeline.field_to_feature)


@pytest.mark.req("FR-01")
def test_fr01_accepts_a_request_with_only_the_required_fields(client: TestClient, listing: dict):
    assert client.post(PREDICT, json=listing).status_code == 200


@pytest.mark.req("FR-01")
def test_fr01_an_equipment_list_is_never_null(client: TestClient, listing: dict):
    problem = assert_problem(client.post(PREDICT, json={**listing, "equipment_safety": None}), 422)

    assert pointers(problem) == ["#/equipment_safety"]


@pytest.mark.req("FR-01")
def test_fr01_an_omitted_equipment_list_is_valued_as_an_empty_one(trained: dict, listing: dict):
    """On `lgbm-extended`, which reads the equipment lists: omitted equals `[]`.

    A null list would reach the model as a state no training row has (EDN-61);
    the request model defaults it to `[]`, and `mask_absent` gives the same.
    """
    with TestClient(create_app(Path(trained["dir"]) / "lgbm-extended")) as extended:
        omitted = extended.post(PREDICT, json=listing).json()["price_eur"]
        empty = extended.post(
            PREDICT,
            json={
                **listing,
                "equipment_comfort": [],
                "equipment_entertainment": [],
                "equipment_extra": [],
                "equipment_safety": [],
            },
        ).json()["price_eur"]

    assert omitted == empty


@pytest.mark.req("FR-01")
def test_fr01_an_optional_field_moves_the_price(client: TestClient, listing: dict):
    """The optional fields reach the model: an extreme one changes the estimate."""
    base = client.post(PREDICT, json=listing).json()["price_eur"]
    with_owners = client.post(PREDICT, json={**listing, "nr_prev_owners": 9}).json()["price_eur"]
    with_doors = client.post(PREDICT, json={**listing, "nr_doors": 2}).json()["price_eur"]

    assert {with_owners, with_doors} != {base}


@pytest.mark.req("FR-01", "FR-03")
def test_fr01_refuses_a_field_that_is_not_part_of_the_request(client: TestClient, listing: dict):
    """A misspelt optional field is an error, not an input the model never sees."""
    problem = assert_problem(client.post(PREDICT, json={**listing, "power_ps": 150}), 422)

    assert pointers(problem) == ["#/power_ps"]


# --------------------------------------------------------------------------
# FR-03: validation
# --------------------------------------------------------------------------

_INVALID = [
    pytest.param("registration_date", "1899-12", id="registration-before-1900"),
    pytest.param("registration_date", _next_month(), id="registration-after-this-month"),
    pytest.param("registration_date", "2019-13", id="registration-month-13"),
    pytest.param("registration_date", "06/2019", id="registration-not-yyyy-mm"),
    pytest.param("mileage_km_raw", -1, id="mileage-below-0"),
    pytest.param("mileage_km_raw", 1_000_001, id="mileage-above-1m"),
    pytest.param("mileage_km_raw", "many", id="mileage-not-a-number"),
    pytest.param("power_kw", 0, id="power-below-1"),
    pytest.param("power_kw", 1_201, id="power-above-1200"),
    pytest.param("power_kw", "fast", id="power-not-a-number"),
    pytest.param("fuel_category", "Steam", id="fuel-unknown"),
    pytest.param("transmission", "Sequential", id="transmission-unknown"),
    pytest.param("body_type", "Spaceship", id="body-type-unknown"),
    pytest.param("drive_train", "6WD", id="drive-train-unknown"),
    pytest.param("seller_type", "Auction", id="seller-type-unknown"),
    pytest.param("country_code", "US", id="country-outside-the-8"),
    pytest.param("make", 42, id="make-not-a-string"),
    pytest.param("nr_doors", "four", id="optional-not-a-number"),
]


@pytest.mark.req("FR-03")
@pytest.mark.parametrize(("field", "value"), _INVALID)
def test_fr03_refuses_an_invalid_value(field: str, value, client: TestClient, listing: dict):
    problem = assert_problem(client.post(PREDICT, json={**listing, field: value}), 422)

    assert pointers(problem) == [f"#/{field}"]


_BOUNDARIES = [
    pytest.param("registration_date", "1900-01", id="registration-1900-01"),
    pytest.param("registration_date", _this_month(), id="registration-this-month"),
    pytest.param("mileage_km_raw", 0, id="mileage-0"),
    pytest.param("mileage_km_raw", 1_000_000, id="mileage-1m"),
    pytest.param("power_kw", 1, id="power-1"),
    pytest.param("power_kw", 1_200, id="power-1200"),
]


@pytest.mark.req("FR-03")
@pytest.mark.parametrize(("field", "value"), _BOUNDARIES)
def test_fr03_accepts_the_edges_of_a_range(field: str, value, client: TestClient, listing: dict):
    assert client.post(PREDICT, json={**listing, field: value}).status_code == 200


@pytest.mark.req("FR-03")
def test_fr03_a_country_outside_the_8_lists_the_8(client: TestClient, listing: dict):
    problem = assert_problem(client.post(PREDICT, json={**listing, "country_code": "US"}), 422)

    assert problem["errors"][0]["allowed"] == list(COUNTRIES)


@pytest.mark.req("FR-03")
def test_fr03_a_categorical_lists_the_levels_of_the_served_model(
    client: TestClient, listing: dict, served: ServedModel
):
    problem = assert_problem(client.post(PREDICT, json={**listing, "fuel_category": "Steam"}), 422)

    assert problem["errors"][0]["allowed"] == list(
        served.model.input_schema.levels["fuel_category"]
    )


@pytest.mark.req("FR-03", "FR-04")
def test_fr03_every_fault_of_a_request_is_in_one_response(client: TestClient, listing: dict):
    """A type fault, a range fault, a missing field and the model's own checks, together."""
    del listing["model"]
    faulty = {
        **listing,
        "make": "Trabant",
        "power_kw": 5000,
        "mileage_km_raw": "many",
        "country_code": "US",
        "fuel_category": "Steam",
    }

    problem = assert_problem(client.post(PREDICT, json=faulty), 422)

    assert pointers(problem) == [
        "#/make",
        "#/model",
        "#/mileage_km_raw",
        "#/power_kw",
        "#/fuel_category",
        "#/country_code",
    ]
    assert problem["detail"] == "The request has 6 invalid fields."


@pytest.mark.req("FR-03")
def test_fr03_a_list_with_several_bad_items_is_one_error(client: TestClient, listing: dict):
    problem = assert_problem(
        client.post(PREDICT, json={**listing, "equipment_comfort": [1, "ABS", 2]}), 422
    )

    assert pointers(problem) == ["#/equipment_comfort/0"]


# --------------------------------------------------------------------------
# FR-04: the scope check
# --------------------------------------------------------------------------


@pytest.mark.req("FR-04")
def test_fr04_refuses_an_unsupported_make_and_lists_the_supported_ones(
    client: TestClient, listing: dict, served: ServedModel
):
    problem = assert_problem(client.post(PREDICT, json={**listing, "make": "Trabant"}), 422)

    assert pointers(problem) == ["#/make"]
    supported = served.model.metadata["training"]["supported_makes"]
    assert sorted(problem["errors"][0]["allowed"]) == sorted(supported)


@pytest.mark.req("FR-04", "FR-12")
def test_fr04_the_supported_makes_come_from_the_bundle_not_the_schema(
    bundle: Path, tmp_path: Path, listing: dict
):
    """A bundle that supports one make fewer refuses that make, with no change to the API."""
    edited = tmp_path / "bundle"
    shutil.copytree(bundle, edited)
    record = json.loads((edited / MODEL_FILE).read_text(encoding="utf-8"))
    dropped = listing["make"]
    record["training"]["supported_makes"].remove(dropped)
    (edited / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")

    with TestClient(create_app(edited)) as narrower:
        problem = assert_problem(narrower.post(PREDICT, json=listing), 422)
        another = {**listing, "make": record["training"]["supported_makes"][0]}
        assert narrower.post(PREDICT, json=another).status_code == 200

    assert pointers(problem) == ["#/make"]
    assert dropped not in problem["errors"][0]["allowed"]


# --------------------------------------------------------------------------
# FR-05: unseen values
# --------------------------------------------------------------------------


@pytest.mark.req("FR-05")
def test_fr05_a_request_from_es_is_answered_with_a_warning(
    client: TestClient, listing: dict, served: ServedModel
):
    assert "ES" not in served.model.input_schema.levels["country_code"]

    response = client.post(PREDICT, json={**listing, "country_code": "ES"})

    assert response.status_code == 200, response.text
    assert [warning["pointer"] for warning in response.json()["warnings"]] == ["#/country_code"]


@pytest.mark.req("FR-05")
def test_fr05_an_unseen_model_is_answered_with_a_warning(client: TestClient, listing: dict):
    response = client.post(PREDICT, json={**listing, "model": "Concept One"})

    assert response.status_code == 200, response.text
    warnings = response.json()["warnings"]
    assert [warning["pointer"] for warning in warnings] == ["#/model"]
    assert "'Concept One'" in warnings[0]["detail"]


@pytest.mark.req("FR-05")
def test_fr05_an_unseen_model_reaches_the_model_as_unknown(
    client: TestClient, listing: dict, served: ServedModel
):
    """The price for an unseen `model` is the price with that column missing."""
    known = served.matrix(ValuationRequest(**listing))
    unknown = served.model.mask_absent(known, ["model"])
    expected = float(served.model.predict_eur(unknown).iloc[0])

    response = client.post(PREDICT, json={**listing, "model": "Concept One"})

    assert response.json()["price_eur"] == round(expected, 2)
