"""`POST /v1/predict`: the valuation (FR-06), its inputs (FR-01, FR-03 to FR-05)."""

import ast
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import shutil
import threading

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from tests.api.helpers import assert_problem, pointers

from recommenditos.api.app import create_app
from recommenditos.api.schemas.predict import Valuation, ValuationRequest
from recommenditos.api.serving import (
    COUNTRIES,
    PREDICTION_THREADS,
    REQUEST_CONTRACT,
    TOP_FEATURES,
    ServedModel,
    ServingError,
    explain,
)
from recommenditos.data.build_features import weight_kg
from recommenditos.modeling.model import BASE_LOG_PRICE, MODEL_FILE, RequestContract, load_model
from recommenditos.pipeline import read_frame
from recommenditos.schema import PROCESSED_SCHEMA

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
    assert set(body) == {"price_eur", "model", "request_id", "explanation", "warnings"}
    training = served.model.metadata["training"]
    assert training["price_min_eur"] <= body["price_eur"] <= training["price_max_eur"]
    assert body["model"] == served.info.model_dump()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["warnings"] == []


@pytest.mark.req("FR-06")
def test_fr06_every_request_gets_its_own_id(client: TestClient, listing: dict):
    first, second = (client.post(PREDICT, json=listing).json()["request_id"] for _ in range(2))

    assert first != second


def _request_body(row: pd.Series) -> dict:
    """A processed-split row as the `/v1/predict` body a seller would send for it."""
    body: dict = {}
    for name, field in ValuationRequest.model_fields.items():
        value = row.get(name)
        if name in _EQUIPMENT:
            body[name] = ast.literal_eval(value)
        elif value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        elif name == "registration_date":
            body[name] = value.strftime("%Y-%m")
        elif name == "weight_kg":
            body[name] = int(weight_kg(pd.Series([value])).iloc[0])
        elif field.annotation in (bool,) or isinstance(value, bool | np.bool_):
            body[name] = bool(value)
        elif (
            isinstance(value, float | np.floating) and float(value).is_integer() and name in _INTS
        ):
            body[name] = int(value)
        else:
            body[name] = value.item() if isinstance(value, np.generic) else value
    return body


_EQUIPMENT = {
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
}
_INTS = {"nr_prev_owners", "gears", "nr_seats", "nr_doors", "cylinders"}


@pytest.mark.req("FR-06", "FR-01")
@pytest.mark.parametrize("variant", ["lgbm-basic", "lgbm-extended"])
def test_fr06_the_api_prices_a_listing_as_the_pipeline_encodes_it(
    variant: str, trained: dict, matrices: dict
):
    """Rows of the synthetic test split, sent as requests, get the pipeline's own price.

    The oracle is the `features` stage's path: `space.matrix` on the processed
    row at the request date, then `predict_eur`. The API builds its row from
    the request instead, so a skew in that translation (a shifted age, a weight
    in the wrong unit, a lost optional field) changes a price here.
    """
    model = load_model(Path(trained["dir"]) / variant)
    rows = read_frame(Path(matrices["processed"]) / "test.parquet", PROCESSED_SCHEMA)
    today = datetime.now(UTC).date().isoformat()
    checked = 0
    with TestClient(create_app(Path(trained["dir"]) / variant)) as api:
        for _, row in rows.head(200).iterrows():
            body = _request_body(row)
            try:
                ValuationRequest(**body)
            except ValidationError:
                continue  # a row the API refuses, e.g. without a required field
            response = api.post(PREDICT, json=body)
            if response.status_code == 422:
                continue  # an unsupported make or an unseen strict level
            expected = model.predict_eur(
                model.space.matrix(row.to_frame().T.astype(rows.dtypes), reference_date=today)
            )
            assert response.json()["price_eur"] == round(float(expected.iloc[0]), 2), body
            checked += 1

    assert checked >= 50


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
    # The schema's types hold as written: no number for a boolean, no numeric
    # string for a number (found by the contract tests, #144).
    pytest.param("is_rental", 0, id="flag-a-number"),
    pytest.param("mileage_km_raw", "45000", id="mileage-a-numeric-string"),
    pytest.param("nr_doors", 4.5, id="integer-a-fraction"),
]


@pytest.mark.req("FR-03")
@pytest.mark.parametrize("month", ["1899-12", "0000-01", _next_month()])
def test_fr03_a_month_outside_the_range_is_told_the_range(
    month: str, client: TestClient, listing: dict
):
    problem = assert_problem(
        client.post(PREDICT, json={**listing, "registration_date": month}), 422
    )

    assert "between 1900-01 and the current month" in problem["errors"][0]["detail"]


@pytest.mark.req("FR-03")
@pytest.mark.parametrize(("field", "value"), _INVALID)
def test_fr03_refuses_an_invalid_value(field: str, value, client: TestClient, listing: dict):
    problem = assert_problem(client.post(PREDICT, json={**listing, field: value}), 422)

    assert pointers(problem) == [f"#/{field}"]


def _post(client: TestClient, body: dict):
    """POST `body` as Python's `json` writes it, `NaN` and `Infinity` included.

    The test client refuses to send them, but a client in another language may,
    and the server's JSON parser accepts them.
    """
    return client.post(
        PREDICT, content=json.dumps(body), headers={"content-type": "application/json"}
    )


_OUT_OF_BOUNDS = [
    pytest.param("mileage_km_raw", float("nan"), id="mileage-nan"),
    pytest.param("power_kw", float("inf"), id="power-infinity"),
    pytest.param("nr_prev_owners", -3, id="owners-negative"),
    pytest.param("nr_prev_owners", 21, id="owners-above-20"),
    pytest.param("gears", 13, id="gears-above-12"),
    pytest.param("nr_seats", 0, id="seats-0"),
    pytest.param("nr_doors", -5, id="doors-negative"),
    pytest.param("nr_doors", 10**29, id="doors-huge"),
    pytest.param("cylinders", 17, id="cylinders-above-16"),
    pytest.param("cylinders_volume_cc", -100, id="displacement-negative"),
    pytest.param("cylinders_volume_cc", float("nan"), id="displacement-nan"),
    pytest.param("weight_kg", -10, id="weight-negative"),
    pytest.param("weight_kg", 10_001, id="weight-above-10t"),
    pytest.param("electric_range_km", float("inf"), id="range-infinity"),
    pytest.param("electric_range_km", -1, id="range-negative"),
]


@pytest.mark.req("FR-03")
@pytest.mark.parametrize(("field", "value"), _OUT_OF_BOUNDS)
def test_fr03_refuses_a_number_a_car_cannot_have(field: str, value, trained: dict, listing: dict):
    """On `lgbm-extended`, which reads every optional number, weight included."""
    with TestClient(
        create_app(Path(trained["dir"]) / "lgbm-extended"), raise_server_exceptions=False
    ) as extended:
        problem = assert_problem(_post(extended, {**listing, field: value}), 422)

    assert pointers(problem) == [f"#/{field}"]


@pytest.mark.req("FR-03")
@pytest.mark.parametrize("field", ["mileage_km_raw", "electric_range_km"])
def test_fr03_a_nan_is_refused_as_not_finite(field: str, client: TestClient, listing: dict):
    problem = assert_problem(_post(client, {**listing, field: float("nan")}), 422)

    assert "finite" in problem["errors"][0]["detail"]


_BOUNDARIES = [
    pytest.param("registration_date", "1900-01", id="registration-1900-01"),
    pytest.param("registration_date", _this_month(), id="registration-this-month"),
    pytest.param("mileage_km_raw", 0, id="mileage-0"),
    pytest.param("mileage_km_raw", 1_000_000, id="mileage-1m"),
    pytest.param("power_kw", 1, id="power-1"),
    pytest.param("power_kw", 1_200, id="power-1200"),
    # JSON Schema counts a whole float as an integer.
    pytest.param("nr_doors", 4.0, id="integer-a-whole-float"),
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


# --------------------------------------------------------------------------
# FR-08: the explanation
# --------------------------------------------------------------------------


def _reproduced(explanation: dict) -> float:
    """The base price times `1 + effect` over all six reported effects (FR-08)."""
    effects = [entry["effect_pct"] for entry in explanation["top_features"]]
    effects.append(explanation["other_features_effect_pct"])
    return explanation["base_price_eur"] * math.prod(1 + effect / 100 for effect in effects)


def _assert_reproduces(answer: dict, price: float | None = None) -> None:
    """`answer`'s explanation gives `price`, its estimate unless another is named.

    The tolerance is the rounding of the estimate to the cent and nothing more:
    nothing in the explanation is rounded.
    """
    price = answer["price_eur"] if price is None else price
    tolerance = 0.005 + 1e-9 * price
    assert _reproduced(answer["explanation"]) == pytest.approx(price, rel=0, abs=tolerance), answer


def _field_contributions(served: ServedModel, listing: dict) -> dict[str, float]:
    """Each request field's SHAP value in log price, from the booster's own export."""
    matrix = served.matrix(ValuationRequest(**listing))
    row = served.model.log_price_contributions(matrix).iloc[0]
    return {name: float(row[list(columns)].sum()) for name, columns in served.fields.items()}


@pytest.mark.req("FR-08")
@pytest.mark.parametrize("variant", ["lgbm-basic", "lgbm-extended"])
def test_fr08_the_explanation_reproduces_the_estimate(variant: str, trained: dict, matrices: dict):
    """Rows of the synthetic test split, sent as requests: base x (1 + effect) is the price.

    `lgbm-extended` is the variant whose equipment lists become many columns, so
    it also shows that the explanation names request fields, never those columns.
    """
    rows = read_frame(Path(matrices["processed"]) / "test.parquet", PROCESSED_SCHEMA)
    checked = 0
    with TestClient(create_app(Path(trained["dir"]) / variant)) as api:
        for _, row in rows.head(100).iterrows():
            body = _request_body(row)
            try:
                ValuationRequest(**body)
            except ValidationError:
                continue
            response = api.post(PREDICT, json=body)
            if response.status_code == 422:
                continue
            answer = response.json()
            _assert_reproduces(answer)
            named = [entry["field"] for entry in answer["explanation"]["top_features"]]
            assert len(named) == TOP_FEATURES == len(set(named))
            assert set(named) <= set(ValuationRequest.model_fields)
            checked += 1

    assert checked >= 30


def _valid_requests(served: ServedModel) -> st.SearchStrategy[dict]:
    """Requests `/v1/predict` accepts, from the values the served bundle knows."""
    levels = served.model.input_schema.levels
    this_year = datetime.now(UTC).year
    required = {
        "make": st.sampled_from(served.allowed["make"]),
        "model": st.sampled_from(levels["model"]),
        "registration_date": st.builds(
            "{:04d}-{:02d}".format,
            st.integers(1990, this_year - 1),
            st.integers(1, 12),
        ),
        "mileage_km_raw": st.floats(0, 1_000_000),
        "power_kw": st.floats(1, 1_200),
        "fuel_category": st.sampled_from(served.allowed["fuel_category"]),
        "transmission": st.sampled_from(served.allowed["transmission"]),
        "country_code": st.sampled_from(served.allowed["country_code"]),
        "body_type": st.sampled_from(served.allowed["body_type"]),
        "seller_type": st.sampled_from(served.allowed["seller_type"]),
    }
    optional = {
        "nr_prev_owners": st.integers(0, 20),
        "drive_train": st.sampled_from(served.allowed["drive_train"]),
        "gears": st.integers(0, 12),
        "cylinders_volume_cc": st.floats(0, 10_000),
        "nr_seats": st.integers(1, 9),
        "nr_doors": st.integers(1, 7),
    }
    return st.fixed_dictionaries(required, optional=optional)


@pytest.mark.req("FR-08")
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_fr08_any_valid_request_is_explained_to_its_estimate(
    data: st.DataObject, client: TestClient, served: ServedModel
):
    """The reproduction invariant holds for every valid request, not just the data's rows."""
    body = data.draw(_valid_requests(served))

    response = client.post(PREDICT, json=body)

    assert response.status_code == 200, response.text
    _assert_reproduces(response.json())


@pytest.mark.req("FR-08")
def test_fr08_lists_the_five_fields_with_the_largest_effect_largest_first(
    client: TestClient, listing: dict, served: ServedModel
):
    """Ranked by the size of each field's SHAP value, the rest combined, each as exp(φ) - 1."""
    expected = _field_contributions(served, listing)

    explanation = client.post(PREDICT, json=listing).json()["explanation"]

    ranked = sorted(expected, key=lambda name: abs(expected[name]), reverse=True)
    top, others = ranked[:TOP_FEATURES], ranked[TOP_FEATURES:]
    assert [entry["field"] for entry in explanation["top_features"]] == top
    for entry in explanation["top_features"]:
        assert entry["effect_pct"] == pytest.approx(np.expm1(expected[entry["field"]]) * 100)
    combined = sum(expected[name] for name in others)
    assert explanation["other_features_effect_pct"] == pytest.approx(np.expm1(combined) * 100)
    base = served.model.log_price_contributions(served.matrix(ValuationRequest(**listing)))
    assert explanation["base_price_eur"] == pytest.approx(np.exp(base[BASE_LOG_PRICE].iloc[0]))


@pytest.mark.req("FR-08")
def test_fr08_equal_effects_keep_the_request_order_and_fewer_fields_are_all_listed():
    """Ties do not depend on dict order or on the sort, and a small model lists every field."""
    fields = {"make": ("make",), "model": ("model",), "registration_date": ("age_years",)}
    tied = pd.Series({"make": 0.1, "model": -0.1, "age_years": 0.1, BASE_LOG_PRICE: 9.0})

    explanation = explain(tied, fields, top=2)
    small = explain(tied, fields)

    assert [entry.field for entry in explanation.top_features] == ["make", "model"]
    assert explanation.other_features_effect_pct == pytest.approx(np.expm1(0.1) * 100)
    assert [entry.field for entry in small.top_features] == list(fields)
    assert small.other_features_effect_pct == 0.0


@pytest.mark.req("FR-08")
def test_fr08_an_equipment_list_is_explained_as_one_field(trained: dict, listing: dict):
    """The multi-hot columns of one equipment list add up to one field's effect."""
    with TestClient(create_app(Path(trained["dir"]) / "lgbm-extended")) as extended:
        served = extended.app.state.served  # type: ignore[attr-defined]
        listing = {
            **listing,
            "equipment_comfort": list(
                served.model.space.vocabulary.equipment["equipment_comfort"]
            ),
        }
        expected = _field_contributions(served, listing)
        answer = extended.post(PREDICT, json=listing).json()

    assert len(served.fields["equipment_comfort"]) > 1
    _assert_reproduces(answer)
    assert set(expected) <= set(ValuationRequest.model_fields)
    for entry in answer["explanation"]["top_features"]:
        assert entry["effect_pct"] == pytest.approx(np.expm1(expected[entry["field"]]) * 100)


@pytest.mark.req("FR-08")
def test_fr08_a_bounded_estimate_is_reported_and_the_explanation_gives_the_unbounded_one(
    bundle: Path, tmp_path: Path, listing: dict, client: TestClient
):
    """A price clipped to the training range is said so; the effects still multiply up.

    The real data reaches the bound rarely (1 of 103,868 rows for `lgbm-basic`, a
    test listing at the lower bound), so the bundle's range is narrowed until
    this listing's estimate lies outside it.
    """
    unbounded = client.post(PREDICT, json=listing).json()["price_eur"]
    edited = tmp_path / "bundle"
    shutil.copytree(bundle, edited)
    record = json.loads((edited / MODEL_FILE).read_text(encoding="utf-8"))
    bound = round(unbounded / 2)
    record["training"]["price_max_eur"] = bound
    record["training"]["price_min_eur"] = min(record["training"]["price_min_eur"], bound / 2)
    (edited / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")

    with TestClient(create_app(edited)) as narrower:
        answer = narrower.post(PREDICT, json=listing).json()

    assert answer["price_eur"] == pytest.approx(bound, abs=0.01)
    assert [warning["pointer"] for warning in answer["warnings"]] == ["#"]
    _assert_reproduces(answer, price=unbounded)


@pytest.mark.req("FR-08")
@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_fr08_a_bundle_without_a_shap_export_is_refused_at_startup(variant: str, trained: dict):
    with pytest.raises(ServingError, match="SHAP export"):
        create_app(Path(trained["dir"]) / variant)


@pytest.mark.req("FR-08")
def test_fr08_the_documented_example_reproduces_its_estimate():
    example = Valuation.model_config["json_schema_extra"]["examples"][0]  # type: ignore[index]

    _assert_reproduces(example)


@pytest.mark.req("NFR-02", "NFR-03")
def test_nfr02_every_booster_call_of_a_valuation_runs_on_one_thread(
    served: ServedModel, listing: dict, monkeypatch: pytest.MonkeyPatch
):
    """The API asks LightGBM for `PREDICTION_THREADS` threads in each prediction.

    LightGBM's default is one OpenMP thread per core, and `Booster.predict` does
    not read the `num_threads` of `booster.txt`. With the default, a one-row
    valuation cost 4.6 times its wall time in CPU, and concurrent requests and
    workers competed for the cores (issue #179).
    """
    booster = served.model.booster
    assert booster is not None
    calls: list[dict] = []
    predict = booster.predict

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return predict(*args, **kwargs)

    monkeypatch.setattr(booster, "predict", spy)
    served.value(ValuationRequest(**listing))
    assert PREDICTION_THREADS == 1
    assert len(calls) == 2  # the price and its explanation
    assert all(call.get("num_threads") == PREDICTION_THREADS for call in calls)


@pytest.mark.req("NFR-02", "NFR-03")
def test_nfr02_a_process_scores_one_request_at_a_time(served: ServedModel, listing: dict):
    """A valuation waits while another holds the model, instead of taking turns at the GIL."""
    request = ValuationRequest(**listing)
    finished = threading.Event()
    worker = threading.Thread(target=lambda: (served.value(request), finished.set()))
    with served._scoring:
        worker.start()
        assert not finished.wait(timeout=0.5)
    worker.join(timeout=30)
    assert finished.is_set()
