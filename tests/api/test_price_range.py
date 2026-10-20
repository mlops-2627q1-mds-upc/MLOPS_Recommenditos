"""`POST /v1/price-range`: the price range of a car described in part (FR-02, FR-07).

The request validation is `/v1/predict`'s for every field that is given, so the
cases here check that the same rules apply with only `make` required, not every
rule again; `tests/api/test_predict.py` holds those.
"""

from pathlib import Path

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np
import pandas as pd
import pytest
from tests.api.helpers import assert_problem, pointers
from tests.api.test_predict import _INVALID

from recommenditos.api import create_app
from recommenditos.api.schemas.partial_listing import PartialListing
from recommenditos.api.schemas.predict import ValuationRequest
from recommenditos.api.serving import COUNTRIES, REQUEST_CONTRACT, ServedModel, price_range
from recommenditos.modeling.model import (
    INTERVAL_COLUMNS,
    RequestContract,
    price_range_seam_fields,
    seam_names,
)

PRICE_RANGE = "/v1/price-range"
PREDICT = "/v1/predict"


def _assert_range(body: dict) -> None:
    assert body["lower_eur"] <= body["price_eur"] <= body["upper_eur"], body


@pytest.fixture
def p1(listing: dict, params: dict) -> dict:
    """Scenario P1 of SC-05 (problem specification section 8), from the listing fixture."""
    return {name: listing[name] for name in params["evaluate"]["sc05_p1_fields"]}


# --------------------------------------------------------------------------
# FR-02: the inputs
# --------------------------------------------------------------------------


@pytest.mark.req("FR-02", "FR-07")
def test_fr02_answers_a_request_that_names_only_the_make(
    client: TestClient, listing: dict, served: ServedModel
):
    response = client.post(PRICE_RANGE, json={"make": listing["make"]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "price_eur",
        "lower_eur",
        "upper_eur",
        "confidence_level",
        "model",
        "request_id",
        "warnings",
    }
    _assert_range(body)
    assert body["confidence_level"] == served.model.metadata["intervals"]["confidence_level"]
    assert body["model"] == served.info.model_dump()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["warnings"] == []


@pytest.mark.req("FR-02", "FR-07")
def test_fr02_answers_scenario_p1(client: TestClient, p1: dict):
    response = client.post(PRICE_RANGE, json=p1)

    assert response.status_code == 200, response.text
    _assert_range(response.json())


@pytest.mark.req("FR-02")
def test_fr02_only_the_make_is_required(client: TestClient, params: dict):
    problem = assert_problem(client.post(PRICE_RANGE, json={"model": "Golf"}), 422)

    assert pointers(problem) == ["#/make"]
    assert REQUEST_CONTRACT.price_range_required_fields == ("make",)
    assert REQUEST_CONTRACT.price_range_required_fields == (
        RequestContract.from_params(params).price_range_required_fields
    )


@pytest.mark.req("FR-02")
def test_fr02_the_request_has_the_fields_of_predict():
    """Field for field, in the same order and of the same type, only `make` required.

    So a field added to `/v1/predict` cannot be forgotten here, and a field that
    is given is checked by the same rules (FR-03).
    """
    partial = PartialListing.model_json_schema()["properties"]
    full = ValuationRequest.model_json_schema()
    optional = set(full["properties"]) - set(full["required"])

    assert list(partial) == list(full["properties"])
    for name, schema in full["properties"].items():
        if name == "make" or name in optional:
            assert partial[name] == schema, name
        else:
            branch = {key: value for key, value in schema.items() if key != "title"}
            assert partial[name]["anyOf"] == [branch, {"type": "null"}], name


@pytest.mark.req("FR-02", "FR-03")
@pytest.mark.parametrize(("field", "value"), _INVALID)
def test_fr02_a_given_field_is_checked_as_predict_checks_it(
    field: str, value, client: TestClient, listing: dict
):
    body = {"make": listing["make"], field: value}

    problem = assert_problem(client.post(PRICE_RANGE, json=body), 422)

    assert pointers(problem) == [f"#/{field}"]


@pytest.mark.req("FR-02", "FR-03", "FR-04")
def test_fr02_every_fault_of_a_request_is_in_one_response(client: TestClient):
    body = {"make": "Trabant", "power_kw": 5000, "colour": "red", "country_code": "US"}

    problem = assert_problem(client.post(PRICE_RANGE, json=body), 422)

    assert pointers(problem) == ["#/make", "#/power_kw", "#/country_code", "#/colour"]
    assert problem["errors"][2]["allowed"] == list(COUNTRIES)


@pytest.mark.req("FR-02", "FR-03")
def test_fr02_a_field_sent_as_null_is_a_field_left_out(client: TestClient, listing: dict):
    make = {"make": listing["make"]}

    left_out = client.post(PRICE_RANGE, json=make).json()
    null = client.post(PRICE_RANGE, json={**make, "model": None, "power_kw": None}).json()

    assert {key: null[key] for key in ("price_eur", "lower_eur", "upper_eur")} == {
        key: left_out[key] for key in ("price_eur", "lower_eur", "upper_eur")
    }


# --------------------------------------------------------------------------
# FR-07: the typical price and the bounds
# --------------------------------------------------------------------------


@pytest.mark.req("FR-07")
def test_fr07_a_request_with_every_required_field_gets_the_price_of_predict(
    client: TestClient, listing: dict
):
    """Option C of EDN-85, first branch: the point model's price, as `/v1/predict` gives it."""
    full = {**listing, "nr_prev_owners": 1}

    price_range_answer = client.post(PRICE_RANGE, json=full).json()

    assert price_range_answer["price_eur"] == client.post(PREDICT, json=full).json()["price_eur"]
    _assert_range(price_range_answer)


@pytest.mark.req("FR-07")
@pytest.mark.parametrize("left_out", ["power_kw", "registration_date"])
def test_fr07_a_request_missing_a_required_field_gets_the_median(
    left_out: str, client: TestClient, listing: dict, served: ServedModel
):
    """Option C of EDN-85, second branch: the 50 % quantile model's price."""
    del listing[left_out]
    matrix = served.matrix(PartialListing(**listing))
    median = float(served.model.predict_interval_eur(matrix)[INTERVAL_COLUMNS["median"]].iloc[0])

    answer = client.post(PRICE_RANGE, json=listing).json()

    assert answer["price_eur"] == round(median, 2)
    _assert_range(answer)


class _Stub:
    """A model whose point estimate and interval are given, to place one outside the other."""

    def __init__(self, point: float, lower: float, median: float, upper: float) -> None:
        self.point = point
        self.interval = {"lower": lower, "median": median, "upper": upper}

    def predict_eur(self, matrix: pd.DataFrame) -> pd.Series:
        return pd.Series(self.point, index=matrix.index)

    def predict_interval_eur(self, matrix: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {INTERVAL_COLUMNS[name]: value for name, value in self.interval.items()},
            index=matrix.index,
        )


@pytest.mark.req("FR-07")
@pytest.mark.parametrize(
    ("point", "expected"),
    [
        pytest.param(25_000.0, (10_000.0, 25_000.0, 20_000.0), id="point-above-the-upper-bound"),
        pytest.param(8_000.0, (8_000.0, 8_000.0, 20_000.0), id="point-below-the-lower-bound"),
        pytest.param(15_000.0, (10_000.0, 15_000.0, 20_000.0), id="point-inside"),
    ],
)
def test_fr07_the_bounds_are_widened_to_contain_the_typical_price(
    point: float, expected: tuple[float, float, float]
):
    stub = _Stub(point=point, lower=10_000.0, median=14_000.0, upper=20_000.0)

    row = price_range(stub, pd.DataFrame(index=[0]), complete=True).iloc[0]  # type: ignore[arg-type]

    assert (row["lower_eur"], row["price_eur"], row["upper_eur"]) == expected


@pytest.mark.req("FR-07")
def test_fr07_the_median_is_never_widened_past_the_interval_it_lies_in():
    stub = _Stub(point=99_000.0, lower=10_000.0, median=14_000.0, upper=20_000.0)

    row = price_range(stub, pd.DataFrame(index=[0]), complete=False).iloc[0]  # type: ignore[arg-type]

    assert (row["lower_eur"], row["price_eur"], row["upper_eur"]) == (10_000.0, 14_000.0, 20_000.0)


@pytest.mark.req("FR-07", "FR-02")
def test_fr07_adding_fields_narrows_the_range(
    trained: dict, test_frames: dict, params: dict, served: ServedModel
):
    """The mean relative width falls from make-only to P1 to full, on the fixture's test split.

    `reports/analysis/price_range_widths.py` measures the same on the real one.
    """
    model = served.model
    frame = test_frames[model.feature_set]
    frame = frame[frame["make"].isin(served.allowed["make"])]
    contract = RequestContract.from_params(params)
    seam = seam_names(served.fields, contract)
    p1_fields = set(params["evaluate"]["sc05_p1_fields"])
    scenarios = {
        "make only": (price_range_seam_fields(model, contract), False),
        "P1": (tuple(seam[name] for name in served.fields if name not in p1_fields), False),
        "full": ((), True),
    }

    widths = {}
    for name, (left_out, complete) in scenarios.items():
        masked = model.mask_absent(frame, left_out) if left_out else frame
        answer = price_range(model, masked, complete=complete)
        relative = (answer["upper_eur"] - answer["lower_eur"]) / answer["price_eur"]
        widths[name] = float(np.mean(relative))

    assert widths["make only"] > widths["P1"] > widths["full"], widths


def _partial_requests(served: ServedModel) -> st.SearchStrategy[dict]:
    """Valid `/v1/price-range` bodies: a supported make and any subset of the other fields."""
    levels = served.model.input_schema.levels
    allowed = served.allowed
    months = st.builds(
        lambda year, month: f"{year:04d}-{month:02d}",
        st.integers(1990, 2024),
        st.integers(1, 12),
    )
    optional = {
        "model": st.sampled_from([*levels["model"], "Concept One"]),
        "registration_date": months,
        "mileage_km_raw": st.floats(0, 1_000_000),
        "power_kw": st.floats(1, 1_200),
        "fuel_category": st.sampled_from(allowed["fuel_category"]),
        "transmission": st.sampled_from(allowed["transmission"]),
        "country_code": st.sampled_from(COUNTRIES),
        "body_type": st.sampled_from(allowed["body_type"]),
        "seller_type": st.sampled_from(allowed["seller_type"]),
        "nr_prev_owners": st.integers(0, 20),
        "drive_train": st.sampled_from(allowed["drive_train"]),
        "gears": st.integers(0, 12),
        "nr_seats": st.integers(1, 9),
        "nr_doors": st.integers(1, 7),
        "cylinders_volume_cc": st.floats(0, 10_000),
    }
    return st.fixed_dictionaries({"make": st.sampled_from(allowed["make"])}, optional=optional)


@pytest.mark.req("FR-07", "FR-02")
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_fr07_every_valid_partial_request_gets_lower_le_typical_le_upper(
    data: st.DataObject, client: TestClient, served: ServedModel
):
    body = data.draw(_partial_requests(served))

    response = client.post(PRICE_RANGE, json=body)

    assert response.status_code == 200, response.text
    _assert_range(response.json())


@pytest.mark.req("FR-07")
def test_fr07_a_bundle_without_interval_models_answers_503(trained: dict, listing: dict):
    """`lgbm-extended` serves `/v1/predict` and carries no price range (EDN-85)."""
    with TestClient(
        create_app(Path(trained["dir"]) / "lgbm-extended"), raise_server_exceptions=False
    ) as extended:
        assert not extended.app.state.served.has_price_range  # type: ignore[attr-defined]

        problem = assert_problem(extended.post(PRICE_RANGE, json={"make": listing["make"]}), 503)

    assert "lgbm-extended" in problem["detail"]


# --------------------------------------------------------------------------
# FR-05: unseen values
# --------------------------------------------------------------------------


@pytest.mark.req("FR-05", "FR-02")
def test_fr05_unseen_values_are_answered_with_warnings(
    client: TestClient, listing: dict, served: ServedModel
):
    assert "ES" not in served.model.input_schema.levels["country_code"]
    body = {"make": listing["make"], "model": "Concept One", "country_code": "ES"}

    response = client.post(PRICE_RANGE, json=body)

    assert response.status_code == 200, response.text
    warnings = response.json()["warnings"]
    assert [warning["pointer"] for warning in warnings] == ["#/model", "#/country_code"]
    assert "'Concept One'" in warnings[0]["detail"]
    _assert_range(response.json())


@pytest.mark.req("FR-05", "FR-02")
def test_fr05_an_unseen_model_is_priced_as_a_model_left_out(client: TestClient, listing: dict):
    make = {"make": listing["make"]}

    unseen = client.post(PRICE_RANGE, json={**make, "model": "Concept One"}).json()
    left_out = client.post(PRICE_RANGE, json=make).json()

    assert (unseen["lower_eur"], unseen["price_eur"], unseen["upper_eur"]) == (
        left_out["lower_eur"],
        left_out["price_eur"],
        left_out["upper_eur"],
    )
