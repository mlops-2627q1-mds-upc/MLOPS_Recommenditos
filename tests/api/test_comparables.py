"""`POST /v1/comparables`: listings like a car described in part (UC2, FR-09).

The app serves the index the `comparables` stage built from the synthetic
fixture's train and validation frames. Each answer is held against the rule
written out a second time here with pandas over that index, so a test checks
the filter, the order and `k` together; `tests/test_comparables_index.py` tests each
window at its bound on a hand-built index.
"""

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest
from tests.api.helpers import assert_problem, pointers

from recommenditos.api.app import create_app
from recommenditos.api.comparables import ComparablesIndex
from recommenditos.api.schemas.comparables import ComparablesRequest
from recommenditos.api.schemas.partial_listing import PartialListing
from recommenditos.api.settings import Settings
from recommenditos.config import PROCESSED_DATA_DIR
from recommenditos.data.build_comparables import INDEX_FILE

COMPARABLES = "/v1/comparables"


def expected(index: ComparablesIndex, body: dict) -> pd.DataFrame:
    """FR-09's answer to `body`, written out as a filter and a sort over the index."""
    rows = index.listings.assign(position=range(len(index)))
    rows = rows[rows["make"] == body["make"]]
    sort = ["position"]
    if "model" in body:
        rows = rows[rows["model"] == body["model"]]
    if "registration_date" in body:
        registered = pd.Timestamp(f"{body['registration_date']}-01")
        age = (pd.Timestamp(datetime.now(UTC).date()) - registered).days / 365.25
        rows = rows.assign(age_distance=(rows["age_years"] - age).abs())
        rows = rows[rows["age_distance"] <= 2]
        sort = ["age_distance", *sort]
    if "mileage_km_raw" in body:
        mileage = body["mileage_km_raw"]
        rows = rows.assign(mileage_distance=(rows["mileage_km_raw"] - mileage).abs())
        rows = rows[rows["mileage_distance"] <= 0.25 * mileage]
        sort = ["mileage_distance", *sort]
    return rows.sort_values(sort).head(body.get("k", 5))


def age(registration_date: str, on: str) -> float:
    """The age in years on `on` of a car first registered in `registration_date`."""
    return (pd.Timestamp(on) - pd.Timestamp(f"{registration_date}-01")).days / 365.25


def answered(response) -> list[tuple]:
    assert response.status_code == 200, response.text
    return [
        (each["model"], each["registration_date"], each["mileage_km_raw"], each["price_eur"])
        for each in response.json()["listings"]
    ]


def rows_of(frame: pd.DataFrame) -> list[tuple]:
    return [
        (row.model, row.registration_date, row.mileage_km_raw, row.price)
        for row in frame.itertuples()
    ]


@pytest.fixture(scope="module")
def car(index: ComparablesIndex) -> dict:
    """A full description of the index's most listed make and model, from a listing of it."""
    listings = index.listings.dropna(subset=["registration_date", "mileage_km_raw", "model"])
    make, model = listings.groupby(["make", "model"]).size().idxmax()
    listing = listings[(listings["make"] == make) & (listings["model"] == model)].iloc[0]
    return {
        "make": make,
        "model": model,
        "registration_date": listing["registration_date"],
        "mileage_km_raw": float(listing["mileage_km_raw"]),
    }


@pytest.mark.req("FR-09")
@pytest.mark.parametrize(
    "fields",
    [
        ("make", "model", "registration_date", "mileage_km_raw"),
        ("make", "model", "mileage_km_raw"),
        ("make", "model", "registration_date"),
        ("make", "model"),
        ("make",),
    ],
)
def test_fr09_the_answer_is_the_filter_and_order_of_fr09(
    client: TestClient, index: ComparablesIndex, car: dict, fields: tuple
):
    body = {name: car[name] for name in fields} | {"k": 20}

    found = answered(client.post(COMPARABLES, json=body))

    assert found, "the request names a listing of the index, so something must match"
    assert found == rows_of(expected(index, body))


@pytest.mark.req("FR-09")
def test_fr09_every_listing_matches_make_and_model_and_lies_in_both_windows(
    client: TestClient, car: dict, params: dict
):
    response = client.post(COMPARABLES, json=car | {"k": 20})

    listings = response.json()["listings"]
    assert listings
    car_age = age(car["registration_date"], datetime.now(UTC).date().isoformat())
    for listing in listings:
        assert (listing["make"], listing["model"]) == (car["make"], car["model"])
        assert (
            abs(listing["mileage_km_raw"] - car["mileage_km_raw"]) <= 0.25 * car["mileage_km_raw"]
        )
        # The listing's age when it was listed, on the snapshot date. The index
        # computed it from the full date, the answer gives the month: a month
        # of slack.
        listed_age = age(listing["registration_date"], params["reference_date"])
        assert abs(listed_age - car_age) <= 2 + 31 / 365.25
    distances = [abs(each["mileage_km_raw"] - car["mileage_km_raw"]) for each in listings]
    assert distances == sorted(distances)


@pytest.mark.req("FR-09")
def test_fr09_a_car_outside_every_window_gets_an_empty_list(client: TestClient, car: dict):
    far = car | {"mileage_km_raw": 999_000}
    old = car | {"registration_date": "1901-01"}

    for body in (far, old):
        response = client.post(COMPARABLES, json=body)
        assert response.status_code == 200, response.text
        assert response.json()["listings"] == []
        assert response.json()["warnings"] == []


@pytest.mark.req("FR-09")
def test_fr09_a_model_the_index_lacks_gets_an_empty_list_and_a_warning(
    client: TestClient, car: dict
):
    response = client.post(COMPARABLES, json=car | {"model": "No such model"})

    assert response.status_code == 200, response.text
    assert response.json()["listings"] == []
    assert [each["pointer"] for each in response.json()["warnings"]] == ["#/model"]


@pytest.mark.req("FR-09")
def test_fr09_k_defaults_to_5_and_is_capped_at_20(
    client: TestClient, index: ComparablesIndex, car: dict
):
    make = {"make": car["make"]}
    assert len(expected(index, make | {"k": 21})) > 20, "the fixture must have more than 20"

    assert len(answered(client.post(COMPARABLES, json=make))) == 5
    assert len(answered(client.post(COMPARABLES, json=make | {"k": 20}))) == 20
    for k in (0, 21):
        problem = assert_problem(client.post(COMPARABLES, json=make | {"k": k}), 422)
        assert pointers(problem) == ["#/k"]


@pytest.mark.req("FR-09", "FR-03")
def test_fr09_k_is_a_whole_number_like_the_other_integer_fields(client: TestClient, car: dict):
    make = {"make": car["make"]}

    assert len(answered(client.post(COMPARABLES, json=make | {"k": 5.0}))) == 5
    for k in ("5", True, 5.5):
        problem = assert_problem(client.post(COMPARABLES, json=make | {"k": k}), 422)
        assert pointers(problem) == ["#/k"]


@pytest.mark.req("FR-09")
def test_fr09_a_field_sent_as_null_is_a_field_left_out(client: TestClient, car: dict):
    nulls = {"make": car["make"], "model": car["model"], "registration_date": None}

    assert answered(client.post(COMPARABLES, json=nulls | {"mileage_km_raw": None})) == answered(
        client.post(COMPARABLES, json={"make": car["make"], "model": car["model"]})
    )


@pytest.mark.req("FR-09", "FR-02")
def test_fr09_the_request_is_the_price_range_body_plus_k():
    assert list(ComparablesRequest.model_fields) == [*PartialListing.model_fields, "k"]


@pytest.mark.req("FR-09", "FR-16")
def test_fr09_a_listing_is_checked_as_price_range_checks_it(client: TestClient, car: dict):
    for path in ("/v1/price-range", COMPARABLES):
        problem = assert_problem(
            client.post(path, json=car | {"make": "Trabant", "power_kw": -1}), 422
        )
        assert pointers(problem) == ["#/make", "#/power_kw"]
        assert problem["errors"][0]["allowed"]


@pytest.mark.req("FR-09", "NFR-08")
def test_fr09_a_listing_carries_its_price_and_key_features_and_no_pii(
    client: TestClient, car: dict
):
    listing = client.post(COMPARABLES, json=car).json()["listings"][0]

    assert set(listing) == {
        "make",
        "model",
        "model_version",
        "registration_date",
        "mileage_km_raw",
        "power_kw",
        "fuel_category",
        "transmission",
        "body_type",
        "seller_type",
        "country_code",
        "price_eur",
    }
    assert listing["price_eur"] > 0


@pytest.mark.req("FR-09", "FR-12")
def test_fr09_the_answer_names_the_index_and_the_model(
    client: TestClient, index: ComparablesIndex, car: dict
):
    body = client.post(COMPARABLES, json=car).json()
    health = client.get("/health").json()

    assert body["index"] == {"version": index.version, "listings": len(index)}
    assert body["index"] == health["comparables"]
    assert body["model"] == health["model"]


@pytest.mark.req("FR-09", "FR-11")
def test_fr09_metrics_count_the_listings_of_every_answer(client: TestClient, car: dict):
    def answers() -> float:
        for line in client.get("/metrics").text.splitlines():
            if line.startswith("recommenditos_comparables_listings_count"):
                return float(line.split()[-1])
        return 0.0

    before = answers()
    client.post(COMPARABLES, json=car)

    assert answers() == before + 1


@pytest.mark.req("FR-09", "FR-16")
def test_fr09_a_service_without_an_index_answers_503(bundle: Path, car: dict, tmp_path: Path):
    missing = tmp_path / INDEX_FILE
    settings = Settings(prediction_log=tmp_path / "log.sqlite3", comparables_file=missing)

    with TestClient(create_app(bundle, settings), raise_server_exceptions=False) as api:
        assert_problem(api.post(COMPARABLES, json=car), 503)
        assert api.get("/health").json()["comparables"] is None


def test_the_default_comparables_file_is_where_the_stage_writes_it():
    assert Settings.model_fields["comparables_file"].default == PROCESSED_DATA_DIR / INDEX_FILE
