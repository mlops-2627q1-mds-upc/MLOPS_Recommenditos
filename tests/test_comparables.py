"""The `comparables` stage and the search over its index (FR-09).

The stage tests run `build_comparables.main` on the `split` frames of the
synthetic fixture. The search tests build a small index by hand, so that every
listing's distance to the request is known and each window can be tested at its
bound; the API tests in `tests/api/test_comparables.py` then check the same
rules through the endpoint.
"""

from datetime import date
from pathlib import Path
import shutil

import pandas as pd
import pytest
from tests.conftest import PII_COLUMNS
import yaml

from recommenditos.api.comparables import (
    MAX_AGE_DIFF_YEARS,
    MAX_MILEAGE_DIFF_SHARE,
    ComparablesIndex,
    Query,
    file_md5,
)
from recommenditos.config import PARAMS_FILE, PROJ_ROOT
from recommenditos.data import build_comparables
from recommenditos.data.build_comparables import (
    COMPARABLES_SCHEMA,
    INDEX_FILE,
    SOURCES,
    build_index,
)
from recommenditos.data.split_data import HOLDOUT_NAME
from recommenditos.schema import PROCESSED_SCHEMA

#: Columns the index must never carry besides NFR-08's: the listing identifiers
#: of problem-spec section 4 and the seller pseudonym (EDN-20).
IDENTIFIERS = ("id", "german_hsn_tsn", "seller_group_id")


@pytest.fixture(scope="module")
def processed(matrices: dict) -> Path:
    return Path(matrices["processed"])


@pytest.fixture(scope="module")
def index_path(processed: Path, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("comparables") / INDEX_FILE
    build_comparables.main(processed, path, PARAMS_FILE)
    return path


def frames(processed: Path, *names: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(processed / f"{name}.parquet") for name in names], ignore_index=True
    )


def listing_keys(frame: pd.DataFrame) -> list[tuple]:
    """Each row as the values the index keeps from it, sorted, to compare row sets."""
    columns = ["make", "model", "model_version", "mileage_km_raw", "power_kw", "price"]
    return sorted(
        tuple("" if pd.isna(value) else str(value) for value in row)
        for row in frame[columns].itertuples(index=False)
    )


# -- the stage ------------------------------------------------------------


@pytest.mark.req("FR-09")
def test_the_index_holds_every_train_and_validation_listing_and_nothing_else(
    processed, index_path
):
    index = pd.read_parquet(index_path)
    expected = frames(processed, *SOURCES)

    assert SOURCES == ("train", "validation")
    assert listing_keys(index) == listing_keys(expected)
    for name in ("test", "calibration", HOLDOUT_NAME):
        held_back = pd.read_parquet(processed / f"{name}.parquet")
        assert not held_back.empty
        assert set(listing_keys(held_back)).isdisjoint(listing_keys(index)), name


@pytest.mark.req("FR-09")
def test_the_stage_reads_no_frame_but_train_and_validation(processed, tmp_path):
    # Only the two source frames exist, so a read of any other would fail.
    for name in SOURCES:
        shutil.copy(processed / f"{name}.parquet", tmp_path / f"{name}.parquet")

    build_comparables.main(tmp_path, tmp_path / INDEX_FILE, PARAMS_FILE)

    assert len(pd.read_parquet(tmp_path / INDEX_FILE)) == len(frames(processed, *SOURCES))


@pytest.mark.req("FR-09")
def test_dvc_yaml_gives_the_stage_train_and_validation_as_its_only_data():
    stage = yaml.safe_load((PROJ_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"][
        "comparables"
    ]
    data = [dep for dep in stage["deps"] if dep.startswith("data/")]

    assert data == [f"data/processed/{name}.parquet" for name in SOURCES]
    assert stage["outs"] == [f"models/{INDEX_FILE}"]


@pytest.mark.req("FR-09")
@pytest.mark.parametrize("name", ["test", "calibration", HOLDOUT_NAME])
def test_build_index_refuses_a_frame_outside_train_and_validation(processed, name):
    with pytest.raises(ValueError, match="FR-09"):
        build_index({"train": frames(processed, "train"), name: frames(processed, name)}, "", "")


@pytest.mark.req("FR-09")
def test_build_index_refuses_a_listing_of_the_holdout_country(processed, params):
    holdout_country = params["split"]["holdout_country"]
    train = frames(processed, "train")
    train.loc[train.index[0], "country_code"] = holdout_country

    with pytest.raises(ValueError, match="EDN-03"):
        build_index({"train": train}, params["reference_date"], holdout_country)


@pytest.mark.req("NFR-08")
def test_no_pii_column_or_identifier_reaches_the_index(index_path):
    columns = set(pd.read_parquet(index_path).columns)

    assert columns == set(COMPARABLES_SCHEMA.names)
    assert columns.isdisjoint(PII_COLUMNS)
    assert columns.isdisjoint(IDENTIFIERS)


def test_the_age_is_the_listings_age_on_the_snapshot_date(index_path, params):
    index = pd.read_parquet(index_path).dropna(subset=["registration_date"])
    reference = pd.Timestamp(params["reference_date"])
    registered = pd.to_datetime(index["registration_date"], format="%Y-%m")
    from_the_month = (reference - registered).dt.days / 365.25

    # The index keeps the month; the age was computed from the full date.
    assert not index.empty
    assert ((index["age_years"] - from_the_month).abs() <= 31 / 365.25).all()


def test_the_stage_writes_the_same_bytes_on_every_run(processed, index_path, tmp_path):
    build_comparables.main(processed, tmp_path / INDEX_FILE, PARAMS_FILE)

    assert file_md5(tmp_path / INDEX_FILE) == file_md5(index_path)


def test_the_served_index_reports_the_md5_dvc_records_for_it(index_path):
    index = ComparablesIndex.load(index_path)

    assert index.version == file_md5(index_path)
    assert len(index) == len(pd.read_parquet(index_path))


# -- the search -----------------------------------------------------------

TODAY = date(2026, 10, 1)
SNAPSHOT = "2025-11-08"


def index_of(tmp_path: Path, rows: list[dict]) -> ComparablesIndex:
    """An index of `rows`, built by the stage's own function from processed rows."""
    defaults = {name: None for name in PROCESSED_SCHEMA.names}
    defaults |= {"has_full_service_history": False, "non_smoking": False}
    defaults |= {"is_rental": False, "has_particle_filter": False, "seller_group_id": "s"}
    defaults |= {"country_code": "DE", "price": 10_000.0, "log_price": 9.2}
    frame = PROCESSED_SCHEMA.conform(pd.DataFrame([defaults | row for row in rows]))
    path = tmp_path / INDEX_FILE
    build_index({"train": frame}, SNAPSHOT, "ES").to_parquet(path, index=False)
    return ComparablesIndex.load(path)


def row(model: str, registered: str, mileage: float | None, price: float, make="BMW") -> dict:
    return {
        "make": make,
        "model": model,
        "registration_date": pd.Timestamp(f"{registered}-01"),
        "mileage_km_raw": mileage,
        "price": price,
    }


def age_today(registered: str) -> float:
    return (pd.Timestamp(TODAY) - pd.Timestamp(f"{registered}-01")).days / 365.25


def age_listed(registered: str) -> float:
    return (pd.Timestamp(SNAPSHOT) - pd.Timestamp(f"{registered}-01")).days / 365.25


def prices(found: pd.DataFrame) -> list[float]:
    return found["price"].tolist()


@pytest.mark.req("FR-09")
def test_every_match_shares_make_and_model_and_lies_in_both_windows(tmp_path):
    index = index_of(
        tmp_path,
        [
            row("320", "2019-06", 40_000, 1),
            row("320", "2019-06", 50_000, 2),  # +25 %: on the bound, in
            row("320", "2019-06", 50_001, 3),  # just over 25 %: out
            row("320", "2019-06", 29_999, 4),  # just under -25 %: out
            row("118", "2019-06", 40_000, 5),  # another model: out
            row("320", "2019-06", 40_000, 6, make="Audi"),  # another make: out
            row("320", "2010-01", 40_000, 7),  # far older: out
            row("320", "2019-06", None, 8),  # no mileage: cannot be shown in the window
        ],
    )
    query = Query(make="BMW", model="320", registration_date="2020-05", mileage_km=40_000)

    found = index.search(query, k=20, today=TODAY)

    assert prices(found) == [1, 2]
    assert (found["make"] == "BMW").all() and (found["model"] == "320").all()
    request_age = age_today("2020-05")
    assert ((found["age_years"] - request_age).abs() <= MAX_AGE_DIFF_YEARS).all()
    assert ((found["mileage_km_raw"] - 40_000).abs() <= MAX_MILEAGE_DIFF_SHARE * 40_000).all()


@pytest.mark.req("FR-09")
def test_the_age_window_is_two_years_either_way(tmp_path):
    request = "2020-01"
    candidates = ["2016-01", "2017-01", "2018-01", "2019-01", "2020-01", "2021-01", "2022-01"]
    index = index_of(
        tmp_path, [row("320", month, 40_000, price) for price, month in enumerate(candidates)]
    )
    query = Query(make="BMW", model="320", registration_date=request)

    found = index.search(query, k=20, today=TODAY)

    expected = [
        price
        for price, month in enumerate(candidates)
        if abs(age_listed(month) - age_today(request)) <= MAX_AGE_DIFF_YEARS
    ]
    assert 0 < len(expected) < len(candidates)
    assert sorted(prices(found)) == expected


@pytest.mark.req("FR-09")
def test_a_request_outside_every_window_gets_no_listing(tmp_path):
    index = index_of(tmp_path, [row("320", "2019-06", 40_000, 1)])

    assert index.search(Query("BMW", "320", "2019-06", 200_000), today=TODAY).empty
    assert index.search(Query("BMW", "320", "1990-01", 40_000), today=TODAY).empty
    assert index.search(Query("BMW", "118"), today=TODAY).empty
    assert index.search(Query("Audi"), today=TODAY).empty


@pytest.mark.req("FR-09")
def test_the_order_is_mileage_distance_then_age_distance(tmp_path):
    index = index_of(
        tmp_path,
        [
            row("320", "2019-06", 44_000, 1),  # mileage distance 4,000
            row("320", "2018-06", 41_000, 2),  # 1,000, age a year off
            row("320", "2019-06", 39_000, 3),  # 1,000, age matched
            row("320", "2019-06", 40_000, 4),  # 0
        ],
    )
    registered = "2020-05"  # the age today of a car registered 2020-05 ...
    assert abs(age_listed("2019-06") - age_today(registered)) < 0.1  # ... is 2019-06's then

    found = index.search(Query("BMW", "320", registered, 40_000), k=20, today=TODAY)

    assert prices(found) == [4, 3, 2, 1]


@pytest.mark.req("FR-09")
def test_omitted_fields_set_no_filter_and_no_distance(tmp_path):
    rows = [
        row("320", "2019-06", 300_000, 1),
        row("118", "2005-01", 10_000, 2),
        row("320", "2019-06", 40_000, 3),
        row("320", "2012-06", 40_500, 4),
    ]
    index = index_of(tmp_path, rows)

    # make alone: every BMW, in the index's order (make, model, age, mileage, price).
    assert prices(index.search(Query("BMW"), k=20, today=TODAY)) == [2, 3, 1, 4]
    # No mileage: the far-driven 320 is in, and only the age orders, here a tie.
    assert prices(index.search(Query("BMW", "320", "2020-05"), k=20, today=TODAY)) == [3, 1]
    # No registration date: the old 320 is in, and the order is the mileage distance alone.
    found = index.search(Query("BMW", "320", mileage_km=40_000), k=20, today=TODAY)
    assert prices(found) == [3, 4]


@pytest.mark.req("FR-09")
def test_k_limits_the_answer_to_the_closest_listings(tmp_path):
    index = index_of(
        tmp_path, [row("320", "2019-06", 40_000 + 100 * price, price) for price in range(30)]
    )
    query = Query("BMW", "320", mileage_km=40_000)

    assert prices(index.search(query, today=TODAY)) == [0, 1, 2, 3, 4]
    assert prices(index.search(query, k=20, today=TODAY)) == list(range(20))


@pytest.mark.req("FR-09")
def test_the_same_request_gets_the_same_listings_when_distances_tie(tmp_path):
    index = index_of(tmp_path, [row("320", "2019-06", 40_000, price) for price in (3, 1, 2)])

    found = index.search(Query("BMW", "320", "2019-06", 40_000), k=20, today=TODAY)

    # The tie is broken by the index's order, which sorts by price last.
    assert prices(found) == [1, 2, 3]
