"""The data contract itself.

Every stage trusts `recommenditos.schema` to catch a broken frame, so these are
the tests that make that trust reasonable.
"""

import pandas as pd
import pytest

from recommenditos.schema import (
    INTERIM_SCHEMA,
    PROCESSED_SCHEMA,
    RAW_SCHEMA,
    Column,
    Schema,
    SchemaError,
    feature_schema,
)

TOY = Schema(
    name="toy",
    columns=(
        Column("name", "str", False, "required text"),
        Column("score", "float64", True, "optional number"),
        Column("flag", "bool", False, "required boolean"),
        Column("seen_at", "datetime64[ns]", True, "optional timestamp"),
    ),
)


def _good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": pd.Series(["a", "b"], dtype="str"),
            "score": pd.Series([1.5, None], dtype="float64"),
            "flag": pd.Series([True, False], dtype="bool"),
            "seen_at": pd.to_datetime(pd.Series(["2025-01-01", None])).astype("datetime64[ns]"),
        }
    )


def test_a_conforming_frame_validates():
    TOY.validate(_good_frame())


def test_validate_reports_every_problem_at_once():
    frame = _good_frame().drop(columns=["score"])
    frame["flag"] = pd.Series([True, None], dtype="object")
    frame["extra"] = 1

    with pytest.raises(SchemaError) as raised:
        TOY.validate(frame)

    message = str(raised.value)
    assert "missing column 'score'" in message
    assert "unexpected column 'extra'" in message
    assert "expected dtype bool, got object" in message
    assert "may not be null" in message
    # One raise, not one per problem: a stage should see the whole picture.
    assert "4 problem(s)" in message


def test_validate_names_the_offending_rows_for_a_null():
    frame = _good_frame()
    frame.loc[1, "name"] = None

    with pytest.raises(SchemaError, match=r"row position\(s\) \[1\]"):
        TOY.validate(frame)


def test_conform_casts_and_orders():
    frame = _good_frame()[["flag", "name", "score", "seen_at"]]
    frame["score"] = frame["score"].astype("object")

    conformed = TOY.conform(frame)

    assert list(conformed.columns) == list(TOY.names)
    assert str(conformed["score"].dtype) == "float64"


def test_conform_drops_a_column_the_contract_does_not_name():
    frame = _good_frame()
    frame["seller_company_name"] = "Example Motors"

    assert "seller_company_name" not in TOY.conform(frame).columns


def test_conform_refuses_a_null_in_a_required_boolean():
    # astype("bool") would turn the missing value into True, so conform has to
    # look before it casts.
    frame = _good_frame()
    frame["flag"] = pd.Series([True, None], dtype="object")

    with pytest.raises(SchemaError, match="may not be null"):
        TOY.conform(frame)


def test_conform_refuses_a_missing_column():
    with pytest.raises(SchemaError, match="missing column 'name'"):
        TOY.conform(_good_frame().drop(columns=["name"]))


def test_every_contract_survives_a_parquet_round_trip(raw_frame, tmp_path):
    # Parquet preserves whatever dtype it is handed rather than normalising it,
    # so this is the property that keeps the stages agreeing with each other.
    path = tmp_path / "round-trip.parquet"
    raw_frame.to_parquet(path, index=False)

    RAW_SCHEMA.validate(pd.read_parquet(path))


def test_the_interim_contract_carries_no_excluded_column():
    forbidden = {
        "vin",
        "street",
        "zip",
        "city",
        "latitude",
        "longitude",
        "seller_company_name",
        "price_net",
        "price_vat_rate",
        "price_tax_deductible",
        "price_negotiable",
        "id",
        "german_hsn_tsn",
        "description",
        "ratings_average",
        "ratings_count",
        "ratings_recommend_percentage",
        "is_used",
        "is_new",
        "is_preregistered",
        "had_accident",
        "offer_type",
        "vehicle_type",
        "price_currency",
        "mileage_km",
        "power_hp",
        "body_color_original",
        "primary_fuel",
        "seller_is_dealer",
    }
    assert forbidden.isdisjoint(INTERIM_SCHEMA.names)


def test_the_interim_contract_carries_the_target_and_the_group_key():
    assert {"price", "log_price", "seller_group_id"}.issubset(INTERIM_SCHEMA.names)


def test_the_split_preserves_the_interim_columns():
    assert PROCESSED_SCHEMA.names == INTERIM_SCHEMA.names
    assert PROCESSED_SCHEMA.name == "processed"


def test_both_feature_sets_of_params_resolve(params):
    for name, columns in params["features"]["sets"].items():
        schema = feature_schema(columns, name=name)
        assert set(columns).issubset(schema.names)
        # The target travels with the matrix, so `train` and `evaluate` need
        # only one file per split.
        assert {"price", "log_price"}.issubset(schema.names)


def test_a_feature_set_naming_an_unknown_column_fails_loudly():
    with pytest.raises(SchemaError, match="no stage produces: horsepower"):
        feature_schema(["make", "horsepower"], name="typo")


def test_the_basic_feature_set_matches_the_problem_specification(params):
    # problem-spec section 4, "Basic feature set". Age is derived, the rest are
    # raw column names.
    assert set(params["features"]["sets"]["basic"]) == {
        "make",
        "model",
        "body_type",
        "age_years",
        "mileage_km_raw",
        "nr_prev_owners",
        "power_kw",
        "fuel_category",
        "transmission",
        "drive_train",
        "gears",
        "cylinders_volume_cc",
        "nr_seats",
        "nr_doors",
        "country_code",
        "seller_type",
    }


def test_the_extended_set_is_a_superset_of_the_basic_one(params):
    sets = params["features"]["sets"]
    assert set(sets["basic"]).issubset(sets["extended"])
