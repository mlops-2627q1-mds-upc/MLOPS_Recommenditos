"""The data contract itself.

Every stage trusts `recommenditos.schema` to catch a broken frame, so these are
the tests that make that trust reasonable.
"""

import json

import pandas as pd
import pytest

from recommenditos.schema import (
    FEATURE_SPACE_FILE,
    INTERIM_SCHEMA,
    PROCESSED_SCHEMA,
    RAW_SCHEMA,
    TARGET_NAMES,
    Column,
    FeatureSpaceError,
    Schema,
    SchemaError,
    feature_schema,
    load_feature_schema,
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

#: A contract with a categorical, which is the only dtype whose meaning depends
#: on the data: `LEVELLED`'s codes are positions in this exact list.
LEVELS = ("Audi", "BMW", "Porsche", "Volvo")
LEVELLED = Schema(
    name="levelled",
    columns=(
        Column("make", "str", True, "manufacturer"),
        Column("score", "float64", True, "optional number"),
    ),
).as_category("make", LEVELS)


def _levelled_frame(*makes: "str | None") -> pd.DataFrame:
    """A frame for `LEVELLED` whose `make` column holds exactly `makes`, as text."""
    return pd.DataFrame(
        {
            "make": pd.Series(list(makes), dtype="str"),
            "score": pd.Series([1.5] * len(makes), dtype="float64"),
        }
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


def test_interim_nullability_is_structural_not_a_fill_rate():
    # `body_type` and `make` are filled in every row of the published
    # snapshot, but that is an expectation for the Great Expectations suite to
    # assert with a tolerance, not a promise this contract makes. A bool column
    # is non-null because the dtype cannot hold a missing value at all.
    assert INTERIM_SCHEMA.column("body_type").nullable
    assert INTERIM_SCHEMA.column("make").nullable
    assert INTERIM_SCHEMA.column("equipment_comfort").nullable
    assert not INTERIM_SCHEMA.column("has_full_service_history").nullable

    # What preprocess constructs, it guarantees.
    for name in ("price", "log_price", "seller_group_id"):
        assert not INTERIM_SCHEMA.column(name).nullable, name


def test_extend_rejects_a_column_name_twice_in_its_own_argument():
    # The seam issue #36 uses for the equipment multi-hot columns. Two items
    # that normalise to the same name must fail here, naming the column, not
    # later inside conform with a TypeError.
    duplicated = (
        Column("eq_abs", "bool", False, ""),
        Column("eq_abs", "bool", False, ""),
    )
    with pytest.raises(SchemaError, match="would repeat column"):
        TOY.extend(duplicated)


def test_drop_and_with_dtype_keep_a_stage_out_of_this_module():
    # What build_features needs: replace the equipment strings with what it
    # built, and change a parsed column's type, without editing schema.py.
    reshaped = TOY.drop(["flag"]).with_dtype("score", "Int64", name="reshaped")

    assert "flag" not in reshaped.names
    assert reshaped.column("score").dtype == "Int64"
    with pytest.raises(SchemaError, match="no column"):
        TOY.with_dtype("nope", "float64")


# --------------------------------------------------------------------------
# Categoricals: the one dtype whose meaning depends on the data
# --------------------------------------------------------------------------


def test_a_categorical_column_cannot_be_declared_without_its_levels():
    # `str(dtype)` is 'category' whatever the levels are, so a contract that
    # omitted them would say nothing about what a code means. Refusing here is
    # what makes the written artefact lossless rather than nearly lossless.
    with pytest.raises(SchemaError, match="declares no levels"):
        Column("make", "category", True, "manufacturer")


def test_only_a_categorical_column_may_declare_levels():
    with pytest.raises(SchemaError, match="only a 'category' column has levels"):
        Column("score", "float64", True, "number", levels=("1", "2"))


def test_with_dtype_refuses_to_make_a_column_categorical():
    # Because it takes no levels. Routing every categorical through
    # `as_category` is what leaves no code path that produces a contract saying
    # 'category' with the levels left for a consumer to infer.
    with pytest.raises(SchemaError, match="use as_category"):
        TOY.with_dtype("name", "category")


def test_the_levels_survive_the_round_trip_through_the_artefact():
    written = LEVELLED.to_dicts()

    assert written[0]["levels"] == list(LEVELS)
    # And a column with no levels does not carry the field at all, so the
    # artefact stays readable rather than 30 lines of `"levels": null`.
    assert "levels" not in written[1]
    assert Schema.from_dicts(written, name="levelled") == LEVELLED


def test_validate_refuses_the_same_levels_in_a_different_order():
    # The failure this whole mechanism exists for. Reversing the levels changes
    # every code in the column while `str(dtype)` stays 'category', so a
    # consumer reading `.cat.codes` gets numbers the model never saw.
    frame = LEVELLED.conform(_levelled_frame("Audi", "Volvo", "BMW"))
    reordered = frame.copy()
    reordered["make"] = frame["make"].cat.set_categories(list(reversed(LEVELS)))

    assert reordered["make"].cat.codes.tolist() != frame["make"].cat.codes.tolist()
    with pytest.raises(SchemaError, match="the same levels in a different order"):
        LEVELLED.validate(reordered)


def test_validate_names_a_level_that_is_absent_and_one_that_was_added():
    frame = LEVELLED.conform(_levelled_frame("Audi", "BMW"))
    narrowed = frame.copy()
    narrowed["make"] = (
        frame["make"].astype("object").astype(pd.CategoricalDtype(("Audi", "BMW", "Fiat")))
    )

    with pytest.raises(SchemaError) as raised:
        LEVELLED.validate(narrowed)

    message = str(raised.value)
    assert "'Porsche', 'Volvo'] absent" in message
    assert "'Fiat'] added" in message


def test_conform_applies_the_declared_levels_rather_than_inferring_them():
    # `astype('category')` infers the levels from the rows in hand, so a subset
    # would come back numbered differently from the matrix a model was fitted
    # on. This is what #37's ridge variant and #39's masking sweep rely on.
    conformed = LEVELLED.conform(_levelled_frame("Volvo", "Audi"))

    assert tuple(conformed["make"].cat.categories) == LEVELS
    assert conformed["make"].cat.codes.tolist() == [3, 0]


def test_conform_corrects_an_input_that_is_already_categorical_over_other_levels():
    # A frame read back from Parquet arrives as a categorical already, and a
    # stale or hand-built one can carry the wrong levels; the cast has to correct
    # it rather than pass it through because `str(dtype)` matches.
    frame = _levelled_frame("Volvo", "Audi")
    frame["make"] = frame["make"].astype(pd.CategoricalDtype(("Volvo", "Audi")))

    conformed = LEVELLED.conform(frame)

    assert tuple(conformed["make"].cat.categories) == LEVELS
    assert conformed["make"].cat.codes.tolist() == [3, 0]


def test_a_value_outside_the_declared_levels_becomes_missing():
    conformed = LEVELLED.conform(_levelled_frame("BMW", "Lada"))

    assert conformed["make"].iloc[0] == "BMW"
    assert conformed["make"].isna().tolist() == [False, True]
    assert "Lada" not in conformed["make"].cat.categories


# --------------------------------------------------------------------------
# Features and targets
# --------------------------------------------------------------------------


def test_a_contract_separates_its_features_from_its_targets(params):
    schema = feature_schema(params["features"]["sets"]["basic"], name="basic")

    assert schema.target_names == TARGET_NAMES
    assert set(schema.feature_names).isdisjoint(TARGET_NAMES)
    assert set(schema.feature_names) | set(schema.target_names) == set(schema.names)


def test_the_feature_only_contract_is_what_a_request_is_checked_against(params):
    # A request describes the car and not its price, so conforming it to the
    # full contract would reject it for a column no caller could supply.
    schema = feature_schema(params["features"]["sets"]["basic"], name="basic")

    features = schema.features()

    assert features.names == schema.feature_names
    assert "price" not in features.names


# --------------------------------------------------------------------------
# The feature-space artefact: every way it can be stale or wrong
# --------------------------------------------------------------------------


def _artefact(tmp_path, payload) -> "pd.Series":
    (tmp_path / FEATURE_SPACE_FILE).write_text(payload, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("truncated", '{"schema": [{"name": "make"'),
        ("empty", ""),
        ("not an object", "[]"),
        ("no schema section", '{"vocabulary": {}}'),
        ("no vocabulary section", '{"schema": []}'),
        ("a zero-column contract", '{"schema": [], "vocabulary": {}}'),
        (
            "a row that lost a field",
            json.dumps({"schema": [{"name": "make", "dtype": "str"}], "vocabulary": {}}),
        ),
        (
            "a row that gained a field",
            json.dumps(
                {
                    "schema": [
                        {
                            "name": "make",
                            "dtype": "str",
                            "nullable": True,
                            "description": "",
                            "unit": "kg",
                        }
                    ],
                    "vocabulary": {},
                }
            ),
        ),
    ],
)
def test_a_broken_feature_space_artefact_names_the_file_and_the_stage(tmp_path, label, payload):
    # Every one of these used to surface as a bare JSONDecodeError, KeyError or
    # TypeError inside whichever stage happened to read the file next, which
    # says nothing about which artefact has to be rebuilt. The zero-column case
    # was worse: it loaded, and then reported all 164 columns as unexpected.
    with pytest.raises(FeatureSpaceError) as raised:
        load_feature_schema(_artefact(tmp_path, payload), name="features-extended")

    message = str(raised.value)
    assert FEATURE_SPACE_FILE in message, label
    assert "dvc repro features" in message, label


def test_a_missing_feature_space_artefact_names_the_file_and_the_stage(tmp_path):
    with pytest.raises(FeatureSpaceError, match="dvc repro features"):
        load_feature_schema(tmp_path / "nowhere", name="features-extended")


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
