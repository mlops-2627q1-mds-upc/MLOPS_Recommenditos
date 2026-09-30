"""The `features` stage: what it derives, what it encodes, and what it refuses to lose.

The frames below are built straight against `PROCESSED_SCHEMA` instead of by
running `preprocess` and `split`, for two reasons. Every value an assertion
depends on is then chosen by the test, which is what lets a threshold or an
unseen level be tested at all on ten rows. And this module stays independent of
two stages that are being replaced in parallel: like every stage, this one codes
against the contract rather than against the stage before it.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from recommenditos.data import build_features
from recommenditos.pipeline import write_frame
from recommenditos.schema import PROCESSED_SCHEMA, load_feature_schema

#: The reference date of the snapshot, as params.yaml carries it.
REFERENCE_DATE = "2025-11-08"

#: The columns problem-spec section 4 excludes, by the reason it gives. None of
#: them may reach a feature matrix, whatever a feature set happens to name.
EXCLUDED_COLUMNS = (
    # Leakage: derived from the target, or a listing option tied to the price.
    "price_net",
    "price_vat_rate",
    "price_tax_deductible",
    "price_negotiable",
    # Identifiers.
    "id",
    "vin",
    "german_hsn_tsn",
    # PII or exact location.
    "street",
    "zip",
    "city",
    "latitude",
    "longitude",
    "seller_company_name",
    # Describe the seller, not the car.
    "ratings_average",
    "ratings_count",
    "ratings_recommend_percentage",
    # Leaks the price and is multilingual.
    "description",
    # Empty, or true in three rows.
    "warranty",
    "has_warranty",
    "fuel_cons_city_l100_km",
    "fuel_cons_highway_l100_km",
    "had_accident",
    # Constant after scoping, or the scope filter itself.
    "price_currency",
    "offer_type",
    "is_new",
    "vehicle_type",
    "is_used",
    "is_preregistered",
    # Duplicate another column.
    "mileage_km",
    "power_hp",
    "body_color_original",
    "primary_fuel",
    "seller_is_dealer",
    # Sparse, and rarely known by a user.
    "production_year",
    "electric_range_city_km",
    "fuel_cons_comb_l100_km",
    "co2_emission_grper_km",
    "fuel_cons_comb_l100_wltp_km",
    "fuel_cons_electric_comb_l100_wltp_km",
    "co2_emission_grper_wltp_km",
)

#: Dtypes a gradient booster takes without a conversion step in front of it.
MODEL_READY_DTYPES = frozenset({"category", "float64", "bool", "boolean"})

_PRICE = 10_000.0

#: A `processed` row a test can override one field of.
#:
#: Everything the contract does not insist on is null, except two groups. The
#: equipment lists are the empty list, because that is what the published file
#: writes for a listing with no equipment in a category, and never null. And
#: every categorical carries a level, because a categorical the training rows
#: never observe fails the stage on purpose - a test that wants one missing says
#: so with an explicit `None`.
_MINIMAL_ROW = {
    column.name: (False if column.dtype == "bool" else None) for column in PROCESSED_SCHEMA.columns
} | {
    "price": _PRICE,
    "log_price": math.log(_PRICE),
    "seller_group_id": "group",
    "equipment_comfort": "[]",
    "equipment_entertainment": "[]",
    "equipment_extra": "[]",
    "equipment_safety": "[]",
    "make": "BMW",
    "model": "3 Series",
    "model_version": "320d Touring",
    "body_type": "Sedan",
    "fuel_category": "Diesel",
    "transmission": "Automatic",
    "drive_train": "Rear Wheel Drive",
    "country_code": "DE",
    "seller_type": "Dealer",
    "body_color": "Black",
    "paint_type": "Metallic",
    "upholstery": "Cloth",
    "upholstery_color": "Grey",
    "envir_standard": "Euro 6",
    "original_market": "Germany",
}


def processed_frame(records: "list[dict]") -> pd.DataFrame:
    """A `processed`-valid frame from just the fields the records name."""
    return PROCESSED_SCHEMA.conform(pd.DataFrame([_MINIMAL_ROW | record for record in records]))


def comfort(*items: str) -> dict:
    """One row whose comfort equipment is `items`, as the raw column encodes it."""
    return {"equipment_comfort": repr(list(items))}


@pytest.fixture
def run_stage(tmp_path: Path, params: dict):
    """Run the stage over frames a test provides, and hand back where it wrote.

    Frames not named fall back to the training one, because the point of most
    tests here is what one split does, not how five of them differ.
    """

    def run(frames: dict, feature_set: str = "extended", **feature_overrides) -> Path:
        processed = tmp_path / "processed"
        for name in build_features.FEATURE_INPUTS:
            frame = frames.get(name, frames[build_features.VOCABULARY_SPLIT])
            write_frame(frame, processed / f"{name}.parquet", PROCESSED_SCHEMA)

        overridden = {**params, "features": {**params["features"], **feature_overrides}}
        params_path = tmp_path / "params.yaml"
        params_path.write_text(yaml.safe_dump(overridden), encoding="utf-8")

        build_features.main(feature_set, processed, tmp_path / "features", params_path)
        return tmp_path / "features" / feature_set

    return run


def matrix(directory: Path, split: str = "train") -> pd.DataFrame:
    return pd.read_parquet(directory / f"{split}.parquet")


# --------------------------------------------------------------------------
# Age
# --------------------------------------------------------------------------


def test_age_is_the_reference_date_minus_the_registration_date():
    ages = build_features.age_years(
        pd.Series(pd.to_datetime(["2015-11-01", "2024-11-01"])), REFERENCE_DATE
    )
    assert ages.iloc[0] == pytest.approx(3660 / 365.25)
    assert ages.iloc[1] == pytest.approx(372 / 365.25)


def test_a_car_registered_in_the_reference_month_is_days_old_not_months_old():
    # Registrations are always the first of a month, so the newest car in the
    # snapshot is seven days old. Counting in whole months would make it zero,
    # and the first year is the steepest part of the depreciation curve.
    ages = build_features.age_years(pd.Series([pd.Timestamp("2025-11-01")]), REFERENCE_DATE)

    assert ages.iloc[0] == pytest.approx(7 / 365.25)
    assert ages.iloc[0] > 0


def test_age_is_missing_where_the_registration_date_is(run_stage):
    written = matrix(
        run_stage({"train": processed_frame([{"registration_date": "2020-01-01"}, {}])})
    )

    assert written["age_years"].notna().iloc[0]
    assert written["age_years"].isna().iloc[1]


# --------------------------------------------------------------------------
# The parsed and normalised columns
# --------------------------------------------------------------------------


def test_weight_kg_is_parsed_out_of_its_text_form():
    parsed = build_features.weight_kg(pd.Series(["1,945 kg", "894 kg", None], dtype="str"))

    assert parsed.tolist()[:2] == [1945.0, 894.0]
    assert str(parsed.dtype) == "float64"


def test_weight_kg_stays_missing_where_the_text_is():
    # Not zero, and not the mean: a weight nobody filled in is a fact about the
    # listing, and the model is meant to learn from it (EDN-15).
    assert build_features.weight_kg(pd.Series([None], dtype="str")).isna().all()


def test_a_weight_that_is_not_a_weight_fails_the_stage_rather_than_going_missing():
    # Coercing it would turn a changed upstream format into a column that
    # quietly lost a fifth of its values, which no structural check would catch.
    with pytest.raises(ValueError, match="not a weight in kg"):
        build_features.weight_kg(pd.Series(["1945 pounds"], dtype="str"))


def test_model_version_is_normalised_to_its_leading_token():
    normalised = build_features.model_version(
        pd.Series(["M40i xDrive | nahezu Vollausstattung", "Avant TDI 150 kW"], dtype="str"),
        tokens=1,
    )

    assert normalised.tolist() == ["m40i", "avant"]


def test_model_version_keeps_a_decimal_point_inside_a_number():
    # '1.2' is an engine size, so splitting on the dot would throw away the
    # informative half of the trim; the dot in '5p.ti' is punctuation.
    normalised = build_features.model_version(
        pd.Series(["1.2 Style Smart Hybrid", "5p.ti cross turismo"], dtype="str"), tokens=1
    )

    assert normalised.tolist() == ["1.2", "5p"]


def test_a_trim_of_nothing_but_punctuation_is_missing_rather_than_a_level():
    normalised = build_features.model_version(pd.Series(["***", None], dtype="str"), tokens=1)

    assert normalised.isna().all()


def test_a_trim_below_the_frequency_threshold_is_missing_rather_than_its_own_level(run_stage):
    # Two of ten rows carry 'Avant', one carries a trim of its own. At a
    # threshold of 0.15 only 'Avant' is a level; the singleton is the long tail
    # the raw column is 80,000 values of.
    frame = processed_frame(
        [{"model_version": "Avant TDI"}, {"model_version": "Avant quattro"}]
        + [{"model_version": "Unique Special Edition"}]
        + [{"model_version": None}] * 7
    )
    written = matrix(run_stage({"train": frame}, model_version_min_frequency=0.15))

    assert list(written["model_version"].cat.categories) == ["avant"]
    assert written["model_version"].tolist()[:2] == ["avant", "avant"]
    assert written["model_version"].isna().iloc[2]


# --------------------------------------------------------------------------
# The equipment multi-hot columns
# --------------------------------------------------------------------------


def test_an_item_above_the_threshold_gets_a_column_and_one_below_it_does_not(run_stage):
    # 'Cruise control' is in 4 of 10 rows, 'Tow bar' in 1. At a threshold of 0.2
    # the first earns a column and the second does not, so the matrix does not
    # grow a tail of near-constant columns.
    frame = processed_frame(
        [comfort("Cruise control")] * 4 + [comfort("Tow bar")] + [comfort()] * 5
    )
    directory = run_stage({"train": frame}, equipment_min_frequency=0.2)
    written = matrix(directory)

    assert "equipment_comfort_cruise_control" in written.columns
    assert "equipment_comfort_tow_bar" not in written.columns
    assert written["equipment_comfort_cruise_control"].tolist() == [True] * 4 + [False] * 6
    # The source list is replaced by what was built from it, never carried too.
    assert "equipment_comfort" not in written.columns


def test_an_empty_equipment_list_is_every_item_false_not_missing(run_stage):
    # '[]' is what the file writes for a listing with no equipment in a
    # category. The list is there and it is empty, which is not the same as
    # nobody having said - so the row is False throughout, not null.
    frame = processed_frame([comfort("Cruise control")] * 5 + [comfort()] * 5)
    written = matrix(run_stage({"train": frame}))
    column = written["equipment_comfort_cruise_control"]

    assert column.tolist() == [True] * 5 + [False] * 5
    assert column.notna().all()


def test_a_null_equipment_list_is_missing_throughout(run_stage):
    # The published file never leaves one null, but the contract allows it, and
    # a null list is "nobody said" rather than "no equipment" - so it must not
    # become a row of Falses the model reads as an assertion.
    frame = processed_frame(
        [comfort("Cruise control")] * 5 + [comfort()] * 4 + [{"equipment_comfort": None}]
    )
    column = matrix(run_stage({"train": frame}))["equipment_comfort_cruise_control"]

    assert column.isna().tolist() == [False] * 9 + [True]


def test_an_item_only_the_other_splits_have_gets_no_column(run_stage):
    # The vocabulary is the training rows'. An item that first appears in
    # validation, test or the holdout getting a column of its own would be those
    # frames deciding the feature space, which is a leak.
    directory = run_stage(
        {
            "train": processed_frame([comfort("Cruise control")] * 10),
            "test": processed_frame([comfort("Massage seats")] * 10),
        }
    )
    vocabulary = build_features.load_vocabulary(directory)

    assert vocabulary.equipment["equipment_comfort"] == ("Cruise control",)
    assert "equipment_comfort_massage_seats" not in matrix(directory, "test").columns
    # And the item it does not know is simply not listed, not an error.
    assert not matrix(directory, "test")["equipment_comfort_cruise_control"].any()


def test_every_split_gets_the_same_columns_in_the_same_order(run_stage):
    directory = run_stage(
        {
            "train": processed_frame([comfort("Cruise control")] * 10),
            "validation": processed_frame([comfort("Massage seats")] * 4),
            "holdout_es": processed_frame([comfort()] * 3),
        }
    )
    columns = list(matrix(directory).columns)

    for split in build_features.FEATURE_INPUTS:
        assert list(matrix(directory, split).columns) == columns


def test_a_feature_name_carries_no_character_a_booster_rejects():
    # LightGBM refuses a feature name holding a JSON character, and the snapshot
    # has items like 'Alloy wheels (18")'.
    name = build_features.equipment_feature_name("equipment_extra", 'Alloy wheels (18")')

    assert name == "equipment_extra_alloy_wheels_18"
    assert not set(name) - set("abcdefghijklmnopqrstuvwxyz0123456789_")


# --------------------------------------------------------------------------
# Categoricals and missing values (EDN-02, EDN-15)
# --------------------------------------------------------------------------


def test_a_categorical_stays_one_column_of_levels_rather_than_becoming_one_hot(run_stage):
    frame = processed_frame([{"make": "BMW"}, {"make": "Audi"}, {"make": "BMW"}])
    written = matrix(run_stage({"train": frame}))

    assert str(written["make"].dtype) == "category"
    assert list(written["make"].cat.categories) == ["Audi", "BMW"]
    assert [column for column in written.columns if column.startswith("make")] == ["make"]


def test_a_categorical_keeps_the_training_levels_in_every_split(run_stage):
    # The codes have to mean the same car in every split and at request time, so
    # the levels are the training rows' even where a split uses none of them.
    directory = run_stage(
        {
            "train": processed_frame([{"make": "BMW"}, {"make": "Audi"}]),
            "test": processed_frame([{"make": "BMW"}]),
        }
    )

    assert list(matrix(directory, "test")["make"].cat.categories) == ["Audi", "BMW"]


def test_a_level_the_training_rows_never_saw_is_missing_not_a_new_code(run_stage):
    directory = run_stage(
        {
            "train": processed_frame([{"make": "BMW"}] * 2),
            "test": processed_frame([{"make": "BMW"}, {"make": "Porsche"}]),
        }
    )
    written = matrix(directory, "test")

    assert written["make"].iloc[0] == "BMW"
    assert written["make"].isna().iloc[1]
    assert "Porsche" not in written["make"].cat.categories


def test_a_categorical_the_training_rows_never_observe_fails_the_stage(run_stage):
    # Its matrix column would be entirely missing, and Parquet cannot even carry
    # a categorical with no level: it comes back as text and breaks whichever
    # stage reads the matrix next. Failing here says which column and why.
    frame = processed_frame([{"original_market": None}] * 5)

    with pytest.raises(ValueError, match="no level in the 5 train rows for: original_market"):
        run_stage({"train": frame})


def test_no_column_of_the_matrix_is_imputed(run_stage):
    # Every column the stage passes through keeps its holes exactly. A filled
    # column would be this stage deciding what a missing gearbox count means,
    # which is the model's job (EDN-15).
    frame = processed_frame(
        [
            {"gears": 6.0, "nr_prev_owners": 1.0, "weight_kg": "1,945 kg"},
            {"upholstery": None},
            {"gears": 7.0, "upholstery": None},
            {},
            {"nr_prev_owners": 2.0},
        ]
    )
    written = matrix(run_stage({"train": frame}))

    for column in ("gears", "nr_prev_owners", "weight_kg", "upholstery"):
        assert written[column].isna().tolist() == frame[column].isna().tolist(), column


# --------------------------------------------------------------------------
# What may and may not reach a matrix
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-08")
def test_no_excluded_column_of_the_specification_reaches_a_matrix(run_stage, params):
    for feature_set in params["features"]["sets"]:
        directory = run_stage(
            {"train": processed_frame([comfort("Cruise control")] * 10)}, feature_set
        )
        for split in build_features.FEATURE_INPUTS:
            present = set(matrix(directory, split).columns)
            assert present.isdisjoint(EXCLUDED_COLUMNS), feature_set


def test_a_matrix_holds_its_feature_set_the_multi_hot_columns_and_the_target_only(
    run_stage, params
):
    columns = params["features"]["sets"]["extended"]
    directory = run_stage({"train": processed_frame([comfort("Cruise control")] * 10)})
    vocabulary = build_features.load_vocabulary(directory)

    expected = (
        (set(columns) - set(vocabulary.equipment))
        | {"price", "log_price"}
        | {
            build_features.equipment_feature_name(source, item)
            for source, items in vocabulary.equipment.items()
            for item in items
        }
    )
    assert set(matrix(directory).columns) == expected


def test_every_matrix_column_is_a_dtype_a_gradient_booster_takes(run_stage, params):
    # `train` fits on these frames directly, so a text column here would mean
    # every estimator needing a conversion step of its own - and each one being
    # a chance for the codes to differ between training and serving.
    for feature_set in params["features"]["sets"]:
        directory = run_stage(
            {"train": processed_frame([comfort("Cruise control")] * 10)}, feature_set
        )
        written = matrix(directory)
        unusable = {str(dtype) for dtype in written.dtypes} - MODEL_READY_DTYPES
        assert not unusable, f"{feature_set}: {unusable}"


# --------------------------------------------------------------------------
# The artefact, and switching feature sets
# --------------------------------------------------------------------------


def test_the_feature_space_artefact_sits_beside_the_matrices(run_stage, params):
    for feature_set in params["features"]["sets"]:
        directory = run_stage(
            {"train": processed_frame([comfort("Cruise control")] * 10)}, feature_set
        )
        # Inside the directory the stage declares as its output, so `dvc repro`
        # versions it with the matrices and neither can be restored alone.
        assert (directory / "feature_space.json").exists()


def test_the_vocabulary_round_trips_through_the_artefact(run_stage):
    directory = run_stage({"train": processed_frame([comfort("Cruise control")] * 10)})
    written = json.loads((directory / "feature_space.json").read_text(encoding="utf-8"))

    assert build_features.Vocabulary.from_dict(written["vocabulary"]) == (
        build_features.load_vocabulary(directory)
    )


def test_the_written_contract_is_the_one_the_stage_built(run_stage, params):
    # The API rebuilds the contract from the vocabulary; every other stage loads
    # the written one. The two must not be able to disagree.
    directory = run_stage({"train": processed_frame([comfort("Cruise control")] * 10)})
    vocabulary = build_features.load_vocabulary(directory)

    rebuilt = build_features.matrix_schema(
        params["features"]["sets"]["extended"], vocabulary, name="features-extended"
    )
    assert load_feature_schema(directory, name="features-extended") == rebuilt


def test_switching_the_feature_set_is_a_params_change_only(run_stage, params):
    frames = {"train": processed_frame([comfort("Cruise control")] * 10)}

    basic = matrix(run_stage(frames, "basic"))
    extended = matrix(run_stage(frames, "extended"))

    assert not [column for column in basic.columns if column.startswith("equipment_")]
    assert "equipment_comfort_cruise_control" in extended.columns
    assert set(params["features"]["sets"]["basic"]).issubset(extended.columns)


def test_every_split_and_the_holdout_get_a_schema_valid_matrix(run_stage, params):
    frames = {
        "train": processed_frame([comfort("Cruise control")] * 10),
        "validation": processed_frame([comfort()] * 4),
        "calibration": processed_frame([comfort()] * 3),
        "test": processed_frame([comfort("Cruise control")] * 5),
        "holdout_es": processed_frame([{"country_code": "ES"}] * 2),
    }
    for feature_set in params["features"]["sets"]:
        directory = run_stage(frames, feature_set)
        schema = load_feature_schema(directory, name=f"features-{feature_set}")
        for split in build_features.FEATURE_INPUTS:
            schema.validate(matrix(directory, split))
