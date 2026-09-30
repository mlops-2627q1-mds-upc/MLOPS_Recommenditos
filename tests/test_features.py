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

from loguru import logger
import pandas as pd
import pytest
import yaml

from recommenditos.data import build_features
from recommenditos.pipeline import write_frame
from recommenditos.schema import (
    PROCESSED_SCHEMA,
    TARGET_NAMES,
    FeatureSpaceError,
    SchemaError,
    load_feature_schema,
)

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

    def run(
        frames: dict,
        feature_set: str = "extended",
        supported_makes: "list[str] | None" = None,
        **feature_overrides,
    ) -> Path:
        processed = tmp_path / "processed"
        for name in build_features.FEATURE_INPUTS:
            frame = frames.get(name, frames[build_features.VOCABULARY_SPLIT])
            write_frame(frame, processed / f"{name}.parquet", PROCESSED_SCHEMA)
        # `split` writes this beside the frames, and the stage declares it as a
        # dependency, so the fixture has to produce it too. In the real artefact
        # each entry carries the count that admitted it, so the fixture writes
        # that shape rather than bare names: a fixture that invents a simpler one
        # is how a reader of the real file gets dicts where it expected strings.
        # The default names every make the training frame holds, which is what
        # `split` produces now that it computes the list (#35).
        named = supported_makes
        if named is None:
            training = frames[build_features.VOCABULARY_SPLIT]
            named = sorted(set(training["make"].dropna()))
        (processed / build_features.SUPPORTED_MAKES_FILE).write_text(
            json.dumps(
                {
                    "min_listings_per_make": 1,
                    "counted_over_rows": len(frames[build_features.VOCABULARY_SPLIT]),
                    "supported_makes": [{"make": make, "listings": 1} for make in named],
                }
            )
            + "\n",
            encoding="utf-8",
        )

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
    # '93,000 kg' is the heaviest value in the scoped snapshot, so a pattern that
    # only admitted four digits would reject real data.
    parsed = build_features.weight_kg(
        pd.Series(["1,945 kg", "894 kg", "93,000 kg", None], dtype="str")
    )

    assert parsed.tolist()[:3] == [1945.0, 894.0, 93000.0]
    assert str(parsed.dtype) == "float64"


@pytest.mark.parametrize("value", ["194,5 kg", "1.945 kg", "1,2,3 kg", "1,9450 kg", "1 945 kg"])
def test_a_weight_in_another_locale_fails_rather_than_becoming_a_plausible_number(value):
    # The likely upstream change on a multilingual German site is a locale flip,
    # not a nonsense value. A pattern that took any mix of digits and commas
    # would read '194,5 kg' as 1,945 kg and '1.945 kg' as 1.945 kg: a tenfold and
    # a thousandfold error, both inside or below the plausible range, so no range
    # check would flag them and no fill-rate expectation would either. Failing the
    # stage is the only outcome a reader can tell apart from a correct parse,
    # which is what the comment on the pattern promises.
    with pytest.raises(ValueError, match="not a weight in kg"):
        build_features.weight_kg(pd.Series([value], dtype="str"))


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


def test_the_equipment_threshold_is_a_share_of_every_row_not_of_the_rows_with_a_list(run_stage):
    # 3 of 10 rows name the item and only 4 rows have a list at all, so the
    # denominator decides: 3/10 is below 0.35 and 3/4 is well above it. Counting
    # only the rows that list something would let a near-constant column in and
    # would make the threshold mean something different per equipment column.
    frame = processed_frame(
        [comfort("Cruise control")] * 3 + [comfort("Tow bar")] + [{"equipment_comfort": None}] * 6
    )

    below = build_features.load_vocabulary(
        run_stage({"train": frame}, equipment_min_frequency=0.35)
    )
    above = build_features.load_vocabulary(
        run_stage({"train": frame}, equipment_min_frequency=0.25)
    )

    assert below.equipment["equipment_comfort"] == ()
    assert above.equipment["equipment_comfort"] == ("Cruise control",)


def test_an_item_listed_twice_in_one_listing_counts_as_one_listing(run_stage):
    # The threshold is a share of listings, not of mentions. Three rows that each
    # name the item twice must not clear a threshold that six mentions would, and
    # the encoded column must still be True once rather than counting.
    frame = processed_frame([comfort("Cruise control", "Cruise control")] * 3 + [comfort()] * 7)

    by_listing = build_features.load_vocabulary(
        run_stage({"train": frame}, equipment_min_frequency=0.35)
    )
    written = matrix(run_stage({"train": frame}, equipment_min_frequency=0.2))

    assert by_listing.equipment["equipment_comfort"] == ()
    assert written["equipment_comfort_cruise_control"].tolist() == [True] * 3 + [False] * 7


def test_the_equipment_items_are_alphabetical_so_two_runs_agree_byte_for_byte(run_stage):
    # Ordering by frequency would make the artefact, and the order of the
    # multi-hot columns, depend on counts whose ties break arbitrarily, so the
    # same data could produce two different files and two different matrices
    # (NFR-06). 'Cruise control' is the most frequent of the three here, so
    # frequency order and alphabetical order disagree.
    frame = processed_frame(
        [comfort("Tow bar", "Cruise control", "Alloy wheels")] * 6
        + [comfort("Cruise control")] * 4
    )
    directory = run_stage({"train": frame}, equipment_min_frequency=0.2)
    items = build_features.load_vocabulary(directory).equipment["equipment_comfort"]

    assert items == ("Alloy wheels", "Cruise control", "Tow bar")
    assert [
        column for column in matrix(directory).columns if column.startswith("equipment_comfort_")
    ] == [
        "equipment_comfort_alloy_wheels",
        "equipment_comfort_cruise_control",
        "equipment_comfort_tow_bar",
    ]


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


#: One value per column the stage passes through, so a test can build a frame
#: where every such column has both a value and a hole. Kept as a constant
#: because the imputation test's whole point is that it covers all of them.
_FILLED_ROW = {
    "mileage_km_raw": 120_000.0,
    "nr_prev_owners": 1.0,
    "power_kw": 110.0,
    "gears": 6.0,
    "cylinders_volume_cc": 1_995.0,
    "cylinders": 4.0,
    "nr_seats": 5.0,
    "nr_doors": 4.0,
    "weight_kg": "1,945 kg",
    "electric_range_km": 0.0,
    "registration_date": "2020-01-01",
}


def test_no_pass_through_column_of_the_matrix_is_imputed(run_stage, params):
    # Every column the stage passes through keeps its holes: a value that was
    # missing on the way in is missing on the way out. A filled column would be
    # this stage deciding what a missing gearbox count means, which is the
    # model's job (EDN-15), and the no-imputation rule is the headline of this
    # ticket.
    #
    # Asserted over every pass-through column rather than a chosen few. The
    # columns most at risk are exactly the ones a plausible default exists for -
    # `power_kw`, `mileage_km_raw` and `nr_seats` - and a hand-picked list
    # covering four of about thirty columns let all three through.
    frame = processed_frame(
        [_FILLED_ROW, {}, _FILLED_ROW | {"gears": None}, {}, _FILLED_ROW, {"upholstery": None}]
    )
    written = matrix(run_stage({"train": frame}))

    passed_through = set(written.columns) & set(frame.columns)
    # Pinned, so a change that stops passing a column through has to say so here
    # rather than quietly shrinking what this test covers.
    assert passed_through == (
        set(params["features"]["sets"]["extended"])
        - set(params["features"]["equipment_columns"])
        - {"age_years"}
    ) | set(TARGET_NAMES)

    for column in sorted(passed_through):
        was_missing = frame[column].isna()
        assert not (was_missing & written[column].notna()).any(), column


def test_a_pass_through_column_keeps_its_null_mask_exactly(run_stage):
    # The other direction, for the columns nothing may add a hole to either: a
    # number is parsed or passed through, never dropped. The level rules are
    # allowed to add holes, which is why they are not in this list.
    frame = processed_frame(
        [
            {"gears": 6.0, "nr_prev_owners": 1.0, "weight_kg": "1,945 kg"},
            {},
            {"gears": 7.0},
            {},
            {"nr_prev_owners": 2.0},
        ]
    )
    written = matrix(run_stage({"train": frame}))

    for column in ("gears", "nr_prev_owners", "weight_kg"):
        assert written[column].isna().tolist() == frame[column].isna().tolist(), column


def test_a_matrix_reads_left_to_right_as_its_feature_set_then_the_target(run_stage, params):
    # `_selection` promises "features then target" and nothing held it: one test
    # compares the splits with each other and another compares sets, so putting
    # the target first or reversing the feature order changed nothing visible.
    # The order matters because a model artefact lists the features by name and
    # position, and because two runs must produce byte-identical matrices.
    columns = params["features"]["sets"]["extended"]
    directory = run_stage({"train": processed_frame([comfort("Cruise control")] * 10)})
    vocabulary = build_features.load_vocabulary(directory)

    expected = (
        [name for name in columns if name not in vocabulary.equipment]
        + [
            build_features.equipment_feature_name(source, item)
            for source, items in vocabulary.equipment.items()
            for item in items
        ]
        + list(TARGET_NAMES)
    )
    assert list(matrix(directory).columns) == expected


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


#: Four makes, so a level set with an order worth getting wrong.
_MAKES = ("Audi", "BMW", "Porsche", "Volvo")


@pytest.fixture
def levelled(run_stage):
    """A run whose `make` column has four levels, and its loaded contract."""
    frame = processed_frame([{"make": make} for make in _MAKES] * 3)
    directory = run_stage({"train": frame})
    return directory, load_feature_schema(directory, name="features-extended")


def test_the_written_contract_carries_every_categorical_s_levels(levelled):
    # Without them the artefact is lossy for the only data-dependent dtype the
    # stage produces: `str(dtype)` is 'category' whatever the levels are, so a
    # consumer holding the contract would have to infer them from its own rows.
    directory, schema = levelled
    vocabulary = build_features.load_vocabulary(directory)

    categoricals = {
        column.name: column.levels for column in schema.columns if column.dtype == "category"
    }
    assert categoricals == vocabulary.categories
    assert categoricals["make"] == _MAKES


def test_a_matrix_whose_levels_are_in_the_wrong_order_is_refused(levelled):
    # The codes are level positions, so reversing the levels changes every code
    # in the column. `str(dtype)` does not change, which is why the contract has
    # to compare the level list and not only the dtype name.
    directory, schema = levelled
    written = matrix(directory)
    reordered = written.copy()
    reordered["make"] = written["make"].cat.set_categories(list(reversed(_MAKES)))

    assert reordered["make"].cat.codes.tolist() != written["make"].cat.codes.tolist()
    with pytest.raises(SchemaError, match="different order"):
        schema.validate(reordered)


def test_conforming_a_subset_uses_the_declared_levels_not_the_ones_it_holds(levelled):
    # What #37's ridge variant does: take a slice of the matrix and encode the
    # categoricals itself. Inferring the levels from the slice would give it a
    # numbering the model never saw, with no error anywhere.
    directory, schema = levelled
    written = matrix(directory)
    subset = written.head(2).copy()
    subset["make"] = subset["make"].astype("str")

    conformed = schema.conform(subset)

    assert tuple(conformed["make"].cat.categories) == _MAKES
    assert conformed["make"].cat.codes.tolist() == written["make"].cat.codes.tolist()[:2]


def test_a_one_row_serving_frame_gets_the_training_code(levelled):
    # What the API does: one listing, no price, cast against the loaded
    # contract. Inferring the levels would give every request the code 0.
    directory, schema = levelled
    request = matrix(directory).head(1).drop(columns=list(TARGET_NAMES))
    request["make"] = pd.Series(["Volvo"], index=request.index, dtype="str")

    served = schema.features().conform(request)

    assert tuple(served["make"].cat.categories) == _MAKES
    assert served["make"].cat.codes.tolist() == [_MAKES.index("Volvo")]
    assert not set(TARGET_NAMES) & set(served.columns)


def test_the_feature_space_is_one_public_way_to_load_and_build(levelled, params):
    # The entry point #37, #39 and the API use, so none of them re-derives a
    # level set or reimplements the cast.
    directory, schema = levelled
    space = build_features.FeatureSpace.load(directory, name="features-extended")

    assert space.schema == schema
    assert space.vocabulary == build_features.load_vocabulary(directory)

    rebuilt = space.matrix(
        processed_frame([{"make": "Volvo"}]), reference_date=params["reference_date"]
    )
    assert list(rebuilt.columns) == list(schema.names)
    assert rebuilt["make"].cat.codes.tolist() == [_MAKES.index("Volvo")]


def test_a_stale_feature_space_artefact_names_the_file_and_the_stage(levelled):
    # It is a DVC output, so a hand-edited or half-written one is a real state,
    # and the message has to say which file and which stage rather than raising a
    # bare KeyError inside whichever stage read it next.
    directory, _ = levelled
    (directory / build_features.FEATURE_SPACE_FILE).write_text('{"schema": []}', encoding="utf-8")

    for load in (
        lambda: load_feature_schema(directory, name="features-extended"),
        lambda: build_features.load_vocabulary(directory),
        lambda: build_features.FeatureSpace.load(directory, name="features-extended"),
    ):
        with pytest.raises(FeatureSpaceError) as raised:
            load()
        assert build_features.FEATURE_SPACE_FILE in str(raised.value)
        assert "dvc repro features" in str(raised.value)


def test_a_vocabulary_with_an_unknown_field_is_refused_rather_than_read_in_part(levelled):
    # The same posture the contract takes: both halves come out of one artefact,
    # and a field this code does not know means the file was written by a version
    # that recorded something more, so reading only the known fields would drop
    # part of the feature space.
    directory, _ = levelled
    space = json.loads((directory / build_features.FEATURE_SPACE_FILE).read_text(encoding="utf-8"))
    space["vocabulary"]["hashing_seed"] = 7
    (directory / build_features.FEATURE_SPACE_FILE).write_text(json.dumps(space), encoding="utf-8")

    with pytest.raises(FeatureSpaceError, match="hashing_seed"):
        build_features.load_vocabulary(directory)


def test_the_supported_make_list_is_read_as_names_and_not_as_its_entries(run_stage, tmp_path):
    # This replaces a test of the empty list `split` wrote while it was a stub.
    # `split` now computes a real list and refuses to write an empty one (#35), so
    # that case cannot arise from the only producer; what can still go wrong is
    # the shape. Each entry is `{"make": ..., "listings": ...}`, and reading the
    # entries instead of the names satisfies the `len()` this stage needs today
    # while silently matching nothing the first time a caller restricts on it -
    # which is what EDN-48 leaves #36, #37 and #39 to do.
    run_stage(
        {"train": processed_frame([{"make": "BMW"}] * 4 + [{"make": "Audi"}] * 4)},
        supported_makes=["Audi", "BMW"],
    )

    supported = build_features.read_supported_makes(tmp_path / "processed")

    assert supported == ("Audi", "BMW")
    assert all(isinstance(make, str) for make in supported)


def test_an_empty_supported_make_list_is_refused_rather_than_read_as_no_make(run_stage, tmp_path):
    # An empty list would restrict the model to no make at all. `split` raises
    # before writing one, so an empty list here means the file did not come from
    # `split`, and reading it as "not computed yet" would hide that.
    run_stage({"train": processed_frame([{"make": "BMW"}] * 4)})
    (tmp_path / "processed" / build_features.SUPPORTED_MAKES_FILE).write_text(
        json.dumps({"min_listings_per_make": 300, "counted_over_rows": 0, "supported_makes": []})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FeatureSpaceError, match="names no supported make"):
        build_features.read_supported_makes(tmp_path / "processed")


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


def test_the_holdout_country_is_missing_throughout_its_matrix_on_purpose(run_stage, params):
    # The `ES` holdout is the new-market drift scenario (EDN-03): it is held out
    # by construction, so `ES` is not a training level, and a value outside the
    # training levels becomes missing rather than a level of its own - which is
    # exactly what EDN-18 requires of an unseen value. So the holdout's
    # `country_code` is empty in every row, and that is the scenario working
    # rather than a defect.
    #
    # Pinned because the alternative readings are both plausible and both wrong:
    # somebody debugging it reads an empty column as a bug in this stage, and
    # somebody "fixing" it makes the holdout country a level, which would hand
    # the model a code it was never fitted on. A pull-request description does
    # not survive the squash merge; this does.
    holdout_country = params["split"]["holdout_country"]
    frames = {
        "train": processed_frame([{"country_code": "DE"}, {"country_code": "IT"}] * 5),
        "holdout_es": processed_frame([{"country_code": holdout_country}] * 4),
    }
    directory = run_stage(frames)
    written = matrix(directory, "holdout_es")

    assert written["country_code"].isna().all()
    assert holdout_country not in written["country_code"].cat.categories
    assert list(written["country_code"].cat.categories) == ["DE", "IT"]
    # And it is the only such column, so the warning names one thing rather than
    # being noise somebody learns to ignore.
    training = matrix(directory)
    assert build_features.unobserved_columns(written) - build_features.unobserved_columns(
        training
    ) == {"country_code"}


def test_a_column_only_one_split_leaves_empty_is_named_in_a_warning(run_stage, params):
    # Because nothing else says so. The stage cannot tell the intended case from
    # a defect, so it names the column and says which is which.
    holdout_country = params["split"]["holdout_country"]
    frames = {
        "train": processed_frame([{"country_code": "DE"}, {"country_code": "IT"}] * 5),
        "holdout_es": processed_frame([{"country_code": holdout_country}] * 4),
    }
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        run_stage(frames)
    finally:
        logger.remove(handler)

    unobserved = [message for message in messages if "no value at all" in message]
    assert len(unobserved) == 1, messages
    assert "holdout_es" in unobserved[0]
    assert "country_code" in unobserved[0]
    assert "EDN-03" in unobserved[0]


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
