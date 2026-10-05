"""The four estimators, the bundle they write, and the seam that serves them.

The fixtures below run `preprocess`, `split` and `features` on the synthetic
fixture and then train all four variants once, into `tmp_path_factory`. Nothing
here reaches `data/`, the DagsHub server or the network: a variant that could not
be trained on a clean clone with no credentials would be a variant CI cannot
check, so the whole module trains with the MLflow variables removed and the few
tests that need a server point it at a SQLite database under `tmp_path`.

The fixtures are session-scoped and the models are fitted once. Measured on this
fixture at `num_threads: 1`: the four fits together cost well under a second, so
a per-test refit would be affordable - but the trained models are also the thing
under test, and one set of them shared by every assertion is what makes a
failure about the model rather than about which copy of it ran.

The numbers this module asserts on are shapes and invariants, never results. The
generator derives price from a formula, so the MdAPE of a variant here says
nothing about the real snapshot; what it does say is that the estimator fits,
persists, reloads and predicts the same thing twice.

Issue #39 owns everything below the `--- the SC gate (#39)` marker at the end and
builds on these fixtures rather than refitting.
"""

from fractions import Fraction
import json
from pathlib import Path
import shutil
from statistics import median

import lightgbm
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import pytest
from tests.conftest import PII_COLUMNS, params_override
import yaml

from recommenditos.config import PARAMS_FILE, PROJ_ROOT
from recommenditos.data import build_features, preprocess, split_data
from recommenditos.data.build_features import FeatureSpace, Vocabulary, read_supported_makes
from recommenditos.modeling import train
from recommenditos.modeling.evaluate import (
    ALL_OPTIONAL_FIELDS,
    CRITERIA,
    EXCLUDED_ABSENT_REQUIRED_FIELD,
    EXCLUDED_TARGET_CONDITIONED,
    EXCLUDED_TOO_FEW_ROWS,
    INTERVAL_METHOD,
    INVALID_SEGMENT_LEVEL,
    MISSING_SEGMENT_LEVEL,
    P1_SCENARIO,
    TARGET_CONDITIONED_SEGMENTS,
    GateInput,
    bucket_labels,
    bucket_levels,
    check_make_levels_are_served,
    criterion_segments,
    evaluate_gate,
    evaluate_sc04,
    evaluate_sc05,
    evaluate_sc06,
    gate_blocked_by,
    gate_passed,
    gate_summary,
    input_field_columns,
    interval_metrics,
    masking_sweep,
    optional_input_fields,
    point_metrics,
    sc05_scenarios,
    sc06_scenarios,
    segment_level_column,
    segment_levels,
    segment_rows,
)
from recommenditos.modeling.evaluate import main as evaluate_main
from recommenditos.modeling.model import (
    ESTIMATORS,
    MISSING_LEVEL,
    MODEL_FILE,
    PREDICTION_NAME,
    MedianBaselineModel,
    Model,
    ModelError,
    TrainingData,
    fit_variant,
    load_model,
)
from recommenditos.pipeline import load_params
from recommenditos.schema import (
    FEATURE_SPACE_FILE,
    TARGET_NAMES,
    Schema,
    SchemaError,
    feature_schema,
)
from recommenditos.tracking import (
    REQUIRE_TRACKING_ENV_VAR,
    REQUIRED_ENV_VARS,
    optional_run,
    resume_run,
)

#: A URI that resolves, is syntactically an MLflow tracking server and answers
#: nothing. Port 1 is privileged and closed, so the connection is refused rather
#: than left hanging, and the retry budget `recommenditos.tracking` pins bounds
#: what a refusal costs.
UNREACHABLE_TRACKING_URI = "http://127.0.0.1:1/"


# --------------------------------------------------------------------------
# The fixture pipeline, and the four trained variants
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def matrices(tmp_path_factory, _generated_frame: pd.DataFrame) -> dict:
    """The feature matrices and the make list, built from the synthetic fixture.

    `validate-data` is skipped: it asserts the data's own expectations and writes
    a summary nothing here reads, and `tests/test_pipeline.py` already runs it in
    the chain.
    """
    root = tmp_path_factory.mktemp("model")
    raw = root / "raw" / "listings.parquet"
    raw.parent.mkdir(parents=True)
    _generated_frame.to_parquet(raw, index=False)

    processed = root / "processed"
    features = processed / "features"
    preprocess.main(raw, root / "interim" / "listings.parquet", PARAMS_FILE)
    split_data.main(root / "interim" / "listings.parquet", processed, PARAMS_FILE)
    for feature_set in _feature_sets():
        build_features.main(feature_set, processed, features, PARAMS_FILE)
    return {"root": root, "processed": processed, "features": features}


@pytest.fixture(scope="session")
def supported_makes(matrices: dict) -> tuple[str, ...]:
    return read_supported_makes(matrices["processed"])


@pytest.fixture(scope="session")
def trained(matrices: dict) -> dict:
    """Every variant of the real params.yaml, trained once, with tracking off.

    The environment is cleared inside the fixture rather than by an autouse
    monkeypatch, because a session fixture outlives a function-scoped one and
    would otherwise inherit whatever `.env` the developer's machine has.
    """
    models = matrices["root"] / "models"
    with pytest.MonkeyPatch.context() as environment:
        for name in REQUIRED_ENV_VARS:
            environment.delenv(name, raising=False)
        for variant in _variants():
            train.main(variant, matrices["features"], models, PARAMS_FILE)
    return {"dir": models, "models": {name: load_model(models / name) for name in _variants()}}


@pytest.fixture(scope="session")
def test_frames(matrices: dict) -> dict:
    """The test matrix per feature set, as `features` wrote it.

    Already the population `evaluate` reports on, because the stage restricts
    every frame to the makes the API serves (EDN-48, EDN-67), so a test that
    scores a model here scores it on the same rows the stage would.
    """
    return {
        feature_set: pd.read_parquet(matrices["features"] / feature_set / "test.parquet")
        for feature_set in _feature_sets()
    }


@pytest.fixture(scope="session")
def spaces(matrices: dict) -> dict:
    return {
        feature_set: FeatureSpace.load(
            matrices["features"] / feature_set, name=f"features-{feature_set}"
        )
        for feature_set in _feature_sets()
    }


def _project_params() -> dict:
    return load_params(PARAMS_FILE)


def _variants() -> dict:
    return _project_params()["train"]["variants"]


def _feature_sets() -> list[str]:
    return list(_project_params()["features"]["sets"])


#: Every variant, so a parametrised test says which one failed in its own name.
VARIANTS = tuple(_variants())


# --------------------------------------------------------------------------
# What a trained variant writes
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-06")
def test_every_variant_trains_on_the_fixture_without_credentials(trained: dict):
    """A variant CI cannot train is a variant nothing checks before a delivery."""
    payloads = {
        "median_baseline": "lookup.parquet",
        "ridge": "pipeline.joblib",
        "lightgbm": "booster.txt",
    }
    for variant, model in trained["models"].items():
        written = {path.name for path in (trained["dir"] / variant).iterdir()}
        assert written == {MODEL_FILE, FEATURE_SPACE_FILE, payloads[model.estimator]}, variant


@pytest.mark.req("NFR-08")
def test_model_json_declares_the_seam(trained: dict, spaces: dict):
    """The record is what `evaluate` and the API read, so its shape is a contract.

    The `features` list in particular: the matrix carries the label beside the
    inputs, so a consumer that read one list of columns would fit the target on
    itself, and a bundle that leaked a personal detail into that list would put
    it into the API's own request schema (NFR-08).
    """
    for variant in trained["models"]:
        record = json.loads((trained["dir"] / variant / MODEL_FILE).read_text(encoding="utf-8"))
        assert set(record) >= {
            "variant",
            "estimator",
            "feature_set",
            "features",
            "targets",
            "params",
            "seed",
            "num_threads",
            "training",
            "versions",
            "mlflow",
            "trained_at",
        }, variant
        schema = spaces[record["feature_set"]].schema
        assert record["features"] == list(schema.feature_names), variant
        assert record["targets"] == list(TARGET_NAMES), variant
        assert set(record["features"]).isdisjoint(TARGET_NAMES), variant
        assert set(record["features"]).isdisjoint(PII_COLUMNS), variant
        assert set(record["training"]) >= {
            "n_train_rows",
            "n_validation_rows",
            "price_min_eur",
            "price_max_eur",
            "train_l1_log_price",
            "validation_l1_log_price",
            "supported_makes",
            "best_iteration",
            "boosting_rounds",
            "early_stopped",
        }, variant


def test_the_bundle_carries_the_feature_space_it_was_fitted_against(trained: dict, spaces: dict):
    """The copy in the bundle has to be the artefact, not a re-derivation of it.

    A code is a level's position, so a model encoding a request against one level
    order and scoring it against another is wrong with no error anywhere. The
    bundle travels to the API on its own (EDN-08 bakes it into the image), which
    is why the contract goes with it rather than being looked up beside the
    matrices it will not have.
    """
    for variant, model in trained["models"].items():
        written = FeatureSpace.load(trained["dir"] / variant, name=f"features-{model.feature_set}")
        assert written.schema.columns == spaces[model.feature_set].schema.columns, variant
        assert written.vocabulary == spaces[model.feature_set].vocabulary, variant


def test_the_model_is_fitted_on_the_supported_makes_only(
    trained: dict, supported_makes: tuple[str, ...], matrices: dict
):
    """EDN-48: the model's training rows are the population the API serves.

    Checked rather than intended, which is the fourth of the obligations
    docs/docs/pipeline.md lists for the make list. Asserted on the row count as
    well as on the list, because recording the right makes while fitting on every
    row would pass a check that only read the metadata.

    Counted on the frames `split` wrote rather than on the matrices, because
    `features` is where the restriction happens now (EDN-67): a matrix holds only
    supported makes, so it cannot show that anything was removed, while the split
    frames still hold every make and say what the model should have been fitted
    on.
    """
    expected = {}
    for split in ("train", "validation"):
        frame = pd.read_parquet(matrices["processed"] / f"{split}.parquet", columns=["make"])
        expected[split] = int(frame["make"].isin(supported_makes).sum())
        assert expected[split] < len(frame), (
            f"the fixture's {split} split has to contain an unsupported make, or this test "
            f"cannot fail"
        )
    for variant, model in trained["models"].items():
        assert tuple(model.metadata["training"]["supported_makes"]) == supported_makes, variant
        assert model.metadata["training"]["n_train_rows"] == expected["train"], variant
        # Validation as well as train, and not as a formality: early stopping
        # watches this split and `validation_l1_log_price` is the only metric the
        # stage logs, so a validation set holding makes the API refuses would stop
        # the fit on one population and report a number about another.
        assert model.metadata["training"]["n_validation_rows"] == expected["validation"], variant
        # And the encoding the model was fitted in knows no other make, which is
        # the half of EDN-48 a row count cannot see (obligation 2).
        assert set(model.space.vocabulary.categories["make"]) <= set(supported_makes), variant


def test_the_baseline_lookup_is_a_table_a_person_can_read(trained: dict):
    """B0 is the one estimator whose whole model is inspectable, so keep it so."""
    lookup = pd.read_parquet(trained["dir"] / "b0" / "lookup.parquet")
    assert list(lookup.columns) == ["level", "make", "model", "age_bucket", "price_eur", "n_rows"]
    assert set(lookup["level"]) == {"bucket", "make", "global"}
    assert (lookup["price_eur"] > 0).all()
    assert int((lookup["level"] == "global").sum()) == 1


# --------------------------------------------------------------------------
# The predict seam
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_eur_contract(variant: str, trained: dict, test_frames: dict):
    """Everything `evaluate` and the API are allowed to rely on, in one place."""
    model = trained["models"][variant]
    frame = test_frames[model.feature_set]
    before = frame.copy(deep=True)

    predicted = model.predict_eur(frame)

    assert predicted.name == PREDICTION_NAME
    assert str(predicted.dtype) == "float64"
    assert len(predicted) == len(frame)
    assert predicted.index.equals(frame.index)
    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()
    training = model.metadata["training"]
    assert (predicted >= training["price_min_eur"]).all()
    assert (predicted <= training["price_max_eur"]).all()
    # Pure: #39 predicts up to 25 times off one frame, so a mutation would make
    # every call after the first measure something else.
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("variant", VARIANTS)
def test_one_row_predicts_exactly_what_the_batch_predicts_for_it(
    variant: str, trained: dict, test_frames: dict
):
    """The single property the API is built on, and the one nothing else pins.

    A code is a level's position, so the route that turns a frame into codes has
    to read the level set off the contract and never off the frame it was handed.
    Re-inferring it - `astype("object").astype("category").cat.codes`, which is
    exactly what the class docstring says the explicit route exists to
    prevent - numbers a one-row request by that row's own single value, so every
    categorical becomes 0 and the model scores a different car. Measured on the
    real `lgbm-basic`: 6.83 % to 20.46 % MdAPE over the test frame, and one
    request of 131,543 EUR answered as 63,597 EUR, a 51.7 % error, with no
    failure anywhere.

    A relative tolerance rather than bit-identity, because bit-identity is not
    true of `b1`: `Ridge.predict` is a sparse matrix-vector product, and scipy
    accumulates a one-row matrix in a different order than an n-row one, which was
    measured here at 8 ULP - 1.8e-15 relative - on 1 of 25 sampled rows. The
    tolerance is three orders of magnitude tighter than that noise and eleven
    orders looser than the error this test exists to catch, so it separates the
    two without pretending the arithmetic is exact.
    """
    model = trained["models"][variant]
    frame = test_frames[model.feature_set]
    batch = model.predict_eur(frame)
    # Spread over the frame rather than the head, so a categorical that happens to
    # be constant in the first rows cannot hide the case.
    positions = range(0, len(frame), max(1, len(frame) // 25))

    for position in positions:
        single = model.predict_eur(frame.iloc[[position]])

        assert single.index.equals(frame.index[[position]]), position
        np.testing.assert_allclose(
            single.to_numpy(),
            batch.to_numpy()[[position]],
            rtol=1e-12,
            err_msg=f"row {position}",
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_eur_answers_an_empty_frame_with_an_empty_series(
    variant: str, trained: dict, test_frames: dict
):
    """ "The same length and the same index as `frame`" has to hold at n = 0 too.

    Three of the four estimators refuse an empty matrix in their own words -
    scikit-learn with "Found array with 0 sample(s)", LightGBM with "Input data
    must be 2 dimensional and non empty" - so without the short circuit `b0`
    answers and the other three raise a library's error for something that is not
    an error. SC-06's masking sweep can produce an empty segment, and a batch
    endpoint can be handed an empty list.
    """
    model = trained["models"][variant]
    empty = test_frames[model.feature_set].head(0)

    predicted = model.predict_eur(empty)

    assert len(predicted) == 0
    assert predicted.index.equals(empty.index)
    assert str(predicted.dtype) == "float64"
    assert predicted.name == PREDICTION_NAME


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_eur_still_names_a_missing_column_on_an_empty_frame(
    variant: str, trained: dict, test_frames: dict
):
    """The empty-frame short circuit must not become a way past the contract."""
    model = trained["models"][variant]

    with pytest.raises(SchemaError, match="make"):
        model.predict_eur(test_frames[model.feature_set].head(0).drop(columns=["make"]))


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_log_price_is_exactly_the_log_of_predict_eur(
    variant: str, trained: dict, test_frames: dict
):
    """The two must not be able to disagree about the bound or the transform."""
    model = trained["models"][variant]
    frame = test_frames[model.feature_set].head(50)

    np.testing.assert_array_equal(
        model.predict_log_price(frame).to_numpy(), np.log(model.predict_eur(frame).to_numpy())
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_load_model_round_trips_predictions(variant: str, trained: dict, test_frames: dict):
    """A bundle that predicts something else than the fitted model is a wrong bundle.

    Bit-identical, not close: every payload format here is exact, so a tolerance
    would hide a real change of numbering or of iteration count.
    """
    model = trained["models"][variant]
    frame = test_frames[model.feature_set]

    reloaded = load_model(trained["dir"] / variant)

    np.testing.assert_array_equal(
        reloaded.predict_eur(frame).to_numpy(), model.predict_eur(frame).to_numpy()
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_eur_ignores_extra_and_reordered_columns(
    variant: str, trained: dict, test_frames: dict
):
    """The model selects what it needs; the caller does not have to prepare a frame.

    The matrix carries `price` and `log_price`, so a model that took the frame as
    given would be scored on its own target.
    """
    model = trained["models"][variant]
    frame = test_frames[model.feature_set]
    shuffled = frame.sample(frac=1.0, axis="columns", random_state=0)
    assert list(shuffled.columns) != list(frame.columns), "the columns were not reordered"

    np.testing.assert_array_equal(
        model.predict_eur(shuffled).to_numpy(), model.predict_eur(frame).to_numpy()
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_eur_names_a_missing_feature_column(
    variant: str, trained: dict, test_frames: dict
):
    """A column the model needs and the caller forgot has to say which one."""
    model = trained["models"][variant]
    frame = test_frames[model.feature_set]

    with pytest.raises(SchemaError, match="make"):
        model.predict_eur(frame.drop(columns=["make"]))


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_eur_accepts_a_single_row_built_from_a_dict(
    variant: str, trained: dict, spaces: dict, test_frames: dict
):
    """The API's path: one row, python types, only the required fields of FR-01.

    A request arrives as a dict from Pydantic, not as a slice of a Parquet file,
    so its numerics are python `int` and `float` and its absent fields are `None`.
    """
    model = trained["models"][variant]
    row = _api_request(model, spaces[model.feature_set].schema, test_frames[model.feature_set])

    predicted = model.predict_eur(pd.DataFrame([row]))

    assert len(predicted) == 1
    assert np.isfinite(predicted.iloc[0])
    assert predicted.iloc[0] > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_an_unseen_category_is_treated_as_missing_not_an_error(
    variant: str, trained: dict, spaces: dict, test_frames: dict
):
    """EDN-18: a car in scope carrying a detail the model has not seen is answered.

    A value outside the contract's levels has no fitted signal and giving it a
    code of its own would renumber every other level, so it becomes missing -
    which is a prediction, not a refusal.
    """
    model = trained["models"][variant]
    row = _api_request(model, spaces[model.feature_set].schema, test_frames[model.feature_set])
    row["model"] = "no such model has ever been built"
    row["country_code"] = "ZZ"

    predicted = model.predict_eur(pd.DataFrame([row]))

    assert np.isfinite(predicted.iloc[0])
    assert predicted.iloc[0] > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_eur_does_not_raise_with_every_optional_field_absent(
    variant: str, trained: dict, test_frames: dict
):
    """FR-01's promise, at the model layer: leaving an optional field out is allowed.

    Masked through `Model.mask_absent`, which is the seam's own answer to what an
    omitted field is, so this asserts the promise about the masking the API and
    SC-06 will use rather than about one a test invented. Over the whole test
    frame rather than one row, because a single row cannot show a masked column
    destroying the values in the rows beside it.
    """
    model = trained["models"][variant]
    required = set(_project_params()["evaluate"]["required_input_fields"]) | {"age_years"}
    masked = model.mask_absent(
        test_frames[model.feature_set],
        [name for name in model.features if name not in required],
    )

    predicted = model.predict_eur(masked)

    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_mask_absent_is_the_one_answer_to_what_an_omitted_field_is(
    trained: dict, test_frames: dict
):
    """Four dtypes, three different values, and the wrong one is silent.

    The helper exists because `frame[column] = np.nan`, the one-liner anyone
    reaches for, is wrong twice and loud once. The three assertion flags are
    non-nullable, so a float64 NaN fails the contract's null check and the caller
    finds out. An equipment column is nullable, so the same assignment upcasts to
    float64, survives `conform` and comes back as `pd.NA` - the field arrives as
    absent where EDN-61 says it is empty, and nothing says a word about it. No
    metric would either: on the real snapshot the two rules give **bit-identical**
    predictions on all 19,665 test rows, for each of the four equipment fields and
    for all 23 optional fields at once. That equivalence is a property of the tree
    models rather than of this method, which is why the rule is fixed here on what
    the states mean instead of on a measurement that cannot separate them.
    """
    model = trained["models"]["lgbm-extended"]
    frame = test_frames[model.feature_set]
    before = frame.copy(deep=True)
    equipment = model.columns_of("equipment_comfort")
    assert len(equipment) > 1, "the field has to expand, or this test proves nothing"
    assert frame[list(equipment)].notna().all().all(), (
        "the training domain has no null equipment column, so neither may the fixture"
    )

    masked = model.mask_absent(
        frame, ["equipment_comfort", "has_full_service_history", "body_color", "weight_kg"]
    )

    # An omitted list is an empty list: every item `False`, and never null, which
    # is a state no training row of either scale contains.
    assert str(masked[equipment[0]].dtype) == "boolean"
    assert masked[list(equipment)].notna().all().all()
    assert not masked[list(equipment)].any().any()
    # An unticked checkbox is `False` too, which is all EDN-23 leaves it able to
    # mean, and the contract refuses a null there.
    assert masked["has_full_service_history"].notna().all()
    assert not masked["has_full_service_history"].any()
    # Null over the contract's own levels, so `_align` gives the missing code
    # rather than a level of its own.
    assert masked["body_color"].isna().all()
    assert tuple(masked["body_color"].cat.categories) == model.input_schema.levels["body_color"]
    assert masked["weight_kg"].isna().all()
    # A masking, not a wiped frame, and not a mutation of the caller's frame.
    assert masked["make"].equals(frame["make"])
    pd.testing.assert_frame_equal(frame, before)

    predicted = model.predict_eur(masked)
    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


def test_masking_a_field_the_model_does_not_consume_is_refused(trained: dict):
    """A sweep over the wrong field list would otherwise report a masked metric.

    SC-06's ratio is against the full-input MdAPE, so a field that masked nothing
    reports a ratio of exactly 1.0 and reads as a criterion that passed.

    The third case is the seam's boundary rather than a typo: `registration_date`
    is a real request field, and the matrix carries the derived `age_years`
    instead. `columns_of` knows the feature space and not the request, so it
    refuses - a caller holding a request field applies `evaluate`'s rename first
    (#39), and the API does the same at M4.
    """
    model = trained["models"]["b1"]

    with pytest.raises(ModelError, match="equipment_comfort"):
        # A real field of the extended set, and not a column `basic` carries.
        model.columns_of("equipment_comfort")
    with pytest.raises(ModelError, match="no_such_field"):
        model.mask_absent(pd.DataFrame(), ["no_such_field"])
    with pytest.raises(ModelError, match="registration_date"):
        model.columns_of("registration_date")
    assert "age_years" in model.features, "which is what the matrix carries instead"


def test_align_returns_only_float64_and_category_columns(trained: dict, test_frames: dict):
    """`_align`'s output dtypes are the contract `_predict_log_price` is written against.

    An estimator subclass receives this frame and nothing else, so "every column is
    float64 or category" is what it may assume - and the `bool` and nullable
    `boolean` columns of the contract are the only ones that would otherwise break
    it.

    Worth being exact about what this does and does not pin, because the
    difference is the whole reason the assertion is on the dtype. Removing the
    cast changes **no prediction anywhere** today: measured on the real snapshot,
    all four variants predict bit-identically without it, a Ridge fitted on
    `extended` fits and predicts bit-identically without it, and so does
    `lgbm-extended` on a frame with `pd.NA` actually present in an equipment
    column. LightGBM accepts either dtype, and scikit-learn's `MissingIndicator`
    refuses a `bool` column only when it is handed one on its own - inside the
    `ColumnTransformer` it receives a mixed frame, which `check_array` has already
    made float64.

    So this is a contract assertion, not a regression test for a bug anyone has
    seen, and the cast is insurance of the same kind as `deterministic=True` and
    `solver="lsqr"` (EDN-53). It is stated rather than removed because the
    property a new estimator depends on should not rest on two libraries
    independently choosing to be tolerant.
    """
    model = trained["models"]["lgbm-extended"]
    equipment = model.columns_of("equipment_comfort")
    flags = [name for name in model.features if model.input_schema.column(name).dtype == "bool"]
    assert flags, "the extended set has to carry non-nullable boolean columns"
    assert equipment, "and the multi-hot nullable ones"

    aligned = model._align(test_frames[model.feature_set])

    assert {str(aligned[name].dtype) for name in aligned.columns} <= {"float64", "category"}
    assert {str(aligned[name].dtype) for name in (*equipment, *flags)} == {"float64"}


def test_align_passes_missing_values_through_unimputed(
    trained: dict, spaces: dict, test_frames: dict
):
    """EDN-15's rule, asserted where it is observable.

    On the internal deliberately: the generator puts no price signal in the
    optional columns, so no prediction on this fixture can distinguish a model
    that imputes from one that does not. What can be checked is that the frame
    the estimator receives still says "absent": NaN in a masked number and code
    -1 in a masked categorical, which is LightGBM's missing value.
    """
    model = trained["models"]["lgbm-extended"]
    masked = model.mask_absent(test_frames[model.feature_set], ["weight_kg", "body_color"])

    aligned = model._align(masked)

    assert aligned["weight_kg"].isna().all()
    assert (aligned["body_color"].cat.codes == -1).all()
    # The columns that were not masked keep their values, so the masking is a
    # masking rather than a wiped frame.
    assert aligned["age_years"].notna().any()


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_null_assertion_flag_is_refused_rather_than_becoming_a_third_state(
    variant: str, trained: dict, test_frames: dict
):
    """EDN-23 rejected a true/false/unknown encoding, so absence must not sneak in.

    A checkbox left empty is `False` - the absence of evidence - and the contract
    makes the three flags non-nullable to say so. A caller that sends `None`
    anyway is asking for the encoding EDN-23 refused, and gets a failure naming
    the column instead of a quiet third state.
    """
    model = trained["models"][variant]
    if "has_full_service_history" not in model.features:
        pytest.skip(f"{model.feature_set!r} does not carry the assertion flags")
    frame = test_frames[model.feature_set].copy()
    frame["has_full_service_history"] = pd.array([pd.NA] * len(frame), dtype="boolean")

    with pytest.raises(SchemaError, match="has_full_service_history"):
        model.predict_eur(frame)


def test_a_prediction_that_would_overflow_is_bounded_to_the_training_range(
    trained: dict, test_frames: dict
):
    """The bound is what makes "finite and strictly positive" true of the arithmetic.

    `exp` overflows to `inf` above about 710 in log space and underflows to
    exactly 0.0 below -746, and an `inf` prediction would turn MdAPE into `nan`
    and pass the gate's comparison silently. The estimator is replaced rather
    than coaxed into extrapolating, because the bound is a property of the base
    class and should be tested without depending on how far a particular fit can
    be pushed.
    """
    model = trained["models"]["b1"]
    frame = test_frames[model.feature_set].head(10)
    training = model.metadata["training"]

    for absurd, expected in ((1e4, training["price_max_eur"]), (-1e4, training["price_min_eur"])):
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(
                model,
                "_predict_log_price",
                lambda aligned, value=absurd: np.full(len(aligned), value),
            )
            predicted = model.predict_eur(frame)

        assert np.isfinite(predicted).all()
        assert (predicted > 0).all()
        np.testing.assert_allclose(predicted.to_numpy(), expected)


# --------------------------------------------------------------------------
# B0, whose arithmetic can be checked by hand
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hand_built_baseline(spaces: dict, test_frames: dict) -> MedianBaselineModel:
    """A six-row B0 whose three levels have hand-computed medians.

    | make | model | age  | price  |
    |------|-------|------|--------|
    | A    | M     | 0.0  | 10,000 |
    | A    | M     | 1.5  | 20,000 |
    | A    | M     | 2.0  | 30,000 |
    | A    | N     | 3.0  | 50,000 |
    | B    | M     | 0.5  |  7,000 |
    | B    | M     | null |  9,000 |

    With `age_bucket_years: 2`, that is bucket medians A/M/0 = 15,000,
    A/M/1 = 30,000, A/N/1 = 50,000 and B/M/0 = 7,000; make medians A = 25,000 and
    B = 8,000; and a global median of 15,000. The last row keys no bucket, which
    is how it ends up in B's make median and nowhere finer.
    """
    space = spaces["basic"]
    makes = sorted(test_frames["basic"]["make"].dropna().unique())[:2]
    models = sorted(test_frames["basic"]["model"].dropna().unique())[:2]
    rows = [
        (makes[0], models[0], 0.0, 10_000.0),
        (makes[0], models[0], 1.5, 20_000.0),
        (makes[0], models[0], 2.0, 30_000.0),
        (makes[0], models[1], 3.0, 50_000.0),
        (makes[1], models[0], 0.5, 7_000.0),
        (makes[1], models[0], None, 9_000.0),
    ]
    frame = _frame_of(space.schema, test_frames["basic"], rows)
    data = TrainingData(space, frame, frame, tuple(makes))
    settings = {
        "estimator": "median_baseline",
        "feature_set": "basic",
        "params": {"age_bucket_years": 2},
    }
    return fit_variant("b0-by-hand", settings, data, seed=1, num_threads=1)


def test_b0_falls_back_from_bucket_to_make_to_global(
    hand_built_baseline: MedianBaselineModel, spaces: dict, test_frames: dict
):
    """Each level of the chain fires, and each fires with the median it should."""
    model = hand_built_baseline
    makes = sorted(test_frames["basic"]["make"].dropna().unique())[:2]
    models = sorted(test_frames["basic"]["model"].dropna().unique())[:2]
    probes = [
        # the bucket level
        (makes[0], models[0], 1.0, 15_000.0),
        (makes[0], models[0], 2.5, 30_000.0),
        (makes[0], models[1], 3.9, 50_000.0),
        # no such bucket, so the make median
        (makes[0], models[0], 9.0, 25_000.0),
        (makes[1], models[1], 0.0, 8_000.0),
        # no age, so no bucket key at all
        (makes[0], models[0], None, 25_000.0),
        # the same, for the make whose training rows include one without an age.
        # That row must not have formed a bucket of its own keyed on the missing
        # value: pandas matches a null merge key to a null merge key, so a
        # `dropna=False` grouping would answer 9,000 here instead of B's median.
        (makes[1], models[0], None, 8_000.0),
        # no make, so nothing above the global median
        (None, models[0], 1.0, 15_000.0),
    ]
    frame = _frame_of(
        spaces["basic"].schema, test_frames["basic"], [(*probe[:3], 1.0) for probe in probes]
    )

    predicted = model.predict_eur(frame)

    np.testing.assert_allclose(predicted.to_numpy(), [probe[3] for probe in probes])


def test_b0_bucket_boundaries_are_left_closed(
    hand_built_baseline: MedianBaselineModel, spaces: dict, test_frames: dict
):
    """A band holds its lower edge and not its upper one, or two bands overlap.

    Pinned against the hand-computed model so the assertion is a price a reader
    can check, not a bucket index nothing else reads.
    """
    makes = sorted(test_frames["basic"]["make"].dropna().unique())[:2]
    models = sorted(test_frames["basic"]["model"].dropna().unique())[:2]
    frame = _frame_of(
        spaces["basic"].schema,
        test_frames["basic"],
        [(makes[0], models[0], age, 1.0) for age in (0.0, 1.99, 2.0, 3.99)],
    )

    predicted = hand_built_baseline.predict_eur(frame)

    np.testing.assert_allclose(predicted.to_numpy(), [15_000.0, 15_000.0, 30_000.0, 30_000.0])


# --------------------------------------------------------------------------
# Reproducibility, and where the thread count comes from
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-06")
@pytest.mark.parametrize("variant", VARIANTS)
def test_training_is_reproducible(
    variant: str,
    trained: dict,
    matrices: dict,
    supported_makes: tuple[str, ...],
    test_frames: dict,
):
    """The same rows and the same parameters give the same model, twice.

    Bit-identical predictions, and for the payload formats that are the fit
    itself - LightGBM's text export, joblib's pickle, B0's Parquet table - a
    byte-identical file. Stronger than NFR-06's "within 0.1 percentage points",
    which is the promise that survives a different toolchain; this is the one
    that holds inside one.
    """
    settings = _variants()[variant]
    data = train.read_matrices(matrices["features"], settings["feature_set"], supported_makes)
    seed = _project_params()["seed"]

    again = fit_variant(variant, settings, data, seed=seed, num_threads=1)
    frame = test_frames[settings["feature_set"]]

    np.testing.assert_array_equal(
        again.predict_eur(frame).to_numpy(),
        trained["models"][variant].predict_eur(frame).to_numpy(),
    )


@pytest.mark.req("NFR-06")
def test_the_payload_of_a_refit_is_byte_identical(
    tmp_path: Path, matrices: dict, supported_makes: tuple[str, ...]
):
    """Two fits of the same variant write the same bytes, not merely the same numbers.

    `model.json` is excluded and deliberately so: it carries `trained_at` and the
    MLflow run id, which are provenance and cannot be stable. The estimator's own
    payload is what has to be, and it is the whole of the fitted model.
    """
    for variant, settings in _variants().items():
        data = train.read_matrices(matrices["features"], settings["feature_set"], supported_makes)
        digests = []
        for attempt in ("first", "second"):
            model = fit_variant(variant, settings, data, seed=17, num_threads=1)
            directory = tmp_path / variant / attempt
            model.save(directory)
            payload = next(path for path in sorted(directory.iterdir()) if path.name != MODEL_FILE)
            digests.append((payload.name, payload.read_bytes()))
        assert digests[0] == digests[1], variant


def test_a_budget_early_stopping_never_reaches_is_recorded_as_a_budget_that_bound(
    tmp_path: Path, matrices: dict, params: dict, monkeypatch
):
    """A fit the `n_estimators` budget ended says so, in the record and in the log.

    `best_iteration` alone cannot say it. LightGBM 4.7 reports the best iteration
    inside the budget whether or not early stopping fired, so a fit cut off while
    still improving records the budget itself - and one whose curve wobbled near
    the end records a few trees less, which reads exactly like early stopping.
    That is what happened on the real snapshot at the former budget of 1,000
    trees: `lgbm-extended` recorded 996 and had never stopped (issue #64). The
    fixture always early-stops, so the budget is lowered here to reach the branch.
    """
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    variants = {
        **params["train"]["variants"],
        "lgbm-basic": {
            **params["train"]["variants"]["lgbm-basic"],
            "params": {
                **params["train"]["variants"]["lgbm-basic"]["params"],
                "n_estimators": 3,
            },
        },
    }
    overridden = params_override(tmp_path, params, train={**params["train"], "variants": variants})

    reported: list[str] = []
    sink = logger.add(reported.append, format="{message}", level="WARNING")
    try:
        train.main("lgbm-basic", matrices["features"], tmp_path / "models", overridden)
    finally:
        logger.remove(sink)

    record = json.loads(
        (tmp_path / "models" / "lgbm-basic" / MODEL_FILE).read_text(encoding="utf-8")
    )
    assert record["training"]["best_iteration"] == 3
    assert record["training"]["boosting_rounds"] == 3
    assert record["training"]["early_stopped"] is False
    booster = lightgbm.Booster(model_file=str(tmp_path / "models" / "lgbm-basic" / "booster.txt"))
    assert booster.num_trees() == 3
    assert any("n_estimators" in message and "lgbm-basic" in message for message in reported), (
        reported
    )


def test_a_lightgbm_record_says_early_stopping_ended_the_fit(trained: dict):
    """The property issue #64 is about, recorded where every consumer can read it.

    Early stopping fired when the fit ran exactly `early_stopping_rounds` past its
    best iteration and stopped short of the budget. The record carries both the
    trees kept and the rounds run, so whether the budget bound a fit is a fact of
    the bundle rather than something read back out of a training log. Asserted on
    the fixture, which early-stops; the real snapshot's answer is in the committed
    `model.json` and the MLflow run.
    """
    for variant in ("lgbm-basic", "lgbm-extended"):
        training = trained["models"][variant].metadata["training"]
        settings = _variants()[variant]["params"]
        assert training["early_stopped"] is True, variant
        assert (
            training["boosting_rounds"]
            == training["best_iteration"] + settings["early_stopping_rounds"]
        ), variant
        assert training["boosting_rounds"] < settings["n_estimators"], variant
    for variant in ("b0", "b1"):
        training = trained["models"][variant].metadata["training"]
        assert training["early_stopped"] is None, variant
        assert training["boosting_rounds"] is None, variant


def test_num_threads_is_pinned_from_params(
    tmp_path: Path, matrices: dict, params: dict, monkeypatch
):
    """The thread count in the artefact comes from params.yaml, not from the machine.

    LightGBM writes it into `booster.txt`, so left at the default of "every core"
    the file - and therefore `dvc.lock` - would depend on who ran the pipeline
    (NFR-06). Trained at a value the project does not use, so that a hard-coded 1
    would fail this.
    """
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    overridden = params_override(tmp_path, params, train={**params["train"], "num_threads": 3})

    train.main("lgbm-basic", matrices["features"], tmp_path / "models", overridden)

    booster = (tmp_path / "models" / "lgbm-basic" / "booster.txt").read_text(encoding="utf-8")
    assert "[num_threads: 3]" in booster
    record = json.loads(
        (tmp_path / "models" / "lgbm-basic" / MODEL_FILE).read_text(encoding="utf-8")
    )
    assert record["num_threads"] == 3


def test_lightgbm_splits_a_categorical_as_levels_and_never_as_a_number(trained: dict):
    """`categorical_feature` is the whole of EDN-02's rationale, and it is invisible.

    `_as_codes` hands LightGBM int32 columns, so a fit that was not told which of
    them are categorical succeeds and splits `make` at `code <= 12.5` - which
    orders the makes by their position in an alphabetically sorted level list and
    calls that a feature. Measured on the real snapshot: 6.83 % to 7.05 % MdAPE
    for `lgbm-basic`, with no failure and no warning.

    Asserted on the saved booster rather than on the fitted wrapper, because the
    booster is what the API loads, and in both directions: every categorical
    column that is split at all is split with `==`, and no categorical column is
    ever split with `<=`.
    """
    for variant in ("lgbm-basic", "lgbm-extended"):
        model = trained["models"][variant]
        categorical = set(model.categorical_columns)
        dumped = model.booster.dump_model()
        names = dumped["feature_names"]
        split_by = {"==": set(), "<=": set()}
        for tree in dumped["tree_info"]:
            nodes = [tree["tree_structure"]]
            while nodes:
                node = nodes.pop()
                if "decision_type" in node:
                    split_by[node["decision_type"]].add(names[node["split_feature"]])
                nodes.extend(node[side] for side in ("left_child", "right_child") if side in node)

        assert split_by["=="], f"{variant} has no categorical split at all"
        assert split_by["=="] <= categorical, variant
        assert split_by["<="].isdisjoint(categorical), variant


def test_the_booster_is_saved_at_its_early_stopped_iteration(trained: dict):
    """The file is the model, so predict needs no iteration argument.

    A consumer that had to remember `num_iteration=best_iteration` would sooner
    or later forget, and get a quietly overfitted prediction.

    Worth saying what this does and does not pin: removing the explicit
    `num_iteration` from the save does not make it fail, because lightgbm 4.7
    already truncates the booster when early stopping fires. It pins the
    property, so it would catch a LightGBM that stopped truncating or a
    configuration in which early stopping never fires - which the second
    assertion is there to rule out for the fixture as it stands. It reads the
    record rather than comparing `best_iteration` with the budget, because a best
    iteration below the budget does not mean early stopping fired.
    """
    for variant in ("lgbm-basic", "lgbm-extended"):
        training = trained["models"][variant].metadata["training"]
        booster = lightgbm.Booster(model_file=str(trained["dir"] / variant / "booster.txt"))
        assert booster.num_trees() == training["best_iteration"], variant
        assert training["early_stopped"], (
            f"{variant} never early-stopped, so this test proves nothing"
        )


# --------------------------------------------------------------------------
# Loading, and failing loudly
# --------------------------------------------------------------------------


def test_an_unknown_estimator_fails_loudly(matrices: dict, supported_makes: tuple[str, ...]):
    """A typo in params.yaml must name itself and what was expected instead."""
    data = train.read_matrices(matrices["features"], "basic", supported_makes)
    settings = {"estimator": "nope", "feature_set": "basic", "params": {}}

    with pytest.raises(ModelError, match="nope") as failure:
        fit_variant("broken", settings, data, seed=1, num_threads=1)

    for known in ESTIMATORS:
        assert known in str(failure.value)


@pytest.mark.parametrize(
    ("params_block", "expected"),
    [
        ({"alpha": 1.0, "min_category_rows": 30, "extra": 1}, "extra"),
        ({"alpha": 1.0}, "min_category_rows"),
    ],
)
def test_a_params_block_the_estimator_does_not_match_fails_loudly(
    params_block: dict, expected: str, matrices: dict, supported_makes: tuple[str, ...]
):
    """Both directions, because both are silent otherwise.

    A missing key would surface as a `KeyError` from inside a fit, and an extra
    one as a parameter somebody set, `dvc repro` retrained for, and nothing read.
    """
    data = train.read_matrices(matrices["features"], "basic", supported_makes)
    settings = {"estimator": "ridge", "feature_set": "basic", "params": params_block}

    with pytest.raises(ModelError, match=expected):
        fit_variant("b1", settings, data, seed=1, num_threads=1)


def test_load_model_refuses_a_directory_that_is_not_a_bundle(tmp_path: Path, trained: dict):
    """Each way a bundle can be unusable says which file is the problem."""
    with pytest.raises(ModelError, match=MODEL_FILE):
        load_model(tmp_path / "nothing-here")

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MODEL_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelError, match="not readable JSON"):
        load_model(broken)

    shutil.copytree(trained["dir"] / "b0", tmp_path / "renamed")
    record = json.loads((tmp_path / "renamed" / MODEL_FILE).read_text(encoding="utf-8"))
    record["estimator"] = "gone"
    (tmp_path / "renamed" / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ModelError, match="gone"):
        load_model(tmp_path / "renamed")


@pytest.mark.parametrize("damage", ["truncated", "reordered"])
def test_a_bundle_whose_two_halves_disagree_is_refused(damage: str, tmp_path: Path, trained: dict):
    """A record and a feature space from different runs would predict silently wrong.

    The two files are written together and restored together, so this needs a
    hand-edited or half-restored bundle - which is exactly the case where nothing
    else would notice.

    Both damages, because they are caught by different comparisons. A shorter list
    changes the count, so a check that only compared counts would pass this test
    while letting the worse case through: the same columns in another order, which
    is the symptom the guard names - encoding a request by one level order and
    scoring it by another.
    """
    bundle = tmp_path / f"mismatched-{damage}"
    shutil.copytree(trained["dir"] / "lgbm-basic", bundle)
    record = json.loads((bundle / MODEL_FILE).read_text(encoding="utf-8"))
    features = list(record["features"])
    if damage == "truncated":
        record["features"] = features[:-1]
    else:
        record["features"] = [features[1], features[0], *features[2:]]
        assert len(record["features"]) == len(features), "a reorder keeps the count"
        assert set(record["features"]) == set(features), "a reorder keeps the membership"
    (bundle / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ModelError, match="disagree"):
        load_model(bundle)


def test_the_ridge_emits_a_missing_indicator_for_every_numeric_feature(trained: dict):
    """EDN-51's guard, which no prediction on this fixture can show.

    scikit-learn's default is an indicator only for the features that were
    missing *at fit time*. A column that happens to be complete in the training
    split would then be mean-filled with no indicator the moment SC-06 masks it,
    and the criterion would be measuring the model's own imputation rather than
    the cost of the missing field. `features="all"` is what prevents that, and it
    is invisible from the outside, so it is asserted on the fitted pipeline.

    This assertion does not distinguish `features="all"` from the default on either
    the fixture or the real snapshot, because no numeric column of `basic` is
    complete at fit time in either; the test below is the one that does, and says
    what it constructs.
    """
    model = trained["models"]["b1"]
    encoded = list(model.pipeline["encode"].get_feature_names_out())

    # By the transformer's prefix, not by the word "missing": the categorical
    # branch has a `__missing__` level of its own and would match that too.
    indicators = {
        name.removeprefix("numeric__missing__missingindicator_")
        for name in encoded
        if name.startswith("numeric__missing__")
    }
    assert indicators == set(model.numeric_columns)
    # And the fill is there too, so the indicator is an addition rather than a
    # replacement: dropping the filled copy would leave Ridge only the pattern.
    filled = {
        name.removeprefix("numeric__filled__")
        for name in encoded
        if name.startswith("numeric__filled__")
    }
    assert filled == set(model.numeric_columns)


def test_the_ridge_carries_a_named_level_for_an_absent_categorical(trained: dict):
    """B1 exists to be read, so the coefficient for absence needs a readable name.

    `OneHotEncoder` would call it `nan`, which in a table of coefficients reads
    like a defect rather than like "this listing did not say".
    """
    model = trained["models"]["b1"]
    encoded = set(model.pipeline["encode"].get_feature_names_out())

    named = {name for name in encoded if name.endswith(MISSING_LEVEL)}
    assert named, f"no level named {MISSING_LEVEL!r} among {sorted(encoded)[:8]}"
    assert not any(name.endswith("_nan") for name in encoded)


def test_a_numeric_feature_complete_in_training_still_gets_an_indicator(
    matrices: dict, supported_makes: tuple[str, ...]
):
    """The case `features="all"` exists for, which neither fixture nor snapshot has.

    Every numeric column of the synthetic fixture happens to have a missing value
    in the training split, so the default `features="missing-only"` emits the same
    set of indicators there and the test above cannot tell the two apart. So does
    every one of the 8 numeric features of `basic`, the only set `b1` uses, on the
    real snapshot - the emptiest is `age_years`, with 1 gap in 60,378 rows. The
    construction below is therefore neither configuration, and deliberately: what
    it pins is that the guard works, not that it currently binds.

    It is worth pinning because the configuration it protects is one change away.
    136 of the `extended` set's 147 numeric features are complete at fit time, so a
    Ridge fitted on `extended` would lose 136 indicators to the default, and SC-06
    would measure the model's own imputation for each of them rather than the cost
    of the missing field.
    """
    data = train.read_matrices(matrices["features"], "basic", supported_makes)
    complete = data.train.copy()
    complete["power_kw"] = complete["power_kw"].fillna(complete["power_kw"].median())
    assert complete["power_kw"].notna().all(), "the column has to be complete at fit time"

    model = fit_variant(
        "b1",
        _variants()["b1"],
        TrainingData(data.space, complete, data.validation, supported_makes),
        seed=1,
        num_threads=1,
    )

    assert "numeric__missing__missingindicator_power_kw" in set(
        model.pipeline["encode"].get_feature_names_out()
    )


def test_a_refit_does_not_leave_the_previous_estimator_beside_the_new_one(
    matrices: dict, tmp_path: Path, monkeypatch
):
    """A variant switched from one estimator to another must not keep both payloads.

    `dvc repro` clears a stage's outputs first, but `train.main` is also run by
    hand and by the tests, and a directory holding both a `booster.txt` and a
    `pipeline.joblib` would hash differently depending on which fits a machine
    happened to have run (NFR-06).
    """
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    models = tmp_path / "models"

    train.main("lgbm-basic", matrices["features"], models, PARAMS_FILE)
    # The same output directory, under a variant that writes a different payload.
    shutil.move(models / "lgbm-basic", models / "b1")
    train.main("b1", matrices["features"], models, PARAMS_FILE)

    assert {path.name for path in (models / "b1").iterdir()} == {
        MODEL_FILE,
        FEATURE_SPACE_FILE,
        "pipeline.joblib",
    }


def test_a_bundle_that_cannot_be_read_back_fails_the_training_run(
    matrices: dict, tmp_path: Path, monkeypatch
):
    """A successful fit is not a successful stage: `train` ends by loading its own work.

    What `evaluate` and the API use is what came off the disk, so a payload that
    is not there - or not readable - is a failed training run even though the fit
    succeeded. Without the read-back the stage is green, DVC records the output,
    and the failure surfaces in `evaluate` where nothing points at the stage that
    caused it.

    The payload is simply not written, rather than corrupted, so the failure is
    one exception with one cause instead of whichever error a particular Parquet
    version happens to raise for damaged bytes.
    """
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(MedianBaselineModel, "_save_payload", lambda self, directory: None)

    with pytest.raises(FileNotFoundError, match="lookup.parquet"):
        train.main("b0", matrices["features"], tmp_path / "models", PARAMS_FILE)

    # The fit itself did finish and the record is on disk, so the read-back is
    # what failed rather than the training.
    assert (tmp_path / "models" / "b0" / MODEL_FILE).exists()


def test_a_payload_written_by_another_library_version_is_flagged(tmp_path: Path, trained: dict):
    """A pickle read by a different scikit-learn changes numbers, not behaviour.

    A warning rather than a failure, because a minor upgrade usually loads fine
    and refusing would stop the pipeline for a routine bump. Silence would be
    worse than either: the symptom is a metric that moved for no reason anyone
    can see.
    """
    bundle = tmp_path / "from-the-future"
    shutil.copytree(trained["dir"] / "b1", bundle)
    record = json.loads((bundle / MODEL_FILE).read_text(encoding="utf-8"))
    record["versions"]["scikit-learn"] = "0.0.1"
    (bundle / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")

    reported: list[str] = []
    sink = logger.add(reported.append, format="{message}", level="WARNING")
    try:
        load_model(bundle)
    finally:
        logger.remove(sink)

    assert any("scikit-learn 0.0.1" in message for message in reported), reported


def test_a_training_price_range_that_cannot_bound_a_price_is_refused(
    tmp_path: Path, trained: dict
):
    """`predict_eur` promises a positive price, and the bound is how it keeps it.

    A lower edge of 0 would clip to `log(0)`, so `exp` would return exactly 0.0
    and the promise would be silently false. `preprocess.price_min_eur: 500` makes
    it unreachable from the pipeline; this is the guard for a bundle somebody
    edited.
    """
    bundle = tmp_path / "unbounded"
    shutil.copytree(trained["dir"] / "b0", bundle)
    record = json.loads((bundle / MODEL_FILE).read_text(encoding="utf-8"))
    record["training"]["price_min_eur"] = 0.0
    (bundle / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ModelError, match="cannot bound a price"):
        load_model(bundle)


def test_a_matrix_column_no_estimator_can_encode_is_refused():
    """A feature set may name a column the catalogue has and no estimator can use.

    `registration_date` is such a column: it is in the catalogue, `age_years` is
    what the models actually consume, and casting a timestamp to float64 would
    produce a plausible-looking feature out of nanoseconds since 1970. Refused at
    construction, naming the column, rather than silently cast.
    """
    schema = feature_schema(["make", "registration_date"], name="odd").as_category(
        "make", ("A", "B"), name="odd"
    )
    space = FeatureSpace(schema=schema, vocabulary=Vocabulary({}, {"make": ("A", "B")}, 1, 10))
    metadata = {
        "variant": "odd",
        "estimator": "median_baseline",
        "feature_set": "odd",
        "features": list(schema.feature_names),
        "training": {"price_min_eur": 500.0, "price_max_eur": 2_000_000.0},
    }

    with pytest.raises(ModelError, match="registration_date"):
        Model(metadata, space)


def test_training_refuses_matrices_built_against_another_make_list(
    matrices: dict, tmp_path: Path, params: dict, monkeypatch
):
    """A make list from another `split` run fails the fit rather than being reapplied.

    The way it happens: `split.min_listings_per_make` is swept, the list is
    rewritten and the matrices are not rebuilt. `features` owns the restriction
    (EDN-67), so `train` does not filter again - that would quietly repair the
    rows while keeping a vocabulary decided over a different population - but it
    checks that every row it is about to fit is a make the list names, and says
    which list it checked against.
    """
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    shutil.copytree(matrices["processed"], tmp_path / "processed")
    makes = tmp_path / "processed" / "supported_makes.json"
    record = json.loads(makes.read_text(encoding="utf-8"))
    record["supported_makes"] = [{"make": "No Such Make", "listings": 9_999}]
    makes.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="No Such Make"):
        train.main("b0", tmp_path / "processed" / "features", tmp_path / "models", PARAMS_FILE)


def test_require_tracking_refuses_a_model_with_no_run_to_resume(monkeypatch):
    """`evaluate` must not quietly drop the metrics of a run it cannot find.

    With tracking mandatory, a model whose `run_id` is null was trained without
    tracking, so appending its metrics to "the same run" is impossible and saying
    so is the only honest outcome.
    """
    monkeypatch.setenv(REQUIRE_TRACKING_ENV_VAR, "1")

    with pytest.raises(RuntimeError, match=REQUIRE_TRACKING_ENV_VAR), resume_run(None):
        pass


def test_every_variant_names_a_known_estimator_and_an_existing_feature_set(params: dict):
    """Read the real params.yaml, so a typo there fails here rather than in a run."""
    for variant, settings in params["train"]["variants"].items():
        assert settings["estimator"] in ESTIMATORS, variant
        assert settings["feature_set"] in params["features"]["sets"], variant
        assert set(settings["params"]) == set(ESTIMATORS[settings["estimator"]].hyperparameters), (
            variant
        )


def test_the_interval_method_of_sc05_is_reserved_and_absent(trained: dict):
    """No variant may grow an interval method before SC-05 can really be measured.

    `evaluate` hardcodes SC-05 as null today, so this pins nothing the metrics
    artefact currently says. Issue #39 replaces that hardcoded null with a
    capability check on this attribute, and a stub that raised would then make
    the criterion read as measurable - which is the one thing NFR-01 cannot
    tolerate. Pinned before the check exists, because once it does the stub is
    the cheap way to make a null go away.
    """
    for variant, model in trained["models"].items():
        assert not hasattr(model, "predict_interval_eur"), variant


# --------------------------------------------------------------------------
# MLflow: one run per variant, and training without a server
# --------------------------------------------------------------------------


def test_tracking_is_disabled_without_a_tracking_uri(trained: dict, tmp_path: Path):
    """No credentials means no tracking, never a store in the repository.

    Since MLflow 3.16 an unset tracking URI resolves to
    `sqlite:///$PWD/mlflow.db`, so the wrong answer here is not a crash but a
    database appearing in the repository root. `tests/conftest.py`'s session
    guard is the second net; this is the one that says which stage did it.
    """
    for variant, model in trained["models"].items():
        assert model.metadata["mlflow"]["tracking_mode"] == "disabled", variant
        assert model.metadata["mlflow"]["run_id"] is None, variant
    assert not list(trained["dir"].glob("**/mlflow.db"))
    assert not list(trained["dir"].glob("**/mlruns"))


def test_a_tracking_failure_does_not_fail_training(
    matrices: dict, tmp_path: Path, monkeypatch, caplog
):
    """A server that is down must not stop a model being trained.

    The whole reason tracking degrades rather than raising: CI has no
    credentials, and a training run that a temporary DagsHub outage can destroy
    is one nobody can rely on.
    """
    _point_at_an_unreachable_server(monkeypatch)

    train.main("b0", matrices["features"], tmp_path / "models", PARAMS_FILE)

    record = json.loads((tmp_path / "models" / "b0" / MODEL_FILE).read_text(encoding="utf-8"))
    assert record["mlflow"]["tracking_mode"] == "disabled"
    assert record["mlflow"]["run_id"] is None


def test_require_tracking_makes_a_failure_fatal(matrices: dict, tmp_path: Path, monkeypatch):
    """The escape hatch for the runs whose numbers are cited in the report.

    Without it, "the runs are on the server" is something a reader has to take on
    trust, and a silent skip is found while the report is being written.
    """
    _point_at_an_unreachable_server(monkeypatch)
    monkeypatch.setenv(REQUIRE_TRACKING_ENV_VAR, "1")

    with pytest.raises(Exception, match="127.0.0.1"):
        train.main("b0", matrices["features"], tmp_path / "models", PARAMS_FILE)


@pytest.mark.req("NFR-14")
def test_one_variant_is_one_run_that_evaluate_appends_to(
    matrices: dict, tmp_path: Path, monkeypatch
):
    """The answer to "each DVC stage becomes its own MLflow run".

    `train` opens the run and records its id; `evaluate` resumes that id instead
    of opening a second one, so the experiment holds one run per variant carrying
    its hyperparameters, its artefact, its energy figures and its verdict. Two
    runs per variant joined by a tag would be the same information in a table
    nobody can read.

    A local SQLite store puts its artifacts in `./mlruns`, relative to the
    working directory, so the test runs from `tmp_path`. On DagsHub the artifact
    store is the server's and nothing lands on disk; the `chdir` is an artefact of
    testing without a server, and `tests/conftest.py`'s guard is what caught its
    absence.
    """
    before = mlflow.get_tracking_uri()
    store = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")
    monkeypatch.chdir(tmp_path)
    # The configuration the report's own runs use, so this test walks the path
    # that must not silently skip.
    monkeypatch.setenv(REQUIRE_TRACKING_ENV_VAR, "1")

    try:
        train.main("b0", matrices["features"], tmp_path / "models", PARAMS_FILE)
        record = json.loads((tmp_path / "models" / "b0" / MODEL_FILE).read_text(encoding="utf-8"))
        assert record["mlflow"]["tracking_mode"] == "enabled"
        run_id = record["mlflow"]["run_id"]
        assert run_id

        client = mlflow.MlflowClient(tracking_uri=store)
        run = client.get_run(run_id)
        assert run.data.params["estimator"] == "median_baseline"
        assert run.data.params["n_supported_makes"]
        assert run.data.tags["variant"] == "b0"
        assert run.data.tags["dvc_stage"] == "train@b0"
        # NFR-14 asks every run to record which code produced it, and the seam
        # does that rather than each caller. Called by a test rather than by DVC,
        # and on matrices under `tmp_path`, the run names no input hashes: the
        # ones `dvc.yaml` declares are not the files this fit read.
        assert run.data.tags["git_commit"]
        assert not [name for name in run.data.tags if ".deps." in name]
        assert "train_l1_log_price" in run.data.metrics
        assert {MODEL_FILE, FEATURE_SPACE_FILE, "lookup.parquet"} <= {
            entry.path.split("/")[-1] for entry in client.list_artifacts(run_id, "model")
        }

        with resume_run(run_id) as resumed:
            assert resumed.run_id == run_id
            resumed.log_metrics({"mdape": 0.25, "sc04_measured": None})

        appended = client.get_run(run_id)
        # One run, both halves: the fit's metric and the evaluation's.
        assert (
            appended.data.metrics["train_l1_log_price"] == run.data.metrics["train_l1_log_price"]
        )
        assert appended.data.metrics["mdape"] == 0.25
        # A criterion nothing could measure is not recorded as a number.
        assert "sc04_measured" not in appended.data.metrics
        experiment = client.get_experiment_by_name(_project_params()["train"]["mlflow_experiment"])
        assert len(client.search_runs([experiment.experiment_id])) == 1
    finally:
        mlflow.set_tracking_uri(before)


#: What a `train` run records besides the keys `dvc.yaml` declares for its stage:
#: which variant it is, and the shape of the data those keys were applied to.
#: Listed so that the comparison below is an equality, and a parameter nobody
#: declared cannot reach the run unnoticed either.
NOT_DECLARED_TRAIN_PARAMS = frozenset(
    {"variant", "n_features", "n_train_rows", "n_validation_rows", "n_supported_makes"}
)


def _declared_train_params(variant: str) -> dict:
    """Every `params.yaml` leaf `dvc.yaml` declares for `train@<variant>`, by dotted key.

    Read from `dvc.yaml` itself rather than restated, because the claim under test
    is about what DVC tracks: a key declared there and missing from the run is a
    parameter that changes the model without the run saying so.
    """
    stage = yaml.safe_load((PROJ_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]["train"]
    params = _project_params()
    leaves: dict = {}

    def flatten(key: str, value) -> None:
        if isinstance(value, dict):
            for child, nested in value.items():
                flatten(f"{key}.{child}", nested)
        else:
            leaves[key] = value

    for declared in stage["do"]["params"]:
        assert isinstance(declared, str), "train declares keys of params.yaml, not other files"
        key = declared.replace("${key}", variant)
        value = params
        for part in key.split("."):
            value = value[part]
        flatten(key, value)
    return leaves


def _run_param_name(key: str, variant: str, estimator: str) -> str:
    """Where `train` logs a declared key, which is the naming its docstring promises.

    Without the path that differs between variants, so the four runs of the ladder
    share their columns in the experiment table, and a hyperparameter under its
    estimator's name, so LightGBM's `num_leaves` and the Ridge's `alpha` never
    share one.
    """
    hyperparameters = f"train.variants.{variant}.params."
    if key.startswith(hyperparameters):
        return f"{estimator}.{key.removeprefix(hyperparameters)}"
    return key.removeprefix(f"train.variants.{variant}.").removeprefix("train.")


@pytest.mark.req("NFR-14")
@pytest.mark.parametrize("variant", VARIANTS)
def test_a_run_records_every_parameter_its_stage_declares(
    variant: str, matrices: dict, tmp_path: Path, monkeypatch
):
    """NFR-14's parameter clause, read back from the run rather than from the code.

    Every key `dvc.yaml` declares for `train@<variant>` is a parameter DVC reruns
    the stage for, so it is one that can change the model, and the run has to say
    what it was. `train.mlflow_experiment` is the one recorded as what it is, the
    run's experiment, rather than repeated as a parameter of every run in it.
    """
    before = mlflow.get_tracking_uri()
    store = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(REQUIRE_TRACKING_ENV_VAR, "1")

    try:
        train.main(variant, matrices["features"], tmp_path / "models", PARAMS_FILE)
        record = json.loads((tmp_path / "models" / variant / MODEL_FILE).read_text("utf-8"))
        client = mlflow.MlflowClient(tracking_uri=store)
        run = client.get_run(record["mlflow"]["run_id"])
        experiment = client.get_experiment(run.info.experiment_id).name
    finally:
        mlflow.set_tracking_uri(before)

    declared = _declared_train_params(variant)
    assert experiment == declared.pop("train.mlflow_experiment")
    estimator = declared[f"train.variants.{variant}.estimator"]
    expected = {
        _run_param_name(key, variant, estimator): str(value) for key, value in declared.items()
    }

    logged = dict(run.data.params)
    assert set(logged) - set(expected) == NOT_DECLARED_TRAIN_PARAMS, (
        "the run carries a parameter that is neither declared nor the data's shape"
    )
    assert {name: logged.get(name) for name in expected} == expected


def test_resume_run_without_a_run_id_does_nothing(caplog):
    """A model trained without tracking must not stop `evaluate` from running."""
    with resume_run(None) as run:
        assert run.run_id is None
        run.log_metrics({"mdape": 0.1})


def test_a_run_that_cannot_be_opened_degrades_without_failing_the_stage(
    tmp_path: Path, monkeypatch
):
    """The seam's reason to exist, asserted on the seam rather than through a stage.

    `tests/test_model.py::test_a_tracking_failure_does_not_fail_training` walks
    the same path through `train.main`; this one pins the handle the caller gets
    and the warning it gets it with, which is what `_degrading` may catch.
    """
    _point_at_an_unreachable_server(monkeypatch)
    reported: list[str] = []
    sink = logger.add(reported.append, format="{message}", level="WARNING")

    try:
        with optional_run("some-experiment", "some-run") as run:
            assert run.mode == "disabled"
            assert run.run_id is None
            # Every call is unconditional at the call site, which is the whole
            # point of the null object.
            run.log_params({"variant": "b0"})
            run.log_metrics({"fit_seconds": 1.0})
            run.set_tags({"dvc_stage": "train@b0"})
    finally:
        logger.remove(sink)

    assert any("MLflow tracking is off" in message for message in reported), reported


@pytest.mark.parametrize("manager", ["optional_run", "resume_run"])
def test_a_failure_in_the_body_reaches_the_caller_unchanged(
    manager: str, tmp_path: Path, monkeypatch
):
    """A failing fit must report itself, not the tracking seam that wrapped it.

    With the `yield` inside the `except`, contextlib throws the caller's own
    exception into the generator at the yield, the handler reads a failing fit as
    a failing server and yields a second time, and the caller gets
    `RuntimeError: generator didn't stop after throw()` with the real exception
    only in `__context__` - as DVC's top-line error, with a log line saying
    tracking was off when it was on.

    Asserted against a *working* store, because that is the configuration this
    happens in: a developer who followed getting-started.md has credentials and no
    `RECOMMENDITOS_REQUIRE_TRACKING`, which is exactly the path that degrades.
    """
    before = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")
    monkeypatch.delenv(REQUIRE_TRACKING_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    reported: list[str] = []
    sink = logger.add(reported.append, format="{message}", level="WARNING")

    try:
        with optional_run("some-experiment", "some-run") as opened:
            assert opened.mode == "enabled", "the store has to work, or this test proves nothing"
            run_id = opened.run_id
        under_test = (
            optional_run("some-experiment", "another-run")
            if manager == "optional_run"
            else resume_run(run_id)
        )

        with pytest.raises(ValueError, match="the fit failed"), under_test:
            raise ValueError("the fit failed")
    finally:
        logger.remove(sink)
        mlflow.set_tracking_uri(before)

    assert reported == [], "a failure in the body is not a tracking failure"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _point_at_an_unreachable_server(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", UNREACHABLE_TRACKING_URI)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")
    # Measured: MLflow's default budget of 5 retries takes 246 s to give up
    # against a closed port, which no test may pay. The production default is
    # pinned at 3 in `recommenditos.tracking`; 0 here keeps the suite fast.
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")


def _api_request(model: Model, schema: Schema, frame: pd.DataFrame) -> dict:
    """One row as the API would hand it over: required fields set, the rest absent.

    Python types rather than the matrix's dtypes, because a request comes from
    JSON, which is why this builds a dict instead of calling `Model.mask_absent`
    on a frame: the values have to survive the trip through `pd.DataFrame`.
    """
    required = set(_project_params()["evaluate"]["required_input_fields"]) | {"age_years"}
    row = {}
    for name in model.features:
        column = schema.column(name)
        if name not in required:
            # An unticked checkbox is `False` and the absence of evidence
            # (EDN-23); an equipment list nobody sent is an empty one, so every
            # item of it is `False` too (EDN-61). Everything else is `None`.
            row[name] = False if column.dtype in {"bool", "boolean"} else None
        elif column.levels is not None:
            row[name] = str(frame[name].dropna().iloc[0])
        else:
            row[name] = float(frame[name].dropna().iloc[0])
    return row


def _frame_of(schema: Schema, template: pd.DataFrame, rows: list[tuple]) -> pd.DataFrame:
    """A contract-valid matrix of `(make, model, age_years, price)` rows.

    Every other column is taken from the template's first row, so the frame is
    one the schema accepts while the four columns a hand-computed baseline reads
    are the ones written here.
    """
    frame = pd.concat([template.head(1)] * len(rows), ignore_index=True)
    frame["make"] = pd.Categorical(
        [row[0] for row in rows], categories=schema.column("make").levels
    )
    frame["model"] = pd.Categorical(
        [row[1] for row in rows], categories=schema.column("model").levels
    )
    frame["age_years"] = [row[2] for row in rows]
    frame["price"] = [float(row[3]) for row in rows]
    frame["log_price"] = np.log(frame["price"])
    return schema.conform(frame)


# --- the SC gate (#39)
#
# Issue #39's half of this module: the metric layer, the per-segment breakdown,
# the masking sweep and the SC-01 to SC-06 gate. It builds on the session
# fixtures above rather than refitting, because a LightGBM refit per test would
# be the slowest thing in the suite, and its own fixtures are `gate_`-prefixed so
# the two halves cannot collide.
#
# Every hand-computed expected value below is checked twice: against the literal
# from the design, and against `_rational_metrics`, an independent implementation
# in exact rational arithmetic with no floating point in it at all. A mistake in
# the implementation would have to be repeated identically in both to pass.


# --------------------------------------------------------------------------
# The metric layer: hand-computed values
# --------------------------------------------------------------------------


def _rational_metrics(actual: "list[float]", predicted: "list[float]") -> dict:
    """The five metrics of problem-spec section 6 in exact rational arithmetic.

    Written from the definitions rather than from `point_metrics`, and using
    `Fraction` so that no result is a rounded one. This is the second opinion
    every case below is measured against.
    """
    errors = [abs(Fraction(p) - Fraction(a)) for a, p in zip(actual, predicted, strict=True)]
    relative = [error / abs(Fraction(a)) for a, error in zip(actual, errors, strict=True)]
    rows = len(relative)
    return {
        "mdape": median(relative),
        "within_10pct": Fraction(sum(1 for each in relative if each <= Fraction(1, 10)), rows),
        "within_20pct": Fraction(sum(1 for each in relative if each <= Fraction(1, 5)), rows),
        "mae_eur": sum(errors, Fraction(0)) / rows,
        "mape": sum(relative, Fraction(0)) / rows,
    }


def _assert_point_metrics(actual: "list[float]", predicted: "list[float]", expected: dict) -> None:
    """`point_metrics` agrees with the hand-computed literals and with exact arithmetic.

    `pytest.approx` rather than `==`, because two of the expected values are not
    representable in binary floating point (case A's MAPE comes out as
    0.11000000000000001 and case B's MdAPE as 0.15000000000000002). It still pins
    what the cases exist to pin: a lower-middle median convention would give 0.10
    for case B, which `approx(0.15)` rejects.
    """
    measured = point_metrics(
        pd.Series(actual, dtype="float64"), pd.Series(predicted, dtype="float64")
    )
    assert measured == pytest.approx(expected)
    assert measured == pytest.approx(
        {name: float(value) for name, value in _rational_metrics(actual, predicted).items()}
    )


def test_point_metrics_over_an_odd_number_of_rows():
    # The general case. Errors 10 %, 10 %, 0 %, 15 %, 20 %.
    _assert_point_metrics(
        [10000, 20000, 30000, 40000, 50000],
        [11000, 18000, 30000, 46000, 40000],
        {
            "mdape": 0.10,
            "within_10pct": 0.6,
            "within_20pct": 1.0,
            "mae_eur": 3800.0,
            "mape": 0.11,
        },
    )


def test_the_median_of_an_even_number_of_rows_is_the_mean_of_the_middle_two():
    # Errors 5 %, 10 %, 20 %, 40 %, so the median is (10 + 20) / 2 and not 10.
    # Pinned because pandas' convention is the one the reported MdAPE uses, and a
    # "fix" to a lower-middle convention would quietly lower every number in the
    # report by half a row's worth of error.
    _assert_point_metrics(
        [10000, 10000, 10000, 10000],
        [10500, 11000, 12000, 14000],
        {
            "mdape": 0.15,
            "within_10pct": 0.5,
            "within_20pct": 0.75,
            "mae_eur": 1875.0,
            "mape": 0.1875,
        },
    )


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        (11000, {"mdape": 0.10, "within_10pct": 1.0, "within_20pct": 1.0, "mape": 0.10}),
        (12000, {"mdape": 0.20, "within_10pct": 0.0, "within_20pct": 1.0, "mape": 0.20}),
    ],
)
def test_a_band_counts_an_error_exactly_on_its_edge(predicted, expected):
    # "Within +/-20 %" includes 20 % itself, which is what SC-02's 85 % is a share
    # of. Both edges, so neither can be made exclusive without failing.
    _assert_point_metrics([10000], [predicted], {**expected, "mae_eur": abs(predicted - 10000.0)})


@pytest.mark.parametrize(
    ("actual", "predicted", "mdape"),
    [(10000, 20000, 1.0), (20000, 10000, 0.5)],
)
def test_the_percentage_error_is_relative_to_the_price_not_the_prediction(
    actual, predicted, mdape
):
    # The same 10,000 EUR miss, two different percentage errors. This is the test
    # that fails if the denominator is ever "improved" into the prediction or into
    # a symmetric mean of the two, which would make the reported MdAPE
    # incomparable with the reference values of problem-spec section 8.
    _assert_point_metrics(
        [actual],
        [predicted],
        {
            "mdape": mdape,
            "within_10pct": 0.0,
            "within_20pct": 0.0,
            "mae_eur": 10000.0,
            "mape": mdape,
        },
    )


def test_a_single_row_is_measured_rather_than_refused():
    # Mathematically fine and statistically meaningless, which is what SC-04's row
    # minimum exists for. The segment table carries `n` on every row so a reader
    # can see which is which.
    _assert_point_metrics(
        [30000],
        [24000],
        {
            "mdape": 0.20,
            "within_10pct": 0.0,
            "within_20pct": 1.0,
            "mae_eur": 6000.0,
            "mape": 0.20,
        },
    )


# --------------------------------------------------------------------------
# The metric layer: the four guards
# --------------------------------------------------------------------------


def test_metrics_over_an_empty_set_are_refused():
    empty = pd.Series([], dtype="float64")
    with pytest.raises(ValueError, match="empty set"):
        point_metrics(empty, empty)


def test_a_price_of_zero_is_refused_and_says_where():
    with pytest.raises(ValueError, match=r"price of 0.*\[1\]"):
        point_metrics(
            pd.Series([10000.0, 0.0], dtype="float64"), pd.Series([9000.0, 10.0], dtype="float64")
        )


def test_a_nan_prediction_raises_rather_than_being_skipped_by_the_median():
    actual = pd.Series([10000.0] * 10)
    predicted = pd.Series([10500.0] * 5 + [float("nan")] * 5)

    # What the unguarded arithmetic does, asserted rather than described in a
    # comment: `median` and `mean` skip the NaN rows while the band share counts
    # them, so a model that failed on half its rows reports an unchanged primary
    # metric beside a halved share. That is the reason the guard exists.
    relative = (predicted - actual).abs() / actual.abs()
    assert float(relative.median()) == 0.05
    assert float((relative <= 0.20).mean()) == 0.5

    with pytest.raises(ValueError, match="5 of 10 prediction"):
        point_metrics(actual, predicted)


def test_a_misaligned_index_raises_rather_than_being_aligned_into_nan():
    actual = pd.Series([10000.0] * 10, index=range(10))
    predicted = pd.Series([10500.0] * 10, index=range(5, 15))

    # Again the unguarded behaviour, asserted: pandas aligns the two into the
    # union of their indices, so the median is taken over the five overlapping
    # rows and the band share divides by fifteen. No error anywhere.
    relative = (predicted - actual).abs() / actual.abs()
    assert len(relative) == 15
    assert float(relative.median()) == 0.05
    assert float((relative <= 0.20).mean()) == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="indexed differently"):
        point_metrics(actual, predicted)


def test_a_missing_actual_price_raises():
    # The same failure on the other side. `point_metrics` is also what the drift
    # job will use, on data no range filter has seen.
    with pytest.raises(ValueError, match="actual price"):
        point_metrics(
            pd.Series([10000.0, float("nan")]), pd.Series([10500.0, 10500.0], dtype="float64")
        )


# --------------------------------------------------------------------------
# Age buckets, derived from the edges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (0.0, "0-1"),
        (0.999, "0-1"),
        (1.0, "1-3"),
        (2.999, "1-3"),
        (3.0, "3-6"),
        (5.999, "3-6"),
        (6.0, "6-10"),
        (9.999, "6-10"),
        (10.0, "10-20"),
        (19.999, "10-20"),
        (20.0, "over 20"),
        (45.0, "over 20"),
        (float("nan"), MISSING_SEGMENT_LEVEL),
        (-0.5, INVALID_SEGMENT_LEVEL),
    ],
)
def test_an_age_bucket_is_left_closed_and_names_both_absences(age, expected, params):
    # The project's own edges, so the test fails if they move without the labels.
    # The last two rows are the levels that keep a data-quality defect visible: a
    # null `registration_date` and a listing registered after the reference date
    # are different findings, and folding a negative age into the youngest bucket
    # would hide the second one (EDN-22 removes it in `preprocess`).
    levels = bucket_levels(
        pd.Series([age], dtype="float64"), params["evaluate"]["age_bucket_edges"]
    )
    assert list(levels) == [expected]


def test_the_bucket_labels_are_generated_from_the_edges():
    # The test that fails if the labels are ever written out as a constant: these
    # edges are not the project's, and the labels have to follow them.
    assert bucket_labels([0, 2, 5]) == ["0-2", "2-5", "over 5"]
    assert list(bucket_levels(pd.Series([0.0, 1.9, 2.0, 4.9, 5.0, 99.0]), [0, 2, 5])) == [
        "0-2",
        "0-2",
        "2-5",
        "2-5",
        "over 5",
        "over 5",
    ]


def test_a_level_label_is_never_a_float_nan_among_strings():
    # Measured on pandas 3: `astype("str")` on a categorical holding a null
    # leaves `['Dealer', 'PrivateSeller', nan]`, which raises `TypeError` as soon
    # as anything sorts or groups the levels. The explicit fill is why the
    # segment table can be sorted at all.
    seller = pd.Series(pd.Categorical(["Dealer", "PrivateSeller", None]))
    with pytest.raises(TypeError, match="not supported between"):
        sorted(seller.astype("str").unique())

    assert sorted(segment_levels(seller).unique()) == [
        MISSING_SEGMENT_LEVEL,
        "Dealer",
        "PrivateSeller",
    ]


# --------------------------------------------------------------------------
# SC-04: which levels count, and what happens when none do
# --------------------------------------------------------------------------

#: Ten rows in two makes: six at a 5 % error and four at 30 %. One frame, three
#: row minimums, and together they prove the criterion discriminates in both
#: directions and blocks when it cannot be measured at all.
_SC04_ACTUAL = [10000.0] * 10
_SC04_PREDICTED = [10500.0] * 6 + [13000.0] * 4
_SC04_MAKES = ["A"] * 6 + ["B"] * 4


@pytest.fixture
def gate_criteria(params) -> dict:
    return dict(params["evaluate"]["success_criteria"])


def _sc04_rows(min_rows: int) -> "list[dict]":
    return segment_rows(
        {"make": pd.Series(_SC04_MAKES, dtype="str")},
        pd.Series(_SC04_ACTUAL, dtype="float64"),
        pd.Series(_SC04_PREDICTED, dtype="float64"),
        variant="hand-built",
        min_rows=min_rows,
    )


@pytest.mark.req("NFR-01")
def test_sc04_fails_on_the_worst_level_that_is_large_enough_to_judge(gate_criteria):
    # Both makes qualify at a minimum of 3, so the criterion is decided by the
    # worse of them: 30 % against the 15 % bound.
    verdict = evaluate_sc04(_sc04_rows(3), gate_criteria, segments=("make",))

    assert verdict["sc04_measured"] == pytest.approx(0.30)
    assert verdict["sc04_passed"] is False
    assert verdict["sc04_n_qualifying"] == 2
    assert verdict["sc04_worst_segment"]["level"] == "B"
    assert verdict["sc04_worst_segment"]["n"] == 4


@pytest.mark.req("NFR-01")
def test_sc04_passes_when_the_only_qualifying_level_is_good_enough(gate_criteria):
    # The same frame at a minimum of 5: make B has four rows and drops out, so the
    # criterion is 5 % and passes. The pair of tests is what shows the verdict
    # follows the measurement rather than the code always saying one thing.
    verdict = evaluate_sc04(_sc04_rows(5), gate_criteria, segments=("make",))

    assert verdict["sc04_measured"] == pytest.approx(0.05)
    assert verdict["sc04_passed"] is True
    assert verdict["sc04_n_qualifying"] == 1


@pytest.mark.req("NFR-01")
def test_sc04_is_not_measured_rather_than_vacuously_met_when_no_level_qualifies(gate_criteria):
    # "Every level satisfies P" is vacuously true over an empty set. Reporting
    # that as a pass would say the model had been checked where it had not been,
    # so the criterion reports `None`, which blocks the gate.
    verdict = evaluate_sc04(_sc04_rows(20), gate_criteria, segments=("make",))

    assert verdict["sc04_measured"] is None
    assert verdict["sc04_passed"] is None
    assert verdict["sc04_n_segment_levels"] == 2
    assert verdict["sc04_n_qualifying"] == 0
    assert "make=A at 6" in verdict["sc04_note"]


@pytest.mark.req("NFR-01")
def test_the_sc04_row_minimum_is_inclusive(gate_criteria):
    # Make B has exactly four rows, so at a minimum of four it counts. `n <
    # min_rows` against `n <= min_rows` is invisible on any other frame, and the
    # difference is whether "at least 500 test rows" means 500 or 501.
    assert _SC04_MAKES.count("B") == 4
    qualifying = {row["level"]: row["counts_toward_sc04"] for row in _sc04_rows(4)}

    assert qualifying == {"A": True, "B": True}


@pytest.mark.req("NFR-01")
@pytest.mark.parametrize(("mdape", "passed"), [(0.1499, True), (0.15, True), (0.1500001, False)])
def test_the_sc04_mdape_bound_is_inclusive(mdape, passed, gate_criteria):
    # A level exactly on the 15 % bound passes. Asserted on a hand-built row so
    # the value is the literal rather than one a division produced.
    rows = [
        {
            "variant": "hand-built",
            "segment": "make",
            "level": "BMW",
            "n": 600,
            "mdape": mdape,
            "counts_toward_sc04": True,
            "excluded_because": "",
        }
    ]

    assert evaluate_sc04(rows, gate_criteria, segments=("make",))["sc04_passed"] is passed


def test_a_level_below_the_minimum_is_reported_with_its_row_count(gate_criteria):
    # Not dropped: a reader of the fairness section has to be able to see that a
    # make had four test rows rather than infer it from an absence.
    rows = {row["level"]: row for row in _sc04_rows(5)}

    assert rows["B"]["n"] == 4
    assert rows["B"]["counts_toward_sc04"] is False
    assert rows["B"]["excluded_because"] == EXCLUDED_TOO_FEW_ROWS
    assert rows["B"]["mdape"] == pytest.approx(0.30)


def test_the_absence_of_a_required_field_is_reported_and_excluded_whatever_its_size():
    # FR-01 refuses a request that omits a required field with a 422, so a
    # `"(missing)"` level cannot occur at serving time at all: it is a
    # data-quality defect the Great Expectations suite owns, not a population the
    # deployed API can be asked about. Excluded even at 600 rows, which is well
    # past the 500-row bar, so the rule is the reason rather than the size.
    levels = [MISSING_SEGMENT_LEVEL] * 600 + ["Dealer"] * 600
    rows = segment_rows(
        {"seller_type": pd.Series(levels, dtype="str")},
        pd.Series([10000.0] * 1200, dtype="float64"),
        pd.Series([10500.0] * 1200, dtype="float64"),
        variant="hand-built",
        min_rows=500,
    )
    by_level = {row["level"]: row for row in rows}

    assert by_level[MISSING_SEGMENT_LEVEL]["n"] == 600
    assert by_level[MISSING_SEGMENT_LEVEL]["counts_toward_sc04"] is False
    assert by_level[MISSING_SEGMENT_LEVEL]["excluded_because"] == EXCLUDED_ABSENT_REQUIRED_FIELD
    assert by_level["Dealer"]["counts_toward_sc04"] is True


def test_a_target_conditioned_segment_cannot_become_a_criterion(params):
    # The guard that makes the exclusion structural rather than a convention
    # somebody has to remember: adding `price_bucket` to `evaluate.segments`
    # fails the stage instead of silently gating on the target.
    #
    # Matching the reason and not only the segment name: `price_bucket` is cut
    # from `price`, which FR-01 does not require either, so the guard below it
    # raises a message that also names the segment. A mutation removing *this*
    # guard therefore survived a test that matched the name alone.
    with pytest.raises(ValueError, match=r"price_bucket.*condition on the price"):
        criterion_segments(
            {
                **params,
                "evaluate": {
                    **params["evaluate"],
                    "segments": [*params["evaluate"]["segments"], "price_bucket"],
                },
            }
        )


def test_a_segment_over_an_optional_input_field_is_refused(params):
    # The `"(missing)"` exclusion above is only correct while every segment is a
    # required input field. For an optional one, absence is a case the API
    # answers, so excluding it would hide exactly the rows worth looking at.
    with pytest.raises(ValueError, match="nr_prev_owners"):
        criterion_segments(
            {
                **params,
                "evaluate": {
                    **params["evaluate"],
                    "segments": [*params["evaluate"]["segments"], "nr_prev_owners"],
                },
            }
        )


def test_a_price_bucket_row_is_reported_and_cannot_count(params):
    # Reported with its metrics, and marked. The exclusion is read off the
    # segment's own name, so no caller can build a price-bucket row that counts.
    rows = segment_rows(
        {
            "price_bucket": bucket_levels(
                pd.Series([1000.0] * 600, dtype="float64"),
                params["evaluate"]["price_bucket_edges"],
            )
        },
        pd.Series([1000.0] * 600, dtype="float64"),
        pd.Series([1050.0] * 600, dtype="float64"),
        variant="hand-built",
        min_rows=500,
    )

    assert [row["level"] for row in rows] == ["0-5000"]
    assert rows[0]["n"] == 600
    assert rows[0]["mdape"] == pytest.approx(0.05)
    assert rows[0]["counts_toward_sc04"] is False
    assert rows[0]["excluded_because"] == EXCLUDED_TARGET_CONDITIONED


def test_sc04_refuses_rows_from_a_segment_it_may_not_gate(gate_criteria):
    # The second half of the same guarantee: even handed a price-bucket row
    # directly, the criterion refuses rather than ignoring it, so a future caller
    # cannot leak one in by concatenating the two tables.
    rows = segment_rows(
        {"price_bucket": pd.Series(["0-5000"] * 600, dtype="str")},
        pd.Series([1000.0] * 600, dtype="float64"),
        pd.Series([1050.0] * 600, dtype="float64"),
        variant="hand-built",
        min_rows=500,
    )
    with pytest.raises(ValueError, match="price_bucket"):
        evaluate_sc04(rows, gate_criteria, segments=("make",))


def test_every_segment_partitions_the_rows_and_no_level_is_empty(trained, test_frames, params):
    # Two properties at once, both of which a level built over zero rows would
    # break. `point_metrics` refuses an empty set, so a breakdown that could
    # produce one would fail the stage on a rare level. It cannot: the levels are
    # plain strings and `groupby` yields only the ones that occur. The row counts
    # summing to the frame length is the other half: no row is counted twice and
    # none is dropped.
    frame = test_frames["basic"]
    model = trained["models"]["lgbm-basic"]
    rows = segment_rows(
        {name: segment_level_column(frame, name, params) for name in criterion_segments(params)},
        frame["price"],
        model.predict_eur(frame),
        variant="lgbm-basic",
        min_rows=50,
    )

    assert rows, "the fixture has to produce segment rows"
    assert all(row["n"] >= 1 for row in rows)
    for segment in criterion_segments(params):
        counted = sum(row["n"] for row in rows if row["segment"] == segment)
        assert counted == len(frame), segment


def test_a_test_matrix_holding_a_make_the_model_was_not_fitted_on_is_refused(
    trained, matrices, tmp_path
):
    # Reachable only when the test matrix and the model's make list come from
    # different `split` runs. `features` owns the restriction (EDN-67), so the
    # stage checks the test rows against the model's own list instead of
    # filtering them again: a filter here would report on a population chosen by
    # whichever run the model came from, with no error anywhere.
    models = tmp_path / "models"
    shutil.copytree(trained["dir"], models)
    record = json.loads((models / "b0" / MODEL_FILE).read_text())
    record["training"]["supported_makes"] = ["Trabant"]
    (models / "b0" / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=r"b0 test: .* not one of the 1 supported make\(s\)"):
        evaluate_main(
            matrices["features"], models, tmp_path / "metrics", tmp_path / "metrics.json"
        )


def test_a_per_make_level_the_api_would_reject_is_refused():
    # EDN-48's third obligation. Without this the report would print an error
    # figure for a make FR-04 answers with a 422, which is worse than printing
    # nothing, and the restriction upstream would be an intention rather than a
    # checked property.
    rows = [{"segment": "make", "level": name} for name in ("BMW", "Trabant")]
    check_make_levels_are_served(rows[:1], supported=("BMW", "Audi"), variant="lgbm-basic")
    with pytest.raises(ValueError, match="Trabant"):
        check_make_levels_are_served(rows, supported=("BMW", "Audi"), variant="lgbm-basic")


def test_the_segments_of_the_real_params_are_all_required_input_fields(params):
    # The project's own configuration passes both guards, so neither is a rule
    # nobody could satisfy.
    assert criterion_segments(params) == tuple(params["evaluate"]["segments"])
    assert TARGET_CONDITIONED_SEGMENTS.isdisjoint(criterion_segments(params))


# --------------------------------------------------------------------------
# What an input field is, and what its absence looks like
# --------------------------------------------------------------------------

# No `req("FR-01")` marker on the SC-06 tests below, deliberately. The
# specification names two pieces of evidence for FR-01 - "API test per required
# field ...; the model test for SC-06" - and only the second exists. The matrix
# of NFR-07 has no notion of partial coverage, so a marker here would read as
# "verified by a test" and satisfy FR-01's M4 gate before the API tests that
# reject a request missing a required field are written. These tests *are* half of
# that evidence, and the marker belongs on them once the other half lands.


#: The optional fields of the basic set, which EDN-15 fixes at six: everything
#: FR-01 does not require. `age_years` must not be among them - FR-01 requires
#: `registration_date` and the model consumes the derived column - and that is
#: the whole reason `evaluate.input_field_to_feature` exists.
_BASIC_OPTIONAL_FIELDS = (
    "cylinders_volume_cc",
    "drive_train",
    "gears",
    "nr_doors",
    "nr_prev_owners",
    "nr_seats",
)


def test_the_optional_fields_of_the_basic_set_are_the_six_of_edn15(trained, params):
    fields = input_field_columns(trained["models"]["lgbm-basic"], params)

    assert optional_input_fields(fields, params) == _BASIC_OPTIONAL_FIELDS
    # The trap this mapping exists for: a naive difference over feature columns
    # would make `age_years` the seventh optional field and mask the single most
    # important feature in the model.
    assert "age_years" not in fields
    assert fields["registration_date"] == ("age_years",)


def test_an_equipment_field_covers_every_multi_hot_column_it_produced(trained, params):
    model = trained["models"]["lgbm-extended"]
    fields = input_field_columns(model, params)
    vocabulary = model.space.vocabulary

    assert vocabulary.equipment, "the extended fixture has to hold equipment columns"
    for source, items in vocabulary.equipment.items():
        assert len(fields[source]) == len(items), source
    # One request cannot leave out a single equipment item, so the sweep masks the
    # field: 23 scenarios rather than one per multi-hot column.
    assert len(optional_input_fields(fields, params)) == 23


def test_every_feature_column_is_claimed_by_exactly_one_input_field(trained, params):
    for variant in VARIANTS:
        model = trained["models"][variant]
        claimed = [
            column for columns in input_field_columns(model, params).values() for column in columns
        ]

        assert sorted(claimed) == sorted(model.features), variant
        assert len(claimed) == len(set(claimed)), variant


def test_a_required_field_the_model_has_no_column_for_is_refused(trained, params):
    # The required list and the trained model would then disagree about what a
    # valuation needs, and SC-06 would silently treat a required field as absent.
    with pytest.raises(ValueError, match="sunroof"):
        input_field_columns(
            trained["models"]["b0"],
            {
                **params,
                "evaluate": {
                    **params["evaluate"],
                    "required_input_fields": [
                        *params["evaluate"]["required_input_fields"],
                        "sunroof",
                    ],
                },
            },
        )


def test_a_derived_mapping_that_names_a_column_the_model_consumes_is_refused(trained, params):
    # `{"make": "age_years"}` would let the input field `make` claim both its own
    # column and the derived one, so masking `make` would mask two unrelated
    # fields at once.
    with pytest.raises(ValueError, match="make"):
        input_field_columns(
            trained["models"]["b0"],
            {
                **params,
                "evaluate": {
                    **params["evaluate"],
                    "input_field_to_feature": {"make": "age_years"},
                },
            },
        )


def test_a_column_claimed_by_two_input_fields_is_refused(trained, params):
    # A mapping that points at a column an equipment field already produced.
    model = trained["models"]["lgbm-extended"]
    equipment_column = input_field_columns(model, params)["equipment_comfort"][0]

    with pytest.raises(ValueError, match=equipment_column):
        input_field_columns(
            model,
            {
                **params,
                "evaluate": {
                    **params["evaluate"],
                    "input_field_to_feature": {
                        **params["evaluate"]["input_field_to_feature"],
                        "sunroof": equipment_column,
                    },
                },
            },
        )


# What an omitted field becomes is `Model.mask_absent`, asserted where it lives
# by `test_mask_absent_is_the_one_answer_to_what_an_omitted_field_is` above: the
# per-dtype rule, the `np.nan` trap it exists to prevent, and that it does not
# mutate the caller's frame. The sweep below calls it rather than carrying a
# second rule, so there is one answer to what masking a field means and the
# tests for it are in one place.


# --------------------------------------------------------------------------
# SC-06: a model that degrades when a field is left out
# --------------------------------------------------------------------------


class _DegradingModel:
    """A model honouring the seam that predicts worse when one field is absent.

    The fixture cannot demonstrate SC-06 on its own: the synthetic generator
    derives price from a formula that ignores the optional columns, so every
    single-field ratio measured on it came out within 0.6 % of 1.0 whatever the
    criterion did, and a fitted model there would pass at any threshold. This
    stub puts a known amount of degradation in one named field instead, so the
    expected ratio is arithmetic rather than a property of the data.

    It reads `price` from the frame it is handed, which no real model may do. That
    is what makes the resulting MdAPE exact: `price * factor` has an absolute
    percentage error of `factor - 1` on every row.
    """

    def __init__(self, real: Model, *, field: str, present: float, absent: float) -> None:
        self.variant = f"degrading-{field}"
        self.features = real.features
        self.input_schema = real.input_schema
        self.space = real.space
        # Borrowed rather than reimplemented: the stub has the real model's
        # features, schema and vocabulary, so a second expansion of a field into
        # its columns, or a second answer to what an omitted field is, could only
        # be a second answer to the same question.
        self.columns_of = real.columns_of
        self.mask_absent = real.mask_absent
        self._field = field
        self._present = present
        self._absent = absent

    def predict_eur(self, frame: pd.DataFrame) -> pd.Series:
        factor = self._absent if frame[self._field].isna().all() else self._present
        return pd.Series(frame["price"] * factor, index=frame.index, name=PREDICTION_NAME)


def _sweep_of(model, frame: pd.DataFrame, params: dict) -> "list[dict]":
    fields = input_field_columns(model, params)
    optional = optional_input_fields(fields, params)
    full = point_metrics(frame["price"], model.predict_eur(frame))["mdape"]
    return masking_sweep(model, frame, sc06_scenarios(fields, optional, params), full_mdape=full)


@pytest.mark.req("NFR-01")
def test_sc06_fails_and_names_the_field_a_model_degrades_on(
    trained, test_frames, params, gate_criteria
):
    # 5 % error with every field given, 20 % with `gears` left out, so the ratio
    # is 4.0 against the 1.5 bound.
    model = _DegradingModel(
        trained["models"]["lgbm-basic"], field="gears", present=1.05, absent=1.20
    )
    verdict = evaluate_sc06(_sweep_of(model, test_frames["basic"], params), gate_criteria)

    assert verdict["sc06_measured"] == pytest.approx(4.0)
    assert verdict["sc06_passed"] is False
    # The all-at-once scenario masks `gears` too, so it ties at 4.0; `max` keeps
    # the first of the two and the named field is the more useful of them.
    assert verdict["sc06_worst_field"] == "gears"
    assert verdict["sc06_n_scenarios"] == len(_BASIC_OPTIONAL_FIELDS) + 1


@pytest.mark.req("NFR-01")
def test_sc06_passes_a_model_that_degrades_within_the_bound(
    trained, test_frames, params, gate_criteria
):
    # The same machinery, 10 % against 14 %: a ratio of 1.4, which passes. The
    # pair is what shows the criterion follows the measurement.
    model = _DegradingModel(
        trained["models"]["lgbm-basic"], field="gears", present=1.10, absent=1.14
    )
    verdict = evaluate_sc06(_sweep_of(model, test_frames["basic"], params), gate_criteria)

    assert verdict["sc06_measured"] == pytest.approx(1.4)
    assert verdict["sc06_passed"] is True


def test_a_field_the_model_ignores_moves_nothing(trained, test_frames, params, gate_criteria):
    # B0 keys on make, model and an age bucket and ignores the rest, so masking
    # any optional field leaves its prediction unchanged. Worth pinning: a ratio
    # of exactly 1.0 across the sweep is the correct answer for that estimator
    # and not a sweep that silently failed to mask anything.
    sweep = _sweep_of(trained["models"]["b0"], test_frames["basic"], params)

    assert [row["field"] for row in sweep] == [*_BASIC_OPTIONAL_FIELDS, ALL_OPTIONAL_FIELDS]
    assert [row["mdape_ratio"] for row in sweep] == pytest.approx([1.0] * len(sweep))
    assert evaluate_sc06(sweep, gate_criteria)["sc06_passed"] is True


def _sweep_row(ratio: float, *, criterion: str = "sc06", field: str = "gears") -> dict:
    return {
        "variant": "hand-built",
        "criterion": criterion,
        "field": field,
        "n_columns_masked": 1,
        "mdape": 0.1 * ratio,
        "mdape_ratio": ratio,
    }


@pytest.mark.parametrize(("ratio", "passed"), [(1.4999, True), (1.5, True), (1.5000001, False)])
def test_the_sc06_bound_is_inclusive(ratio, passed, gate_criteria):
    # Asserted on an exact ratio rather than on one computed from two MdAPEs,
    # because `0.15 / 0.10` is 1.4999999999999998 in binary floating point and
    # would make the boundary look tested when it was not.
    assert evaluate_sc06([_sweep_row(ratio)], gate_criteria)["sc06_passed"] is passed


def test_sc06_is_not_measured_when_the_feature_set_leaves_nothing_optional(trained, gate_criteria):
    # A feature set of required fields only. No scenario is built, not even the
    # all-at-once one, which would otherwise be a call that masked nothing and
    # reported a ratio of 1.0 as if the criterion had been checked.
    params = _project_params()
    fields = input_field_columns(trained["models"]["b0"], params)
    assert sc06_scenarios(fields, (), params) == ()

    verdict = evaluate_sc06([], gate_criteria)

    assert verdict["sc06_measured"] is None
    assert verdict["sc06_passed"] is None
    assert "nothing to mask" in verdict["sc06_note"]


def test_p1_can_never_reach_sc06(gate_criteria):
    # P1 leaves out required fields, so its inflation is a large number about a
    # request FR-01 refuses. Built by a different function and refused here, so
    # the separation is structural rather than a matter of remembering.
    with pytest.raises(ValueError, match="required input field"):
        evaluate_sc06([_sweep_row(3.0, criterion="sc05", field=P1_SCENARIO)], gate_criteria)


def test_p1_masks_every_field_it_does_not_name(trained, params):
    model = trained["models"]["lgbm-basic"]
    fields = input_field_columns(model, params)
    given = params["evaluate"]["sc05_p1_fields"]

    scenarios = sc05_scenarios(fields, params)

    assert len(scenarios) == 1
    assert scenarios[0].criterion == "sc05"
    # The scenario names request fields; what they expand to is the seam's answer,
    # so the assertion goes through `columns_of` rather than re-deriving it.
    masked = {column for field in scenarios[0].fields for column in model.columns_of(field)}
    for field in given:
        assert masked.isdisjoint(fields[field]), field
    assert masked == set(model.features) - {column for field in given for column in fields[field]}
    assert scenarios[0].n_columns == len(masked)


def test_p1_over_a_field_the_feature_set_does_not_have_is_refused(trained, params):
    with pytest.raises(ValueError, match="paint_type"):
        sc05_scenarios(
            input_field_columns(trained["models"]["lgbm-basic"], params),
            {
                **params,
                "evaluate": {
                    **params["evaluate"],
                    "sc05_p1_fields": [*params["evaluate"]["sc05_p1_fields"], "paint_type"],
                },
            },
        )


def test_a_sweep_against_a_perfect_model_is_refused(trained, test_frames, params):
    # A median error of 0 means the model reproduces at least half the test prices
    # exactly, which is a leak or a test set built from training rows. Failing
    # loudly beats dividing by it.
    perfect = _DegradingModel(
        trained["models"]["lgbm-basic"], field="gears", present=1.0, absent=1.0
    )
    with pytest.raises(ValueError, match="no denominator"):
        _sweep_of(perfect, test_frames["basic"], params)


# --------------------------------------------------------------------------
# SC-05: unmeasurable today, and measured when a model can
# --------------------------------------------------------------------------


#: The two factors the stub interval is built from. An interval of
#: `[0.99, 1.01] * price` contains the price; one of `[1.50, 1.60] * price` sits
#: entirely above it, so the coverage of a frame is exactly the share of rows
#: given the first.
_INSIDE, _OUTSIDE = (0.99, 1.01), (1.50, 1.60)


class _IntervalModel(_DegradingModel):
    """A model that does expose `predict_interval_eur`, so the measured branch runs.

    The coverage is produced by construction rather than by fitting anything: the
    first `inside_rows` rows get an interval around the price and the rest get one
    above it, so the measured coverage is `inside_rows / len(frame)` exactly.
    """

    def __init__(self, real: Model, *, inside_rows: int) -> None:
        # Barely wrong on purpose: an exactly perfect point prediction would leave
        # SC-06's ratio without a denominator, which `masking_sweep` refuses.
        super().__init__(real, field="gears", present=1.0001, absent=1.0001)
        self._inside_rows = inside_rows

    def predict_interval_eur(self, frame: pd.DataFrame, coverage: float = 0.90) -> pd.DataFrame:
        inside = np.arange(len(frame)) < self._inside_rows
        price = frame["price"].to_numpy()
        return pd.DataFrame(
            {
                "lower_eur": np.where(inside, price * _INSIDE[0], price * _OUTSIDE[0]),
                "upper_eur": np.where(inside, price * _INSIDE[1], price * _OUTSIDE[1]),
            },
            index=frame.index,
        )


@pytest.mark.req("NFR-01")
def test_sc05_is_not_measured_while_no_model_exposes_an_interval(
    trained, test_frames, gate_criteria
):
    # The state of the first delivery, and the reason NFR-01's gate cannot be met
    # by any model yet. Reported as a structured status so a reader can tell "not
    # measured" from "measured and failed" without parsing English.
    for variant in VARIANTS:
        model = trained["models"][variant]
        assert not hasattr(model, INTERVAL_METHOD), variant
        assert interval_metrics(model, {"full": test_frames[model.feature_set]}) is None, variant

    verdict = evaluate_sc05(None, gate_criteria)

    assert verdict["sc05_measured"] is None
    assert verdict["sc05_passed"] is None
    assert verdict["sc05_status"] == "not_measured"
    assert verdict["sc05_capability_checked"] == INTERVAL_METHOD
    assert INTERVAL_METHOD in verdict["sc05_reason"]


def test_interval_coverage_is_the_share_of_prices_inside_the_interval(trained, test_frames):
    # The capability branch, exercised with a stub that has the method, which is
    # what makes the `hasattr` check a seam rather than a dead branch: when UC2
    # lands, the criterion starts being measured without an edit to the stage.
    frame = test_frames["basic"]
    rows = len(frame)
    # Enough rows outside the interval that a coverage of 1.0 could not pass by
    # accident, and a count rather than a share so the expected value is exact.
    missed = 37
    inside_rows = rows - missed
    model = _IntervalModel(trained["models"]["lgbm-basic"], inside_rows=inside_rows)

    measured = interval_metrics(model, {"full": frame})

    assert rows > missed, "the fixture has to hold more rows than the interval misses"
    assert measured["full"]["coverage"] == pytest.approx(inside_rows / rows)
    # The width is relative to the prediction, which is `price * 1.0001`.
    expected_width = (
        inside_rows * (_INSIDE[1] - _INSIDE[0])
        + (rows - inside_rows) * (_OUTSIDE[1] - _OUTSIDE[0])
    ) / (rows * 1.0001)
    assert measured["full"]["mean_relative_width"] == pytest.approx(expected_width)


@pytest.mark.parametrize(
    ("coverage", "passed"),
    [(0.8799, False), (0.88, True), (0.90, True), (0.92, True), (0.9201, False)],
)
def test_the_sc05_coverage_band_is_inclusive_at_both_edges(coverage, passed, gate_criteria):
    # Exact coverages rather than ones a stub's row count produced, so both edges
    # of the 88 % to 92 % band are tested where they are rather than near them.
    verdict = evaluate_sc05({"full": {"coverage": coverage}}, gate_criteria)

    assert verdict["sc05_passed"] is passed
    assert verdict["sc05_measured"] == coverage
    assert verdict["sc05_status"] == "measured"
    assert verdict["sc05_worst_scenario"] == "full"


def test_sc05_needs_both_scenarios_inside_the_band(gate_criteria):
    # The criterion names full inputs *and* P1, so one scenario outside the band
    # fails it, and the value reported is the one that carried the verdict.
    verdict = evaluate_sc05(
        {"full": {"coverage": 0.90}, P1_SCENARIO: {"coverage": 0.70}}, gate_criteria
    )

    assert verdict["sc05_passed"] is False
    assert verdict["sc05_measured"] == 0.70
    assert verdict["sc05_worst_scenario"] == P1_SCENARIO


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

#: The exploratory reference values of problem-spec section 8, and a model that
#: misses every criterion. Shared by the tests below so a change to one shows up
#: in both directions at once.
_GOOD_METRICS = {"mdape": 0.067, "within_20pct": 0.906}
_BAD_METRICS = {"mdape": 0.547, "within_20pct": 0.161}


def _gate_input(
    metrics: dict, *, baseline: "float | None", mdape: float, ratio: float
) -> GateInput:
    """A gate input whose SC-04 and SC-06 halves are as good or as bad as `metrics`."""
    return GateInput(
        metrics=metrics,
        baseline_mdape=baseline,
        segments=[
            {
                "variant": "hand-built",
                "segment": "make",
                "level": "BMW",
                "n": 600,
                "mdape": mdape,
                "counts_toward_sc04": True,
                "excluded_because": "",
            }
        ],
        criterion_segments=("make",),
        sweep=[_sweep_row(ratio)],
        intervals=None,
    )


@pytest.mark.req("NFR-01")
def test_the_gate_flips_every_measured_criterion_together(gate_criteria):
    # The test that tells a working gate from one that says no to everything: the
    # five criteria that can be measured today all pass on the good measurement
    # and all fail on the bad one.
    passed = evaluate_gate(
        _gate_input(_GOOD_METRICS, baseline=0.119, mdape=0.10, ratio=1.2), gate_criteria
    )
    failed = evaluate_gate(
        _gate_input(_BAD_METRICS, baseline=0.119, mdape=0.40, ratio=3.0), gate_criteria
    )
    measurable = ("sc01", "sc02", "sc03", "sc04", "sc06")

    assert [passed[f"{name}_passed"] for name in measurable] == [True] * len(measurable)
    assert [failed[f"{name}_passed"] for name in measurable] == [False] * len(measurable)


@pytest.mark.req("NFR-01")
def test_an_unmeasured_criterion_blocks_the_gate_rather_than_passing_it(gate_criteria):
    # NFR-01's word is "every". A model meeting all five criteria that can be
    # measured is still not deployable while the sixth has no measurement, and
    # `gate_blocked_by` says which of the two it is.
    record = evaluate_gate(
        _gate_input(_GOOD_METRICS, baseline=0.119, mdape=0.10, ratio=1.2), gate_criteria
    )

    assert record["sc05_passed"] is None
    assert gate_passed(record) is False
    assert gate_blocked_by(record) == "sc05 unmeasured"


@pytest.mark.req("NFR-01")
def test_gate_blocked_by_tells_a_miss_apart_from_a_non_measurement(gate_criteria):
    record = evaluate_gate(
        _gate_input(_BAD_METRICS, baseline=0.119, mdape=0.40, ratio=3.0), gate_criteria
    )

    assert gate_blocked_by(record) == "sc01,sc02,sc03,sc04,sc06 failed; sc05 unmeasured"


@pytest.mark.req("NFR-01")
def test_the_baseline_compared_against_itself_does_not_pass_sc03(gate_criteria):
    # An improvement of exactly 0 over the baseline is not a 30 % improvement, so
    # B0 fails SC-03 against itself. Worth pinning, because reading that as a
    # missing measurement would turn a real failure into a `None`.
    record = evaluate_gate(
        _gate_input(
            {"mdape": 0.119, "within_20pct": 0.709}, baseline=0.119, mdape=0.10, ratio=1.0
        ),
        gate_criteria,
    )

    assert record["sc03_measured"] == 0.0
    assert record["sc03_passed"] is False


@pytest.mark.req("NFR-01")
@pytest.mark.parametrize("baseline", [None, 0.0])
def test_sc03_is_not_measured_without_a_baseline_to_improve_on(baseline, gate_criteria):
    # Two ways there is no improvement to compute. A missing baseline variant is
    # the obvious one; a baseline MdAPE of exactly 0 is the other, and it has to
    # be told apart from a missing one rather than divided by: nothing is 30 %
    # better than a perfect baseline, so the criterion is unmeasurable rather
    # than failed. `not baseline` would conflate the two, and dropping the check
    # raises `ZeroDivisionError` from inside the gate.
    record = evaluate_gate(
        _gate_input(_GOOD_METRICS, baseline=baseline, mdape=0.10, ratio=1.2), gate_criteria
    )

    assert record["sc03_measured"] is None
    assert record["sc03_passed"] is None
    assert gate_passed(record) is False
    assert "sc03" in gate_blocked_by(record)


@pytest.mark.req("NFR-01")
def test_a_record_that_decides_nothing_for_a_criterion_is_refused():
    # `gate_passed` is `all(...)` over the six, so a criterion simply absent would
    # raise `KeyError` instead of blocking. This is the machine-checked version of
    # what the CRITERIA comment says.
    decided = {f"{name}_passed": True for name in CRITERIA}
    assert gate_passed(decided) is True
    assert gate_blocked_by(decided) == ""

    for name in CRITERIA:
        with pytest.raises(KeyError):
            gate_passed({key: value for key, value in decided.items() if key != f"{name}_passed"})


@pytest.mark.req("NFR-01")
def test_a_criterion_added_to_the_list_without_a_measurement_fails_the_stage(
    monkeypatch, gate_criteria
):
    # The failure the check above guards against in practice: a seventh criterion
    # is added to `CRITERIA` and nobody measures it. Without the check the record
    # would simply not carry it, `gate_passed` would raise `KeyError` from inside
    # the stage, and the traceback would name a dictionary rather than the
    # criterion nobody implemented.
    monkeypatch.setattr("recommenditos.modeling.evaluate.CRITERIA", (*CRITERIA, "sc07"))

    with pytest.raises(ValueError, match="decides nothing for sc07"):
        evaluate_gate(
            _gate_input(_GOOD_METRICS, baseline=0.119, mdape=0.10, ratio=1.2), gate_criteria
        )


def _variant_record(name: str, mdape: float, **verdicts) -> dict:
    """One variant's record as `gate_summary` reads it: MdAPE plus the six verdicts."""
    decided = {f"{each}_passed": True for each in CRITERIA} | verdicts
    record = {
        "variant": name,
        "estimator": "lightgbm",
        "mdape": mdape,
        "within_20pct": 0.9,
        **decided,
    }
    return {
        **record,
        "gate_passed": gate_passed(record),
        "gate_blocked_by": gate_blocked_by(record),
    }


@pytest.mark.req("NFR-01")
def test_the_metrics_file_puts_forward_the_best_variant_that_met_every_criterion():
    # `best_variant` is the lowest MdAPE overall and `deployable_variant` the
    # lowest among those that passed, and the two are different leaves precisely
    # because they are different questions: the first is what a reader looks for,
    # the second is what NFR-01 gates on.
    summary = gate_summary(
        {
            "b0": _variant_record("b0", 0.12, sc01_passed=False),
            "lgbm-basic": _variant_record("lgbm-basic", 0.068),
            "lgbm-extended": _variant_record("lgbm-extended", 0.063, sc04_passed=False),
        },
        data_source="zenodo",
    )

    assert summary["gate_passed"] is True
    assert summary["n_variants_passing"] == 1
    assert summary["deployable_variant"] == "lgbm-basic"
    assert summary["best_variant"] == "lgbm-extended"
    assert summary["best_mdape"] == 0.063
    assert summary["criteria_not_measured"] == ""
    assert summary["variants"]["lgbm-extended"]["gate_blocked_by"] == "sc04 failed"


@pytest.mark.req("NFR-01")
def test_the_metrics_file_counts_the_variants_only_a_missing_measurement_blocks():
    # `n_variants_passing_measurable` counts the variants where nothing that
    # *could* be measured failed, which is not the same as the gate: with SC-05
    # unmeasurable, `n_variants_passing` says 0 for a reason that has nothing to
    # do with the models, and this says how close they are. A variant that also
    # failed a measured criterion must not be counted.
    summary = gate_summary(
        {
            "b0": _variant_record("b0", 0.12, sc01_passed=False, sc05_passed=None),
            "lgbm-basic": _variant_record("lgbm-basic", 0.068, sc05_passed=None),
            "lgbm-extended": _variant_record("lgbm-extended", 0.063, sc05_passed=None),
        },
        data_source="zenodo",
    )

    assert summary["gate_passed"] is False
    assert summary["n_variants_passing"] == 0
    assert summary["n_variants_passing_measurable"] == 2
    assert summary["deployable_variant"] is None
    assert summary["criteria_not_measured"] == "sc05"


@pytest.mark.req("NFR-01")
def test_a_verdict_that_is_truthy_without_being_a_bool_is_refused(gate_criteria):
    # Not a hypothetical: a numpy scalar's comparison returns `np.bool_`, which is
    # truthy, prints as `True` and `is not True`. `point_metrics` converts to
    # `float` so the stage cannot produce one today, and this is the check that
    # says so if that ever stops being the case.
    numpy_metrics = {"mdape": np.float64(0.067), "within_20pct": np.float64(0.906)}
    assert (numpy_metrics["mdape"] <= 0.09) is not True

    with pytest.raises(ValueError, match="True, False or None"):
        evaluate_gate(
            _gate_input(numpy_metrics, baseline=0.119, mdape=0.10, ratio=1.2), gate_criteria
        )
