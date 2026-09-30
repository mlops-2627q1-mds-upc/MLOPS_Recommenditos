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

import json
from pathlib import Path
import shutil

import lightgbm
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import pytest
from tests.conftest import PII_COLUMNS, params_override

from recommenditos.config import PARAMS_FILE
from recommenditos.data import build_features, preprocess, split_data
from recommenditos.data.build_features import FeatureSpace, Vocabulary, read_supported_makes
from recommenditos.modeling import train
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
def test_frames(matrices: dict, supported_makes: tuple[str, ...]) -> dict:
    """The test matrix per feature set, restricted to the makes the API serves.

    The population `evaluate` reports on (EDN-48), so a test that scores a model
    here scores it on the same rows the stage would.
    """
    frames = {}
    for feature_set in _feature_sets():
        frame = pd.read_parquet(matrices["features"] / feature_set / "test.parquet")
        frames[feature_set] = frame[frame["make"].isin(supported_makes)]
    return frames


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
    """
    train_matrix = pd.read_parquet(matrices["features"] / "basic" / "train.parquet")
    expected = int(train_matrix["make"].isin(supported_makes).sum())
    assert expected < len(train_matrix), (
        "the fixture has to contain an unsupported make, or this test cannot fail"
    )
    for variant, model in trained["models"].items():
        assert tuple(model.metadata["training"]["supported_makes"]) == supported_makes, variant
        assert model.metadata["training"]["n_train_rows"] == expected, variant


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
    variant: str, trained: dict, spaces: dict, test_frames: dict
):
    """FR-01's promise, at the model layer: leaving an optional field out is allowed.

    Masked the way EDN-23 says the domain represents absence - an assertion flag
    is `False`, an equipment column is `False`, everything else is missing - and
    over the whole test frame rather than one row, because a single row cannot
    show a masked column destroying the values in the rows beside it.
    """
    model = trained["models"][variant]
    schema = spaces[model.feature_set].schema
    required = set(_project_params()["evaluate"]["required_input_fields"]) | {"age_years"}
    masked = _mask(
        test_frames[model.feature_set],
        schema,
        [name for name in model.features if name not in required],
    )

    predicted = model.predict_eur(masked)

    assert np.isfinite(predicted).all()
    assert (predicted > 0).all()


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
    schema = spaces[model.feature_set].schema
    masked = _mask(test_frames[model.feature_set], schema, ["weight_kg", "body_color"])

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


def test_a_budget_early_stopping_never_reaches_is_recorded_as_the_trees_it_built(
    tmp_path: Path, matrices: dict, params: dict, monkeypatch
):
    """`best_iteration_` is 0 when early stopping never fired, and 0 means no trees.

    Not a corner case: on the real snapshot `lgbm-basic` used all 1,000 trees and
    `lgbm-extended` stopped at 996, so the budget is what binds there and this is
    the branch that run takes. The fixture always early-stops, so the budget is
    lowered here to reach it.
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

    train.main("lgbm-basic", matrices["features"], tmp_path / "models", overridden)

    record = json.loads(
        (tmp_path / "models" / "lgbm-basic" / MODEL_FILE).read_text(encoding="utf-8")
    )
    assert record["training"]["best_iteration"] == 3
    booster = lightgbm.Booster(model_file=str(tmp_path / "models" / "lgbm-basic" / "booster.txt"))
    assert booster.num_trees() == 3


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


def test_the_booster_is_saved_at_its_early_stopped_iteration(trained: dict):
    """The file is the model, so predict needs no iteration argument.

    A consumer that had to remember `num_iteration=best_iteration` would sooner
    or later forget, and get a quietly overfitted prediction.

    Worth saying what this does and does not pin: removing the explicit
    `num_iteration` from the save does not make it fail, because lightgbm 4.7
    already truncates the booster when early stopping fires. It pins the
    property, so it would catch a LightGBM that stopped truncating or a
    configuration in which early stopping never fires - which the second
    assertion is there to rule out for the fixture as it stands.
    """
    for variant in ("lgbm-basic", "lgbm-extended"):
        best = trained["models"][variant].metadata["training"]["best_iteration"]
        booster = lightgbm.Booster(model_file=str(trained["dir"] / variant / "booster.txt"))
        assert booster.num_trees() == best, variant
        assert best < _variants()[variant]["params"]["n_estimators"], (
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


def test_a_bundle_whose_two_halves_disagree_is_refused(tmp_path: Path, trained: dict):
    """A record and a feature space from different runs would predict silently wrong.

    The two files are written together and restored together, so this needs a
    hand-edited or half-restored bundle - which is exactly the case where nothing
    else would notice.
    """
    bundle = tmp_path / "mismatched"
    shutil.copytree(trained["dir"] / "lgbm-basic", bundle)
    record = json.loads((bundle / MODEL_FILE).read_text(encoding="utf-8"))
    record["features"] = record["features"][:-1]
    (bundle / MODEL_FILE).write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ModelError, match="disagree"):
        load_model(bundle)


def test_the_ridge_emits_a_missing_indicator_for_every_numeric_feature(trained: dict):
    """EDN-51's load-bearing half, which no prediction on this fixture can show.

    scikit-learn's default is an indicator only for the features that were
    missing *at fit time*. A column that happens to be complete in the training
    split would then be mean-filled with no indicator the moment SC-06 masks it,
    and the criterion would be measuring the model's own imputation rather than
    the cost of the missing field. `features="all"` is what prevents that, and it
    is invisible from the outside, so it is asserted on the fitted pipeline.
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
    """The case `features="all"` exists for, which the fixture does not produce.

    Every numeric column of the synthetic fixture happens to have a missing value
    in the training split, so the default `features="missing-only"` emits the same
    set of indicators there and the test above cannot tell the two apart. This one
    fills one column completely first, which is the situation on real data:
    scikit-learn would then mean-fill it with no indicator the moment SC-06 masks
    it, and the criterion would be measuring the model's own imputation.
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


def test_training_fails_when_no_row_is_a_supported_make(
    matrices: dict, tmp_path: Path, params: dict, monkeypatch
):
    """A make list from another `split` run would otherwise fit on nothing.

    The threshold is set absurdly high rather than the list being edited, because
    that is the way it happens: `split.min_listings_per_make` is swept, the
    matrices are not rebuilt, and an empty intersection would train a model on
    zero rows and report metrics for it.
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
    """`evaluate` reports SC-05 as null exactly while this attribute does not exist.

    A stub that raised would make the capability check pass and the criterion
    look measured, which is the one thing NFR-01 cannot tolerate.
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
        # NFR-06 asks every run to record which code produced it, and the seam
        # does that rather than each caller.
        assert run.data.tags["git_commit"]
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
    JSON. Absence follows EDN-23: an assertion flag and an equipment column are
    `False`, everything else is `None`.
    """
    required = set(_project_params()["evaluate"]["required_input_fields"]) | {"age_years"}
    row = {}
    for name in model.features:
        column = schema.column(name)
        if name not in required:
            row[name] = False if column.dtype in {"bool", "boolean"} else None
        elif column.levels is not None:
            row[name] = str(frame[name].dropna().iloc[0])
        else:
            row[name] = float(frame[name].dropna().iloc[0])
    return row


def _mask(frame: pd.DataFrame, schema: Schema, columns: list[str]) -> pd.DataFrame:
    """`frame` with `columns` set to what the API sends when the field is omitted.

    A dtype-driven rule, because the values are not interchangeable: assigning
    `np.nan` into a pandas `bool` column upcasts the column to float64 and turns
    every value into NaN, which destroys the column rather than masking a field.
    """
    masked = frame.copy()
    for name in columns:
        column = schema.column(name)
        if column.dtype == "bool":
            masked[name] = False
        elif column.dtype == "boolean":
            masked[name] = pd.array([False] * len(masked), dtype="boolean")
        elif column.levels is not None:
            masked[name] = pd.Categorical([None] * len(masked), categories=column.levels)
        else:
            masked[name] = np.nan
    return masked


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
