"""Training on blanked copies of the listings (EDN-82).

A variant whose `params.yaml` block names `blanked_copies` is fitted on its
training rows plus copies of them with optional input fields set to what an
omitted field is, so that a request carrying few optional fields is something
the model has seen. These tests hold the three places that behaviour lives in:
the seam's row-wise `mask_absent`, the copies themselves, and the fit and the
stage that use them.
"""

import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from tests.conftest import params_override, project_params, train_variants

from recommenditos.modeling import evaluate, train
from recommenditos.modeling.model import (
    ESTIMATORS,
    MODEL_FILE,
    Model,
    ModelError,
    RequestContract,
    fit_variant,
    optional_seam_fields,
    with_blanked_copies,
)
from recommenditos.tracking import REQUIRED_ENV_VARS

#: One field of each kind `mask_absent` treats differently: an equipment list,
#: an assertion flag, a categorical and a number.
FOUR_KINDS = ("equipment_comfort", "has_full_service_history", "body_color", "weight_kg")

#: The optional input fields of the basic set, which are the six of EDN-15. Each
#: is one matrix column, and an omitted one is null.
BASIC_OPTIONAL = (
    "cylinders_volume_cc",
    "drive_train",
    "gears",
    "nr_doors",
    "nr_prev_owners",
    "nr_seats",
)


def _rows_with_all_four_given(model, frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """`n` rows whose four fields all differ from an omitted one, so blanking shows."""
    equipment = list(model.columns_of("equipment_comfort"))
    given = (
        frame[equipment].any(axis=1)
        & frame["has_full_service_history"]
        & frame["body_color"].notna()
        & frame["weight_kg"].notna()
    )
    rows = frame[given.to_numpy(dtype=bool)].head(n).reset_index(drop=True)
    assert len(rows) == n, "the fixture has too few fully described rows for this test"
    return rows


def test_mask_absent_blanks_a_field_only_in_the_rows_it_is_told_to(
    trained: dict, test_frames: dict
):
    """Row by row, the value is the listing's own or the omitted one, and nothing else."""
    model = trained["models"]["lgbm-extended"]
    frame = _rows_with_all_four_given(model, test_frames[model.feature_set], 4)
    before = frame.copy(deep=True)
    equipment = list(model.columns_of("equipment_comfort"))
    where = np.array(
        [
            [True, True, True, True],  # everything
            [False, False, False, False],  # nothing
            [False, False, False, True],  # the number alone
            [True, False, False, False],  # the equipment list alone
        ]
    )

    blanked = model.mask_absent(frame, FOUR_KINDS, where=where)

    # A row told to lose every field is the row `mask_absent` gives without `where`,
    # and a row told to lose none is the listing's own.
    everywhere = model.mask_absent(frame, FOUR_KINDS)
    pd.testing.assert_frame_equal(blanked.iloc[[0]], everywhere.iloc[[0]])
    pd.testing.assert_frame_equal(blanked.iloc[[1]], frame.iloc[[1]])
    # The number alone: NaN there, and the other three as the listing has them.
    assert np.isnan(blanked.loc[2, "weight_kg"])
    assert blanked.loc[2, "body_color"] == frame.loc[2, "body_color"]
    assert blanked.loc[2, "has_full_service_history"]
    assert blanked.loc[2, equipment].any()
    # The equipment list alone: empty, never null, and the number still there.
    assert not blanked.loc[3, equipment].any()
    assert blanked.loc[3, equipment].notna().all()
    assert blanked.loc[3, "weight_kg"] == frame.loc[3, "weight_kg"]
    # Still the contract's frame, and the caller's frame is untouched.
    assert blanked.dtypes.equals(frame.dtypes)
    pd.testing.assert_frame_equal(frame, before)


def test_mask_absent_refuses_a_where_of_another_shape(trained: dict, test_frames: dict):
    model = trained["models"]["lgbm-extended"]
    frame = test_frames[model.feature_set].head(3)

    with pytest.raises(ModelError, match="3 row.*4 field"):
        model.mask_absent(frame, FOUR_KINDS, where=np.ones((3, 2), dtype=bool))


def test_mask_absent_goes_by_position_whatever_index_the_frame_has(
    trained: dict, test_frames: dict
):
    """A request frame need not be numbered from zero, and its labels need not be unique."""
    model = trained["models"]["lgbm-extended"]
    frame = _rows_with_all_four_given(model, test_frames[model.feature_set], 3)
    frame.index = [10, 10, 7]
    where = [[False] * 4, [True] * 4, [False] * 4]  # a plain list, the second row only

    blanked = model.mask_absent(frame, FOUR_KINDS, where=where)

    assert list(blanked.index) == [10, 10, 7]
    assert blanked["weight_kg"].isna().tolist() == [False, True, False]
    assert blanked["body_color"].isna().tolist() == [False, True, False]
    assert blanked["has_full_service_history"].tolist() == [True, False, True]


def test_a_field_named_twice_is_omitted_where_either_mention_says_so(
    trained: dict, test_frames: dict
):
    model = trained["models"]["lgbm-extended"]
    frame = _rows_with_all_four_given(model, test_frames[model.feature_set], 3)
    where = pd.DataFrame([[True, False], [False, True], [False, False]])

    blanked = model.mask_absent(frame, ["weight_kg", "weight_kg"], where=where)

    assert blanked["weight_kg"].isna().tolist() == [True, True, False]


def test_mask_absent_refuses_single_rows_of_a_frame_outside_the_contract(
    trained: dict, test_frames: dict
):
    """A categorical over other levels cannot hold the contract's omitted value in some rows."""
    model = trained["models"]["lgbm-extended"]
    frame = _rows_with_all_four_given(model, test_frames[model.feature_set], 3)
    frame["body_color"] = frame["body_color"].astype(str).astype("category")

    with pytest.raises(ModelError, match="Conform the frame"):
        model.mask_absent(frame, ["body_color"], where=np.ones((3, 1), dtype=bool))


# --------------------------------------------------------------------------
# The copies
# --------------------------------------------------------------------------


def test_blanked_copies_follow_the_rows_and_omit_only_the_fields_named(
    trained: dict, test_frames: dict
):
    model = trained["models"]["lgbm-basic"]
    frame = test_frames["basic"].reset_index(drop=True)
    n_rows = len(frame)

    result = with_blanked_copies(
        model, frame, BASIC_OPTIONAL, copies={"all": 1, "random": 2}, seed=7
    )

    assert len(result) == 4 * n_rows
    assert result.dtypes.equals(frame.dtypes)
    rows, everything, first, second = (
        result.iloc[block * n_rows : (block + 1) * n_rows].reset_index(drop=True)
        for block in range(4)
    )
    # The rows themselves come first and are left as they are.
    pd.testing.assert_frame_equal(rows, frame)
    # An `all` copy is the request that carries none of the fields.
    pd.testing.assert_frame_equal(everything, model.mask_absent(frame, BASIC_OPTIONAL))
    # A `random` copy keeps every other column, the price it is fitted on included.
    untouched = [name for name in frame.columns if name not in BASIC_OPTIONAL]
    assert {"price", "log_price", "make", "age_years"} <= set(untouched)
    for copy in (first, second):
        pd.testing.assert_frame_equal(copy[untouched], frame[untouched])
    assert not first.equals(second), "two random copies have to be two draws"
    # Where a row gives all six fields, a random copy omits at least one and at
    # most all of them, and every number in between occurs.
    given = frame[list(BASIC_OPTIONAL)].notna().all(axis=1)
    assert given.sum() >= 50, "the fixture has too few fully described rows for this test"
    omitted = first.loc[given, list(BASIC_OPTIONAL)].isna().sum(axis=1)
    assert set(omitted) == {1, 2, 3, 4, 5, 6}


def test_the_seed_decides_the_random_copies_and_nothing_else_does(
    trained: dict, test_frames: dict
):
    model = trained["models"]["lgbm-basic"]
    frame = test_frames["basic"].reset_index(drop=True)

    def copies(seed: int) -> pd.DataFrame:
        return with_blanked_copies(
            model, frame, BASIC_OPTIONAL, copies={"all": 0, "random": 1}, seed=seed
        )

    pd.testing.assert_frame_equal(copies(7), copies(7))
    assert not copies(7).equals(copies(8))


def test_no_copies_leaves_the_frame_as_it_is(trained: dict, test_frames: dict):
    model = trained["models"]["lgbm-basic"]
    frame = test_frames["basic"]

    result = with_blanked_copies(
        model, frame, BASIC_OPTIONAL, copies={"all": 0, "random": 0}, seed=7
    )

    pd.testing.assert_frame_equal(result, frame)


#: Which of the six fields the measurement's random copy omits in each of six
#: rows, for the project seed: the draw `reports/analysis/blanked_fields.py` made
#: when it ran for EDN-82 (commit 4a7cdfe). Written down here, so that a test
#: need not import a script that is bound to that commit.
MEASURED_DRAW = (
    (1, 0, 0, 0, 1, 0),
    (1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 0),
    (1, 0, 1, 0, 1, 0),
    (1, 0, 1, 1, 1, 1),
    (1, 1, 1, 1, 0, 1),
)


def test_the_random_copies_are_the_draw_the_measurement_made(trained: dict, test_frames: dict):
    """What EDN-82 adopted is the configuration it measured, down to the draws.

    `params.yaml` names the measurement's `all+random` configuration, and the
    figures the decision rests on are those of the frame the measurement built.
    So the pipeline's random copy for the project seed has to omit what the
    measurement's did, or the adopted model is another one.
    """
    model = trained["models"]["lgbm-basic"]
    frame = test_frames["basic"]
    given = frame[frame[list(BASIC_OPTIONAL)].notna().all(axis=1)].head(6).reset_index(drop=True)
    assert len(given) == len(MEASURED_DRAW)
    seed = project_params()["seed"]
    assert seed == 20251108, "the draw above is the one of this seed"

    result = with_blanked_copies(model, given, BASIC_OPTIONAL, copies={"random": 1}, seed=seed)

    omitted = result.iloc[len(given) :][list(BASIC_OPTIONAL)].isna().to_numpy()
    assert omitted.astype(int).tolist() == [list(row) for row in MEASURED_DRAW]


@pytest.mark.parametrize("variant", ["lgbm-basic", "lgbm-extended"])
def test_a_copy_omits_exactly_the_fields_sc06_masks_all_at_once(
    trained: dict, params: dict, variant: str
):
    """One meaning of "optional" for the fit and the gate, also where a field is many columns."""
    model = trained["models"][variant]
    fields = evaluate.input_field_columns(model, params)
    scenarios = evaluate.sc06_scenarios(
        fields, evaluate.optional_input_fields(fields, params), params
    )

    assert scenarios[-1].field == evaluate.ALL_OPTIONAL_FIELDS
    assert optional_seam_fields(model, RequestContract.from_params(params)) == scenarios[-1].fields


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def basic_data(matrices: dict, supported_makes):
    """The basic matrices as the `train` stage hands them to a fit, contract included."""
    return train.read_matrices(
        matrices["features"],
        "basic",
        supported_makes,
        request=RequestContract.from_params(project_params()),
    )


def _lgbm_basic(blanked_copies: dict | None) -> dict:
    """The committed `lgbm-basic` block, with the copies a test asks for or none at all."""
    settings = {
        key: value
        for key, value in train_variants()["lgbm-basic"].items()
        if key != "blanked_copies"
    }
    return settings if blanked_copies is None else {**settings, "blanked_copies": blanked_copies}


def test_a_variant_with_blanked_copies_is_fitted_on_its_rows_and_their_copies(basic_data):
    model = fit_variant(
        "lgbm-basic", _lgbm_basic({"all": 1, "random": 2}), basic_data, seed=1, num_threads=1
    )

    training = model.metadata["training"]
    # The listings stay the listings; what the estimator saw is counted beside them.
    assert training["n_train_rows"] == len(basic_data.train)
    assert training["n_fit_rows"] == 4 * len(basic_data.train)
    # The bundle says what was blanked: the six optional fields of EDN-15.
    assert model.metadata["blanked_copies"] == {
        "all": 1,
        "random": 2,
        "fields": list(BASIC_OPTIONAL),
    }


def test_a_variant_without_copies_is_fitted_on_its_rows_alone(basic_data):
    """So a block that never names `blanked_copies` fits what it fitted before the key existed."""
    unnamed = fit_variant("lgbm-basic", _lgbm_basic(None), basic_data, seed=1, num_threads=1)
    zero = fit_variant(
        "lgbm-basic", _lgbm_basic({"all": 0, "random": 0}), basic_data, seed=1, num_threads=1
    )

    assert unnamed.booster.model_to_string() == zero.booster.model_to_string()
    assert unnamed.metadata["training"]["n_fit_rows"] == len(basic_data.train)
    assert unnamed.metadata["blanked_copies"] == {"all": 0, "random": 0, "fields": []}


def test_the_copies_reach_the_estimator(basic_data):
    plain = fit_variant("lgbm-basic", _lgbm_basic(None), basic_data, seed=1, num_threads=1)
    copied = fit_variant("lgbm-basic", _lgbm_basic({"all": 1}), basic_data, seed=1, num_threads=1)

    assert plain.booster.model_to_string() != copied.booster.model_to_string()
    assert copied.metadata["blanked_copies"] == {
        "all": 1,
        "random": 0,
        "fields": list(BASIC_OPTIONAL),
    }


class _RecordingModel(Model):
    """An estimator that keeps the frames it is fitted on, and prices by one field.

    Registered under `ESTIMATORS` for one test, the way a new estimator joins
    the ladder, so that what `fit_variant` hands an estimator can be read. It
    answers 9 in log euros where `nr_seats` is omitted and 10 where it is
    given, so its error on the listings and on their blanked copies differ.
    """

    estimator_name = "recording"
    supports_blanked_copies = True
    seen: ClassVar[dict] = {}

    @classmethod
    def fit(cls, metadata, data, *, seed, num_threads):
        cls.seen = {"train": data.train, "validation": data.validation}
        return cls(metadata, data.space)

    def _predict_log_price(self, aligned):
        return np.where(aligned["nr_seats"].isna(), 9.0, 10.0)


def test_the_estimator_sees_the_copies_and_nothing_else_changes(basic_data, monkeypatch):
    monkeypatch.setitem(ESTIMATORS, "recording", _RecordingModel)
    settings = {
        "estimator": "recording",
        "feature_set": "basic",
        "params": {},
        "blanked_copies": {"all": 1},
    }
    listings = basic_data.train.reset_index(drop=True)
    n_rows = len(listings)

    model = fit_variant("recording", settings, basic_data, seed=1, num_threads=1)

    seen = _RecordingModel.seen
    # The listings first and as they are, then their copies.
    assert len(seen["train"]) == 2 * n_rows
    pd.testing.assert_frame_equal(seen["train"].iloc[:n_rows], listings)
    assert seen["train"].iloc[n_rows:]["nr_seats"].isna().all()
    # Early stopping watches the validation rows as they are: no copy of them.
    pd.testing.assert_frame_equal(seen["validation"], basic_data.validation)
    # The training error is the model's on the listings, not on the frame with
    # their copies, where every copy would be priced at 9.
    training = model.metadata["training"]
    assert np.exp(9.0) > training["price_min_eur"]
    assert np.exp(10.0) < training["price_max_eur"]
    assert listings["nr_seats"].notna().mean() > 0.5, "or the two frames would score alike"
    priced = np.where(listings["nr_seats"].isna(), 9.0, 10.0)
    on_the_listings = float(np.mean(np.abs(priced - listings["log_price"].to_numpy())))
    assert training["train_l1_log_price"] == pytest.approx(on_the_listings)


def test_an_estimator_that_counts_rows_as_listings_takes_no_copies(basic_data):
    """The Ridge folds a level by its training rows, so a copy would count as a listing."""
    settings = {**train_variants()["b1"], "blanked_copies": {"all": 1}}

    with pytest.raises(ModelError, match="does not take them"):
        fit_variant("b1", settings, basic_data, seed=1, num_threads=1)


def test_blanked_copies_are_refused_without_the_request_contract(matrices: dict, supported_makes):
    """Which fields are optional is FR-01's to say, so a fit that is not told cannot guess."""
    data = train.read_matrices(matrices["features"], "basic", supported_makes)

    with pytest.raises(ModelError, match="request contract"):
        fit_variant("lgbm-basic", _lgbm_basic({"all": 1}), data, seed=1, num_threads=1)


@pytest.mark.parametrize(
    "blanked_copies",
    [{"al": 1}, {"all": -1}, {"all": 1.5}, {"random": "2"}, {"all": True}, [1, 1]],
)
def test_a_copy_count_that_is_not_a_whole_number_of_copies_is_refused(basic_data, blanked_copies):
    with pytest.raises(ModelError, match="blanked_copies"):
        fit_variant("lgbm-basic", _lgbm_basic(blanked_copies), basic_data, seed=1, num_threads=1)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def test_the_train_stage_fits_the_copies_its_params_file_names(
    matrices: dict, params: dict, tmp_path, monkeypatch
):
    """`params.yaml` to bundle, through the real `main`: the stage hands the contract over."""
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    variant = "lgbm-basic"
    committed = params["train"]["variants"][variant]
    variants = {
        **params["train"]["variants"],
        variant: {**committed, "blanked_copies": {"all": 2, "random": 1}},
    }
    params_path = params_override(tmp_path, params, train={"variants": variants})

    train.main(
        variant, matrices["features"], tmp_path / "models", params_path, tmp_path / "emissions"
    )

    record = json.loads((tmp_path / "models" / variant / MODEL_FILE).read_text(encoding="utf-8"))
    assert record["blanked_copies"] == {"all": 2, "random": 1, "fields": list(BASIC_OPTIONAL)}
    assert record["training"]["n_fit_rows"] == 4 * record["training"]["n_train_rows"]
