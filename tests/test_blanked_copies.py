"""Training on blanked copies of the listings (EDN-82).

A variant whose `params.yaml` block names `blanked_copies` is fitted on its
training rows plus copies of them with optional input fields set to what an
omitted field is, so that a request carrying few optional fields is something
the model has seen. These tests hold the three places that behaviour lives in:
the seam's row-wise `mask_absent`, the copies themselves, and the fit and the
stage that use them.
"""

import json

import numpy as np
import pandas as pd
import pytest
from tests.conftest import params_override, project_params, train_variants

from recommenditos.modeling import train
from recommenditos.modeling.model import (
    MODEL_FILE,
    ModelError,
    RequestContract,
    fit_variant,
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
    variants = {
        **params["train"]["variants"],
        "b1": {**params["train"]["variants"]["b1"], "blanked_copies": {"all": 2, "random": 1}},
    }
    params_path = params_override(tmp_path, params, train={"variants": variants})

    train.main(
        "b1", matrices["features"], tmp_path / "models", params_path, tmp_path / "emissions"
    )

    record = json.loads((tmp_path / "models" / "b1" / MODEL_FILE).read_text(encoding="utf-8"))
    assert record["blanked_copies"] == {"all": 2, "random": 1, "fields": list(BASIC_OPTIONAL)}
    assert record["training"]["n_fit_rows"] == 4 * record["training"]["n_train_rows"]
