"""UC2's price range: the quantile models, their calibration and how they are served (EDN-85).

The bundle of a variant that names `intervals` carries three quantile boosters
and the two conformity offsets MAPIE computed on the `calibration` split, and
`Model.predict_interval_eur` adds them up without MAPIE. These tests hold the
three claims that design rests on: the served range is MAPIE's range, the
calibration reads the `calibration` split and no other, and a bundle without a
price range still loads and says it has none.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import project_params, train_variants

from recommenditos.config import PARAMS_FILE
from recommenditos.modeling import intervals, train
from recommenditos.modeling.evaluate import (
    P1_SCENARIO,
    input_field_columns,
    interval_metrics,
    sc05_scenarios,
)
from recommenditos.modeling.model import (
    INTERVAL_COLUMNS,
    ModelError,
    RequestContract,
    fit_quantile_estimators,
    interval_settings,
    load_model,
    price_range_seam_fields,
)
from recommenditos.tracking import REQUIRED_ENV_VARS

#: The variant of the real params.yaml that carries the price range.
INTERVAL_VARIANT = "lgbm-basic"


def _interval_variants() -> list[str]:
    return [
        name
        for name, settings in train_variants().items()
        if interval_settings(name, settings) is not None
    ]


def test_the_candidate_is_the_one_variant_with_a_price_range():
    # FR-07 serves the point estimate of `lgbm-basic` beside the range, so the
    # range belongs to that bundle. Pinned so that moving it is a decision.
    assert _interval_variants() == [INTERVAL_VARIANT]


@pytest.mark.req("FR-07")
@pytest.mark.parametrize("masked", [False, True], ids=["full", "P1"])
def test_the_served_range_equals_mapies_on_the_same_rows(
    trained: dict, matrices: dict, supported_makes: tuple, masked: bool
):
    """The bundle on disk, read without MAPIE, gives MAPIE's interval to float rounding.

    MAPIE is run here from scratch on the same training, validation and
    calibration rows the stage used: the quantile models are refitted (the fit
    is deterministic) and conformalized by `ConformalizedQuantileRegressor`, and
    its `predict_interval` and `predict` are compared with what the loaded
    bundle serves, for complete rows and for P1.
    """
    params = project_params()
    served = trained["models"][INTERVAL_VARIANT]
    settings = params["train"]["variants"][INTERVAL_VARIANT]
    data = train.read_matrices(
        matrices["features"],
        settings["feature_set"],
        supported_makes,
        request=RequestContract.from_params(params),
    )
    calibration = train.read_calibration(
        matrices["features"], settings["feature_set"], supported_makes
    )
    estimators, record = fit_quantile_estimators(
        served,
        data,
        interval_settings(INTERVAL_VARIANT, settings),
        seed=params["seed"],
        num_threads=params["train"]["num_threads"],
    )
    rows = intervals.calibration_rows(served, calibration, record, seed=params["seed"])
    mapie = intervals.conformalizer(estimators, record["confidence_level"])
    mapie.conformalize(served._as_codes(served._align(rows)), rows["log_price"].to_numpy())

    test = pd.read_parquet(matrices["features"] / settings["feature_set"] / "test.parquet")
    if masked:
        p1 = sc05_scenarios(input_field_columns(served, params), params)[0]
        test = served.mask_absent(test, p1.fields)
    features = served._as_codes(served._align(test))
    median, bounds = mapie.predict_interval(features)

    interval = served.predict_interval_eur(test)
    np.testing.assert_allclose(interval["lower_eur"], np.exp(bounds[:, 0, 0]), rtol=1e-12)
    np.testing.assert_allclose(interval["upper_eur"], np.exp(bounds[:, 1, 0]), rtol=1e-12)
    np.testing.assert_allclose(interval["median_eur"], np.exp(median), rtol=1e-12)
    assert list(interval.columns) == list(INTERVAL_COLUMNS.values())
    assert interval.index.equals(test.index)


@pytest.mark.req("NFR-01")
@pytest.mark.parametrize("variant", train_variants())
def test_only_a_price_range_reads_the_calibration_split_and_never_test(
    variant: str, matrices: dict, tmp_path: Path, monkeypatch
):
    """EDN-03 and the evaluation protocol: `calibration` calibrates the range and nothing else.

    Every frame `train` opens is recorded. The point models read `train` and
    `validation`; the variant with a price range reads `calibration` on top, and
    no variant reads `test` or the `ES` holdout.
    """
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    opened: list[str] = []
    read = train.read_frame

    def recording(path, *args, **kwargs):
        opened.append(Path(path).stem)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(train, "read_frame", recording)
    train.main(
        variant, matrices["features"], tmp_path / "models", PARAMS_FILE, tmp_path / "emissions"
    )

    expected = {"train", "validation"}
    if variant in _interval_variants():
        expected.add(intervals.CALIBRATION_SPLIT)
    assert set(opened) == expected
    model = load_model(tmp_path / "models" / variant)
    assert model.has_intervals is (variant in _interval_variants())
    if model.has_intervals:
        calibration = model.metadata["intervals"]["calibration"]
        listings = len(
            pd.read_parquet(matrices["features"] / model.feature_set / "calibration.parquet")
        )
        copies = model.metadata["intervals"]["blanked_copies"]
        assert calibration["split"] == "calibration"
        assert calibration["n_listings"] == listings
        assert calibration["n_rows"] == listings * (1 + copies["all"] + copies["random"])


def test_a_bundle_without_a_price_range_refuses_to_serve_one(trained: dict, test_frames: dict):
    for variant, model in trained["models"].items():
        if variant in _interval_variants():
            continue
        assert not model.has_intervals, variant
        with pytest.raises(ModelError, match="carries no price-range models"):
            model.predict_interval_eur(test_frames[model.feature_set].head(3))
        assert interval_metrics(model, {"full": test_frames[model.feature_set]}) is None


def test_the_range_is_ordered_and_an_empty_frame_gives_an_empty_one(
    trained: dict, test_frames: dict
):
    model = trained["models"][INTERVAL_VARIANT]
    frame = test_frames[model.feature_set]
    interval = model.predict_interval_eur(frame)
    assert np.isfinite(interval.to_numpy()).all()
    assert (interval["lower_eur"] > 0).all()
    assert (interval["lower_eur"] <= interval["upper_eur"]).all()

    empty = model.predict_interval_eur(frame.head(0))
    assert empty.empty
    assert list(empty.columns) == list(INTERVAL_COLUMNS.values())


def test_sc05_measures_both_scenarios_for_the_range(trained: dict, test_frames: dict):
    model = trained["models"][INTERVAL_VARIANT]
    test = test_frames[model.feature_set]
    measured = interval_metrics(model, {"full": test, P1_SCENARIO: test})
    assert set(measured) == {"full", P1_SCENARIO}
    for entry in measured.values():
        assert 0.0 <= entry["coverage"] <= 1.0
        assert entry["mean_relative_width"] > 0


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"confidence_level": 0.9}, "does not set blanked_copies"),
        ({"confidence_level": 1.5, "blanked_copies": {}}, "not a share in"),
        ({"confidence_level": 0.9, "blanked_copies": {}, "alpha": 0.1}, "sets alpha"),
        ({"confidence_level": 0.9, "blanked_copies": {"all": -1}}, "not a whole number"),
    ],
)
def test_an_intervals_block_is_refused_when_it_cannot_be_read(block: dict, message: str):
    settings = {"estimator": "lightgbm", "intervals": block}
    with pytest.raises(ModelError, match=message):
        interval_settings("v", settings)


def test_only_lightgbm_fits_the_quantile_models():
    settings = {
        "estimator": "ridge",
        "intervals": {"confidence_level": 0.9, "blanked_copies": {"all": 1}},
    }
    with pytest.raises(ModelError, match="only 'lightgbm' fits the quantile models"):
        interval_settings("b1", settings)


def test_the_price_range_may_omit_every_field_but_make(trained: dict):
    model = trained["models"][INTERVAL_VARIANT]
    request = RequestContract.from_params(project_params())
    fields = price_range_seam_fields(model, request)
    assert request.price_range_required_fields == ("make",)
    assert "make" not in fields
    assert set(fields) | {"make"} == set(model.features)
