"""UC2's price range: the quantile models, their calibration and how they are served (EDN-85).

The bundle of a variant that names `intervals` carries three quantile boosters
and the two conformity offsets MAPIE computed on the `calibration` split, and
`Model.predict_interval_eur` adds them up without MAPIE. These tests hold the
three claims that design rests on: the served range is MAPIE's range, the
calibration reads the `calibration` split and no other, and a bundle without a
price range still loads and says it has none.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from tests.conftest import project_params, train_variants

from recommenditos.config import PARAMS_FILE
from recommenditos.modeling import intervals, train
from recommenditos.modeling.evaluate import (
    IN_BAND,
    NOT_JUDGED,
    OUTSIDE_BAND,
    P1_SCENARIO,
    POOLED_MAKE,
    input_field_columns,
    interval_make_rows,
    interval_metrics,
    interval_outcomes,
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
    with warnings.catch_warnings():
        # The warning `intervals.conformalize` filters for the same reason: MAPIE's
        # fitted-check predicts on a NumPy array of zeros, which has no column names.
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names", category=UserWarning
        )
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


def _outcome(coverage: float, n: int, *, width: float = 0.5) -> pd.DataFrame:
    """`n` rows of which the first `coverage * n` fall inside their interval."""
    covered = round(coverage * n)
    return pd.DataFrame(
        {"inside": [True] * covered + [False] * (n - covered), "relative_width": [width] * n}
    )


@pytest.mark.req("FR-07")
def test_the_per_make_table_reports_every_make_and_judges_only_the_large_ones():
    criteria = {"sc05_coverage_min": 0.88, "sc05_coverage_max": 0.92, "sc05_min_make_rows": 500}
    parts = {
        "A": _outcome(0.90, 1000),
        "B": _outcome(0.80, 1000),
        "C": _outcome(0.50, 10),
        "D": _outcome(0.50, 500),
        "E": _outcome(0.50, 499),
    }
    outcome = pd.concat(parts, ignore_index=True)
    makes = pd.Series([name for name, part in parts.items() for _ in range(len(part))])

    rows = interval_make_rows({"full": outcome}, makes, variant="v", criteria=criteria)

    assert [row["make"] for row in rows] == [POOLED_MAKE, "A", "B", "D", "E", "C"]
    by_make = {row["make"]: row for row in rows}
    assert by_make[POOLED_MAKE]["n"] == 3009
    assert by_make[POOLED_MAKE]["coverage"] == pytest.approx((900 + 800 + 5 + 250 + 250) / 3009)
    assert by_make["A"]["judgement"] == IN_BAND
    assert by_make["B"]["judgement"] == OUTSIDE_BAND
    # Reported with its count and not judged: ten rows cannot tell 50 % from 90 %.
    assert by_make["C"]["judgement"] == NOT_JUDGED
    # The boundary: exactly `sc05_min_make_rows` rows are judged, one fewer are not.
    assert by_make["D"]["judgement"] == OUTSIDE_BAND
    assert by_make["E"]["judgement"] == NOT_JUDGED
    assert by_make["C"]["n"] == 10
    assert by_make["C"]["coverage"] == 0.5
    # The standard error of a coverage at the middle of the band shrinks with n.
    # At the middle of the band, not at the make's own coverage: B is at 0.80.
    assert by_make["A"]["std_error"] == pytest.approx((0.9 * 0.1 / 1000) ** 0.5)
    assert by_make["B"]["std_error"] == by_make["A"]["std_error"]
    assert by_make["C"]["std_error"] > 10 * by_make["A"]["std_error"]
    assert sum(row["n"] for row in rows if row["make"] != POOLED_MAKE) == 3009


@pytest.mark.req("FR-07")
def test_the_per_make_table_cuts_each_scenario_by_the_makes_it_is_given():
    criteria = {"sc05_coverage_min": 0.88, "sc05_coverage_max": 0.92, "sc05_min_make_rows": 5}
    makes = pd.Series(["A"] * 10 + ["B"] * 10, index=range(20))
    outcomes = {
        "full": _outcome(0.9, 20, width=0.4),
        P1_SCENARIO: pd.concat([_outcome(1.0, 10, width=1.0), _outcome(0.5, 10, width=2.0)]),
    }
    outcomes[P1_SCENARIO].index = range(20)

    rows = interval_make_rows(outcomes, makes, variant="v", criteria=criteria)

    p1 = {r["make"]: r for r in rows if r["scenario"] == P1_SCENARIO}
    assert p1["A"]["coverage"] == 1.0 and p1["A"]["mean_relative_width"] == 1.0
    assert p1["B"]["coverage"] == 0.5 and p1["B"]["mean_relative_width"] == 2.0
    assert p1[POOLED_MAKE]["mean_relative_width"] == 1.5
    assert {r["scenario"] for r in rows} == {"full", P1_SCENARIO}


@pytest.mark.req("FR-07")
def test_the_pooled_row_of_the_per_make_table_is_the_figure_sc05_gates_on(
    trained: dict, test_frames: dict
):
    model = trained["models"][INTERVAL_VARIANT]
    test = test_frames[model.feature_set]
    frames = {"full": test, P1_SCENARIO: test}
    criteria = {"sc05_coverage_min": 0.88, "sc05_coverage_max": 0.92, "sc05_min_make_rows": 5}

    outcomes = interval_outcomes(model, frames)
    rows = interval_make_rows(outcomes, test["make"].astype("str"), variant="v", criteria=criteria)

    measured = interval_metrics(model, frames)
    for row in rows:
        if row["make"] == POOLED_MAKE:
            assert row["coverage"] == measured[row["scenario"]]["coverage"]
            assert row["mean_relative_width"] == measured[row["scenario"]]["mean_relative_width"]
            assert row["n"] == len(test)
    bare = next(each for each in trained["models"].values() if not each.has_intervals)
    assert interval_outcomes(bare, frames) is None


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
