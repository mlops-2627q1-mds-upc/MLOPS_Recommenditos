"""UC2's price range: the quantile models, calibrated with MAPIE (EDN-85).

Conformalized quantile regression (Romano et al., 2019) on three LightGBM
quantile models, fitted in `model.py` at 5 %, 95 % and 50 % on the training rows
and on copies of them with fields left out. MAPIE's
`ConformalizedQuantileRegressor` then scores the lower and upper models on the
`calibration` split, with copies drawn the same way, and turns the scores into
two constants: how far to move each bound so that a nominal 90 % of the
calibration rows fall inside. The coverage guarantee is marginal, over the mix of
complete and partial requests the copies make up, not per pattern of missing
fields; `evaluate` measures it for full inputs and for P1 (SC-05).

MAPIE is a `pipeline` package and this module is the only one that imports it.
The bundle stores the boosters and the two offsets, and `Model.predict_interval_eur`
adds them up at serving time, so the API image needs no MAPIE (EDN-85, the
pattern EDN-11 set for SHAP). The offsets are read off MAPIE's own prediction on
the calibration rows rather than recomputed from its conformity scores, so the
quantile of the scores is MAPIE's computation and nothing here reimplements it.

The `calibration` split is read by this step alone: never by the point model,
never by early stopping (EDN-03 keeps `ES` out of it, and `split` keeps every
seller in one split).
"""

from collections.abc import Mapping
from importlib.metadata import version as distribution_version
import time
import warnings

from mapie.regression import ConformalizedQuantileRegressor
import numpy as np
import pandas as pd

from recommenditos.modeling.model import (
    BLANKED_COPY_KINDS,
    INTERVAL_QUANTILES,
    INTERVAL_STREAMS,
    LightGBMModel,
    Model,
    ModelError,
    TrainingData,
    fit_quantile_estimators,
    fit_variant,
    interval_settings,
    with_blanked_copies_drawn,
)

#: The split the conformity scores are computed on, and no other.
CALIBRATION_SPLIT = "calibration"

#: How far the offsets read off MAPIE's bounds may vary between calibration
#: rows, in log price. They are constants, so anything beyond float rounding
#: means MAPIE computed something other than what serving replays.
_OFFSET_TOLERANCE = 1e-9


def fit_with_intervals(
    variant: str, settings: dict, data: TrainingData, *, seed: int, num_threads: int
) -> Model:
    """`fit_variant`, plus the calibrated price range when the variant names `intervals`.

    The one call `train` measures, so the energy and the time NFR-10 bounds cover
    the point model, the quantile models and the calibration together.
    """
    model = fit_variant(variant, settings, data, seed=seed, num_threads=num_threads)
    chosen = interval_settings(variant, settings)
    if chosen is None:
        return model
    if data.calibration is None:
        raise ModelError(
            f"variant {variant!r} names `intervals`, and their calibration needs the "
            f"{CALIBRATION_SPLIT!r} split, which the fit was not given."
        )
    if not isinstance(model, LightGBMModel):  # interval_settings refuses anything else
        raise TypeError(f"{variant!r} is a {type(model).__name__}, not a LightGBMModel")
    # Timed apart from the point model, so the share of the measured fit the price
    # range takes can be stated; CodeCarbon's figure covers both together.
    started = time.perf_counter()
    estimators, record = fit_quantile_estimators(
        model, data, chosen, seed=seed, num_threads=num_threads
    )
    record = conformalize(model, estimators, record, data.calibration, seed=seed)
    record["seconds"] = time.perf_counter() - started
    model.attach_intervals({name: each.booster_ for name, each in estimators.items()}, record)
    return model


def calibration_rows(
    model: Model, calibration: pd.DataFrame, record: Mapping, *, seed: int
) -> pd.DataFrame:
    """The calibration split plus copies with fields left out, as the training rows got.

    The same fields, kinds and counts of copies as the quantile models were
    fitted on, read from their record, and drawn from a stream of its own, so the
    calibration rows mix complete and partial requests in the proportions the
    models were trained for.
    """
    block = record["blanked_copies"]
    return with_blanked_copies_drawn(
        model,
        calibration,
        tuple(block["fields"]),
        copies={kind: block[kind] for kind in BLANKED_COPY_KINDS},
        rng=np.random.default_rng([seed, INTERVAL_STREAMS[CALIBRATION_SPLIT]]),
    )


def conformalizer(estimators: Mapping, confidence_level: float) -> ConformalizedQuantileRegressor:
    """MAPIE's CQR over the three fitted quantile models, in the order it requires."""
    return ConformalizedQuantileRegressor(
        [estimators[name] for name in INTERVAL_QUANTILES],
        confidence_level=confidence_level,
        prefit=True,
    )


def conformalize(
    model: LightGBMModel,
    estimators: Mapping,
    record: dict,
    calibration: pd.DataFrame,
    *,
    seed: int,
) -> dict:
    """`record` with the two conformity offsets MAPIE computed on `calibration`.

    `lower` is what MAPIE subtracts from the lower quantile and `upper` what it
    adds to the upper one, in log price; either may be negative, when the raw
    quantile models were too wide.
    """
    rows = calibration_rows(model, calibration, record, seed=seed)
    features = model._as_codes(model._align(rows))
    cqr = conformalizer(estimators, record["confidence_level"])
    with warnings.catch_warnings():
        # MAPIE hands the estimators a NumPy array of the same columns in the
        # same order, and scikit-learn warns that it carries no column names.
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        cqr.conformalize(features, rows["log_price"].to_numpy(dtype="float64"))
        _, bounds = cqr.predict_interval(features)
    lower_shift = estimators["lower"].predict(features) - bounds[:, 0, 0]
    upper_shift = bounds[:, 1, 0] - estimators["upper"].predict(features)
    for name, shift in (("lower", lower_shift), ("upper", upper_shift)):
        if np.ptp(shift) > _OFFSET_TOLERANCE:
            raise ModelError(
                f"MAPIE moved the {name} bound of {model.variant!r} by between "
                f"{shift.min():.6f} and {shift.max():.6f} in log price over the calibration "
                f"rows. Serving adds one constant per bound, so this MAPIE version computes "
                f"something serving does not replay."
            )
    return {
        **record,
        "offsets": {"lower": float(lower_shift.mean()), "upper": float(upper_shift.mean())},
        "calibration": {
            "split": CALIBRATION_SPLIT,
            "n_listings": len(calibration),
            "n_rows": len(rows),
            "mapie_version": distribution_version("mapie"),
        },
    }
