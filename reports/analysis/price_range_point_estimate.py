"""Where the typical price of a partial `/price-range` request comes from (#137, open point).

FR-07 takes the typical price of a price range from the point model, `lgbm-basic`, and the bounds
from the quantile models, widening the bounds where the point lies outside them. `lgbm-basic` was
trained on copies of its listings with FR-01's *optional* fields left out (EDN-82), never with a
*required* one missing, while a UC2 request may leave out everything but `make` (FR-02). LightGBM
sends a value that is missing from a feature that had no missing values in training down the
branch of 0 (checked with LightGBM 4.7.0), so a make-only request is priced as a car with 0 km, a
power of 0 kW, and so on.

This script measures that, against the candidate the issue names: the 50 % quantile model MAPIE
requires beside the bounds, which is in the bundle and was trained with the same copies as the
bounds (every field but `make` left out in one, a random subset in the other). It reads the
committed `models/lgbm-basic` bundle and changes nothing.

The split is `calibration`, because neither model was fitted or early-stopped on it: the point
model and the quantile models stopped on `validation`, and the calibration split entered only the
conformity offsets of the lower and upper bounds, which neither typical price depends on. The test
split stays `evaluate`'s.

Scenarios, on all calibration rows of the served makes, each the same rows with fields left out by
`Model.mask_absent`:
- full: every field as listed;
- required (FR-01): only the ten fields `/predict` requires;
- P1 (SC-05): only make, model, registration date and mileage;
- make only: the smallest request FR-02 accepts.

Per scenario: the MdAPE of each typical price, its difference (median minus point) with a 95 %
interval from 2,000 paired bootstrap draws over seller groups (`tune.paired_bootstrap`); how often
the two typical prices differ from each other for the same row (what a user comparing `/predict`
with `/price-range` would see); how often each typical price lies outside the conformal interval, which is how often FR-07's widening would
change the bounds; and the mean relative width of the interval with and without that widening.

Run: `uv run python reports/analysis/price_range_point_estimate.py`, with the log lines removed, gives `price_range_point_estimate_results.txt` (2026-10-20).
"""

from pathlib import Path

import numpy as np

from recommenditos.config import MODELS_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import FeatureSpace
from recommenditos.modeling import evaluate, tune
from recommenditos.modeling.model import (
    RequestContract,
    input_field_columns,
    load_model,
    price_range_seam_fields,
    seam_names,
)
from recommenditos.pipeline import load_params, read_frame
from recommenditos.schema import PROCESSED_SCHEMA

VARIANT = "lgbm-basic"
SPLIT = "calibration"


def seller_groups(features, supported):
    """The seller group of every calibration row, joined back as `tune` joins validation's."""
    frame = read_frame(PROCESSED_DATA_DIR / f"{SPLIT}.parquet", PROCESSED_SCHEMA)
    served = frame[frame["make"].isin(supported)].reset_index(drop=True)
    assert len(served) == len(features)
    assert np.allclose(served["price"].to_numpy(), features["price"].to_numpy())
    return served["seller_group_id"].to_numpy()


def main():
    params = load_params(PARAMS_FILE)
    model = load_model(MODELS_DIR / VARIANT)
    assert model.has_intervals
    space = FeatureSpace.load(
        PROCESSED_DATA_DIR / "features" / model.feature_set, name=f"features-{model.feature_set}"
    )
    frame = read_frame(
        PROCESSED_DATA_DIR / "features" / model.feature_set / f"{SPLIT}.parquet", space.schema
    )
    supported = tuple(model.metadata["training"]["supported_makes"])
    groups = seller_groups(frame, supported)
    request = RequestContract.from_params(params)
    fields = input_field_columns(model, request)
    seam = seam_names(fields, request)
    every = price_range_seam_fields(model, request)

    def leave_out_all_but(given):
        return tuple(seam[field] for field in fields if field not in given)

    scenarios = {
        "full": (),
        "required (FR-01)": leave_out_all_but(set(request.required_fields)),
        "P1 (SC-05)": leave_out_all_but(set(params["evaluate"]["sc05_p1_fields"])),
        "make only": every,
    }
    price = frame["price"].to_numpy()
    print(f"{VARIANT}, {SPLIT} split, {len(frame):,} rows, {len(np.unique(groups)):,} seller groups")
    print(f"conformal offsets (log price): {model.metadata['intervals']['offsets']}")
    print()
    header = (
        f"{'scenario':<18} {'MdAPE point':>12} {'MdAPE median':>13} {'median - point (95 % CI)':>30}"
        f" {'gap':>6} {'point out':>10} {'median out':>11} {'coverage':>9} {'width':>7} {'width+widen':>12}"
    )
    print(header)
    for name, left_out in scenarios.items():
        masked = model.mask_absent(frame, left_out) if left_out else frame
        point = model.predict_eur(masked).to_numpy()
        interval = model.predict_interval_eur(masked)
        lower, upper = interval["lower_eur"].to_numpy(), interval["upper_eur"].to_numpy()
        median = interval["median_eur"].to_numpy()
        errors = {
            "point": np.abs(point - price) / price,
            "median": np.abs(median - price) / price,
        }
        _, draws = tune.paired_bootstrap(
            errors, "point", tune.median_error, tune.Bootstrap(groups=groups, seed=params["seed"])
        )
        low, high = np.percentile(draws[:, 0], [2.5, 97.5])
        difference = np.median(errors["median"]) - np.median(errors["point"])
        widened = (np.minimum(lower, point), np.maximum(upper, point))
        print(
            f"{name:<18} {np.median(errors['point']):>12.2%} {np.median(errors['median']):>13.2%}"
            f" {difference * 100:>+10.2f} pp [{low * 100:+.2f}, {high * 100:+.2f}]"
            f" {np.median(np.abs(median / point - 1)):>6.2%}"
            f" {np.mean((point < lower) | (point > upper)):>10.2%}"
            f" {np.mean((median < lower) | (median > upper)):>11.2%}"
            f" {np.mean((lower <= price) & (price <= upper)):>9.2%}"
            f" {np.mean((upper - lower) / median):>7.3f}"
            f" {np.mean((widened[1] - widened[0]) / point):>12.3f}"
        )
    print()
    print("point: lgbm-basic's predict_eur; median: the 50 % quantile model of the price range.")
    print("gap: median over rows of |median / point - 1|, how far the two typical prices differ.")
    print("out: share of rows whose typical price lies outside the conformal interval.")
    print("width: mean (upper - lower) / median; width+widen: FR-07's widened bounds over the point.")
    print("coverage on the calibration split is in-sample for the offsets; SC-05 is measured on test.")


if __name__ == "__main__":
    main()
