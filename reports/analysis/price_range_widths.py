"""How wide the ranges of `/v1/price-range` are on the test split, by how much a request says (#139).

FR-02 lets a UC2 request leave out every field but `make`, and the range should narrow as fields are
added. This script measures that with the function the endpoint answers by
(`recommenditos.api.serving.price_range`): the typical price of FR-07's option C, the conformal
bounds, and their widening to contain the typical price. It reads the committed `models/lgbm-basic`
bundle and the `basic` test matrix and changes nothing.

Scenarios, on every test row (the matrix holds only the served makes), each the same rows with fields
left out by `Model.mask_absent`:
- make only: the smallest request FR-02 accepts;
- P1 (SC-05): make, model, registration date and mileage;
- required (FR-01): the ten fields `/v1/predict` requires, so the typical price is the point model's;
- full: every field as listed.

Per scenario: the mean and the median relative width, `(upper - lower) / typical price`; the
coverage of the served (widened) range; and the share of rows whose bounds were widened.

Run: `uv run python reports/analysis/price_range_widths.py`, with the log lines removed, gives
`price_range_widths_results.txt` (2026-10-20).
"""

import numpy as np

from recommenditos.api.serving import price_range
from recommenditos.config import MODELS_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import FeatureSpace
from recommenditos.modeling.model import (
    RequestContract,
    input_field_columns,
    load_model,
    price_range_seam_fields,
    seam_names,
)
from recommenditos.pipeline import load_params, read_frame

VARIANT = "lgbm-basic"
SPLIT = "test"


def main():
    params = load_params(PARAMS_FILE)
    model = load_model(MODELS_DIR / VARIANT)
    space = FeatureSpace.load(
        PROCESSED_DATA_DIR / "features" / model.feature_set, name=f"features-{model.feature_set}"
    )
    frame = read_frame(
        PROCESSED_DATA_DIR / "features" / model.feature_set / f"{SPLIT}.parquet", space.schema
    )
    request = RequestContract.from_params(params)
    fields = input_field_columns(model, request)
    seam = seam_names(fields, request)

    def leave_out_all_but(given):
        return tuple(seam[field] for field in fields if field not in given)

    # name: (fields left out, whether the request carries every field FR-01 requires)
    scenarios = {
        "make only": (price_range_seam_fields(model, request), False),
        "P1 (SC-05)": (leave_out_all_but(set(params["evaluate"]["sc05_p1_fields"])), False),
        "required (FR-01)": (leave_out_all_but(set(request.required_fields)), True),
        "full": ((), True),
    }
    price = frame["price"].to_numpy()
    print(f"{VARIANT}, {SPLIT} split, {len(frame):,} rows")
    print()
    print(
        f"{'scenario':<18} {'typical price':>14} {'mean width':>11} {'median width':>13}"
        f" {'coverage':>9} {'widened':>8}"
    )
    for name, (left_out, complete) in scenarios.items():
        masked = model.mask_absent(frame, left_out) if left_out else frame
        interval = model.predict_interval_eur(masked)
        answer = price_range(model, masked, complete=complete)
        lower, upper = answer["lower_eur"].to_numpy(), answer["upper_eur"].to_numpy()
        width = (upper - lower) / answer["price_eur"].to_numpy()
        widened = (lower < interval["lower_eur"].to_numpy()) | (
            upper > interval["upper_eur"].to_numpy()
        )
        print(
            f"{name:<18} {'point' if complete else 'median':>14} {np.mean(width):>11.3f}"
            f" {np.median(width):>13.3f} {np.mean((lower <= price) & (price <= upper)):>9.2%}"
            f" {np.mean(widened):>8.2%}"
        )
    print()
    print("typical price: FR-07's option C, the point model's with every FR-01 field, else the median.")
    print("width: (upper - lower) / typical price of the served range, widening included.")
    print("coverage: share of test prices inside the served range; widened: rows whose bounds moved.")


if __name__ == "__main__":
    main()
