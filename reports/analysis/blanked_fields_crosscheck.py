"""Two checks of `blanked_fields.py` by another route, written after its run (EDN-82).

Not part of that script's protocol, and neither decides anything.

1. How many training rows already look like a required-only request: every optional field in the
   state `Model.mask_absent` gives an omitted one. This is the premise of the try-out.
2. What a required-only request can reach at all. The run puts the blanked configurations at 7.1 %
   to 7.4 % validation MdAPE on it, against 8.57 % for `lgbm-basic` as committed. Here the same
   request is reached without any blanking code: one LightGBM fitted on the ten required columns
   alone, with the committed hyperparameters and early stopping on the validation split. If the
   two agree, the 8.57 % is what `lgbm-basic` loses by never having been trained for the request,
   not what the missing fields are worth.

It reads the `train` and `validation` matrices only, fits one model, logs nothing to MLflow and
writes nothing. Run from the repository root after `dvc pull`; its output is appended to
`blanked_fields_results.txt`:

    MLFLOW_TRACKING_URI= uv run python reports/analysis/blanked_fields_crosscheck.py
"""

import sys

import lightgbm
from loguru import logger
import numpy as np

from recommenditos.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommenditos.modeling import evaluate
from recommenditos.modeling.model import load_model
from recommenditos.pipeline import load_params, read_frame

logger.remove()
logger.add(sys.stderr, level="WARNING")

params = load_params()
settings = params["train"]["variants"]["lgbm-basic"]["params"]
derived = params["evaluate"]["input_field_to_feature"]
required = [derived.get(field, field) for field in params["evaluate"]["required_input_fields"]]

# 1. Training rows in which every optional field is already what an omitted one would be.
for variant in ("lgbm-basic", "lgbm-extended"):
    bundle = load_model(MODELS_DIR / variant)
    frame = read_frame(
        PROCESSED_DATA_DIR / "features" / bundle.feature_set / "train.parquet", bundle.space.schema
    )
    fields = evaluate.input_field_columns(bundle, params)
    optional = evaluate.optional_input_fields(fields, params)
    absent = bundle.mask_absent(frame, optional)
    columns = [column for field in optional for column in fields[field]]
    same = np.column_stack(
        [
            ((frame[name] == absent[name]) | (frame[name].isna() & absent[name].isna())).to_numpy(
                dtype=bool, na_value=False
            )
            for name in columns
        ]
    )
    print(
        f"{variant}: {int(same.all(axis=1).sum()):,} of {len(frame):,} training rows have all "
        f"{len(optional)} optional fields as an omitted one"
    )

# 2. The required-only specialist.
# The committed bundle is used for its contract only: the column order and the category codes.
contract = load_model(MODELS_DIR / "lgbm-basic")
directory = PROCESSED_DATA_DIR / "features" / "basic"
train = read_frame(directory / "train.parquet", contract.space.schema)
validation = read_frame(directory / "validation.parquet", contract.space.schema)

estimator = lightgbm.LGBMRegressor(
    objective="regression",
    metric="l1",
    first_metric_only=True,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
    random_state=params["seed"],
    n_jobs=params["train"]["num_threads"],
    learning_rate=settings["learning_rate"],
    num_leaves=settings["num_leaves"],
    n_estimators=settings["n_estimators"],
)
estimator.fit(
    contract._as_codes(contract._align(train))[required],
    train["log_price"].to_numpy(dtype="float64"),
    eval_X=contract._as_codes(contract._align(validation))[required],
    eval_y=validation["log_price"].to_numpy(dtype="float64"),
    categorical_feature=[name for name in required if name in contract.categorical_columns],
    callbacks=[
        lightgbm.early_stopping(settings["early_stopping_rounds"], first_metric_only=True, verbose=False)
    ],
)
predicted = np.exp(estimator.predict(contract._as_codes(contract._align(validation))[required]))
price = validation["price"].to_numpy(dtype="float64")
print(
    f"required-only specialist: {len(required)} columns, {len(train):,} train rows, best round "
    f"{estimator.best_iteration_:,}, validation MdAPE {np.median(np.abs(predicted - price) / price) * 100:.2f} %, "
    f"L1 of the log price {np.mean(np.abs(np.log(predicted) - validation['log_price'].to_numpy())):.4f}"
)
