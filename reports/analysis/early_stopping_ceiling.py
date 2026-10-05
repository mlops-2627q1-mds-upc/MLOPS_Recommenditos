"""Where early stopping lands when `n_estimators` does not bind it (EDN-70, issue #64).

At `n_estimators: 1000` both LightGBM variants ran out of budget before early stopping fired, so
the budget, not the validation curve, decided how many trees they kept. This script fits each
LightGBM variant once with the budget raised far past anything the curve could plausibly need,
every other setting exactly as `params.yaml` has it, and reports:

- the round early stopping chose, the rounds the fit ran, and whether early stopping ended it;
- the validation curve at fixed checkpoints, as L1 in log space (what early stopping reads) and as
  MdAPE in euros, so "what the extra trees are worth" has a number before any test row is read;
- what the trees cost: fit seconds, seconds per 1,000 trees, the `booster.txt` size, and the
  single-row latency of a prediction and of the native SHAP export FR-08 serves, at the
  checkpoints, because both scale with the number of trees.

It reads the `train` and `validation` matrices only, through the `train` stage's own
`read_matrices`, and fits through `fit_variant`, so the measured fit is the stage's fit. The test
split is never opened: the budget is chosen on this output, and choosing it on test rows would
leak them into the model the gate is then asked to judge.

Run from the repository root after `dvc pull` (or a `dvc repro` up to `features`), with tracking
off, because nothing here belongs in the ladder's MLflow experiment:

    MLFLOW_TRACKING_URI= uv run python reports/analysis/early_stopping_ceiling.py

The fits run at `train.num_threads`, as the stage does, so the seconds are comparable with the
stage's `fit_seconds`; like every fit time in this project they move with whatever else holds a
core (EDN-53), and the ratios within one run are the measurement.
"""

from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from recommenditos.data.build_features import read_supported_makes
from recommenditos.modeling.model import fit_variant
from recommenditos.modeling.train import read_matrices
from recommenditos.pipeline import load_params

FEATURES = Path("data/processed/features")

#: Far past anything the curve could need: 20 times the budget that was binding. If early stopping
#: has not fired by here, the fit says so and the answer is "the curve never flattens", which would
#: be a finding of its own.
EXPLORATORY_BUDGET = 20_000

#: The validation curve is read at these tree counts, plus the round early stopping chose.
CHECKPOINTS = (250, 500, 1_000, 1_500, 2_000, 3_000, 4_000, 5_000, 6_000, 8_000, 10_000)

#: Single-row latency: the median of this many calls, after one call to warm the booster.
LATENCY_CALLS = 200

params = load_params(Path("params.yaml"))
seed = params["seed"]
num_threads = params["train"]["num_threads"]
supported = read_supported_makes(FEATURES.parent)
print(f"seed {seed}, num_threads {num_threads}, exploratory budget {EXPLORATORY_BUDGET:,}")


def curve_point(model, coded: pd.DataFrame, frame: pd.DataFrame, trees: int) -> tuple[float, float]:
    """Validation L1 in log space and MdAPE in euros, from the first `trees` trees.

    Bounded to the training price range exactly as `predict_eur` bounds the full model, so the
    point at the round early stopping chose reproduces the stage's own `validation_l1_log_price`.
    """
    log_price = np.clip(model.booster.predict(coded, num_iteration=trees), *model._log_bounds)
    l1 = float(np.mean(np.abs(log_price - frame["log_price"].to_numpy(dtype="float64"))))
    actual = frame["price"].to_numpy(dtype="float64")
    mdape = float(np.median(np.abs(np.exp(log_price) - actual) / np.abs(actual)))
    return l1, mdape


def latency_ms(model, row: pd.DataFrame, trees: int, *, contributions: bool) -> float:
    model.booster.predict(row, num_iteration=trees, pred_contrib=contributions)
    timings = []
    for _ in range(LATENCY_CALLS):
        started = time.perf_counter()
        model.booster.predict(row, num_iteration=trees, pred_contrib=contributions)
        timings.append(time.perf_counter() - started)
    return 1_000 * float(np.median(timings))


for variant, settings in params["train"]["variants"].items():
    if settings["estimator"] != "lightgbm":
        continue
    committed = settings["params"]
    explored = {**settings, "params": {**committed, "n_estimators": EXPLORATORY_BUDGET}}
    data = read_matrices(FEATURES, settings["feature_set"], supported)

    started = time.perf_counter()
    model = fit_variant(variant, explored, data, seed=seed, num_threads=num_threads)
    fit_seconds = time.perf_counter() - started
    training = model.metadata["training"]
    best = training["best_iteration"]

    with tempfile.TemporaryDirectory() as scratch:
        model.save(Path(scratch))
        booster_bytes = (Path(scratch) / "booster.txt").stat().st_size

    print(f"\n== {variant} ({settings['feature_set']}, {len(model.features)} features)")
    print(f"  params as committed: {committed}")
    print(f"  train rows {training['n_train_rows']:,}, validation rows {training['n_validation_rows']:,}")
    print(
        f"  early stopping chose round {best:,}; the fit ran {training['boosting_rounds']:,} "
        f"rounds; early stopping ended it: {training['early_stopped']}"
    )
    print(
        f"  fit {fit_seconds:.1f} s for {training['boosting_rounds']:,} rounds = "
        f"{1_000 * fit_seconds / training['boosting_rounds']:.2f} s per 1,000 rounds"
    )
    print(
        f"  booster.txt {booster_bytes / 1e6:.1f} MB for {best:,} trees = "
        f"{booster_bytes / 1e6 / best * 1_000:.2f} MB per 1,000 trees"
    )
    print(
        f"  validation L1(log price) as the stage records it: "
        f"{training['validation_l1_log_price']:.5f}"
    )

    coded = model._as_codes(model._align(data.validation))
    row = coded.iloc[[0]]
    print("  trees    val L1(log)   val MdAPE   predict 1 row   + SHAP 1 row")
    for trees in sorted({*(point for point in CHECKPOINTS if point < best), best}):
        l1, mdape = curve_point(model, coded, data.validation, trees)
        plain = latency_ms(model, row, trees, contributions=False)
        shap = latency_ms(model, row, trees, contributions=True)
        marker = "  <- early stopping" if trees == best else ""
        print(
            f"  {trees:>6,}   {l1:.5f}      {mdape:.4%}     {plain:7.2f} ms      "
            f"{shap:7.2f} ms{marker}"
        )
