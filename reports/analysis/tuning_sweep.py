"""How much `learning_rate` and `num_leaves` could still buy, read on validation only (EDN-73).

The evidence behind the decision not to tune them (issue #64), not a tuning run: a small, bounded
grid around the committed values of the candidate `lgbm-basic` (EDN-62), each fit with
`n_estimators` far past anything the curve needs, so early stopping decides every one and no
point of the grid is judged at a budget that binds it. Everything else is as `params.yaml` has it.

For each point it reports the round early stopping chose, the fit time, and the validation L1 in
log space (what early stopping reads and the natural selection metric) and validation MdAPE (what
the gate reads, on validation rows). The last block asks the question the decision turns on:
whether the validation split can tell any point from the committed one at all. It is a paired
bootstrap of the difference in validation L1 and in validation MdAPE, resampled by seller group
because the split is grouped by seller (EDN-14), so rows of one dealer move together, with the same
draws for every point so the comparisons share their noise.

It reads the `train` and `validation` matrices only, through the `train` stage's own
`read_matrices`, and fits through `fit_variant`. The test split is never opened, so nothing here
can make a later test number optimistic.

Run from the repository root after `dvc pull`, with tracking off:

    MLFLOW_TRACKING_URI= uv run python reports/analysis/tuning_sweep.py
"""

from itertools import product
from pathlib import Path
import time

import numpy as np

from recommenditos.data.build_features import read_supported_makes
from recommenditos.modeling.model import fit_variant
from recommenditos.modeling.train import read_matrices
from recommenditos.pipeline import load_params, read_frame
from recommenditos.schema import PROCESSED_SCHEMA

FEATURES = Path("data/processed/features")
VARIANT = "lgbm-basic"
EXPLORATORY_BUDGET = 20_000
LEARNING_RATES = (0.025, 0.05, 0.1)
NUM_LEAVES = (31, 63, 127, 255)
BOOTSTRAP_DRAWS = 2_000

params = load_params(Path("params.yaml"))
seed = params["seed"]
num_threads = params["train"]["num_threads"]
settings = params["train"]["variants"][VARIANT]
supported = read_supported_makes(FEATURES.parent)
data = read_matrices(FEATURES, settings["feature_set"], supported)
print(f"{VARIANT}: committed params {settings['params']}")
print(f"seed {seed}, num_threads {num_threads}, exploratory budget {EXPLORATORY_BUDGET:,}")
print(f"grid: learning_rate {LEARNING_RATES} x num_leaves {NUM_LEAVES}")

# The seller groups of the validation rows, to resample them as the split drew them. The matrix
# does not carry `seller_group_id`, so it is joined back from the split frame `features` read: the
# matrix keeps that frame's served rows in their order, which the row count checks.
validation_frame = read_frame(FEATURES.parent / "validation.parquet", PROCESSED_SCHEMA)
served = validation_frame[validation_frame["make"].isin(supported)].reset_index(drop=True)
assert len(served) == len(data.validation), (len(served), len(data.validation))
assert np.allclose(served["price"].to_numpy(), data.validation["price"].to_numpy())
groups = served["seller_group_id"].to_numpy()

log_price = data.validation["log_price"].to_numpy(dtype="float64")
price = data.validation["price"].to_numpy(dtype="float64")
errors: dict[tuple[float, int], np.ndarray] = {}
relative: dict[tuple[float, int], np.ndarray] = {}

print("\nlearning_rate  num_leaves  best round  rounds run  fit s   val L1(log)  val MdAPE")
for learning_rate, num_leaves in product(LEARNING_RATES, NUM_LEAVES):
    point = {
        **settings,
        "params": {
            **settings["params"],
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "n_estimators": EXPLORATORY_BUDGET,
        },
    }
    started = time.perf_counter()
    model = fit_variant(VARIANT, point, data, seed=seed, num_threads=num_threads)
    fit_seconds = time.perf_counter() - started
    training = model.metadata["training"]
    predicted = model.predict_log_price(data.validation).to_numpy(dtype="float64")
    errors[(learning_rate, num_leaves)] = np.abs(predicted - log_price)
    relative[(learning_rate, num_leaves)] = np.abs(np.exp(predicted) - price) / price
    mdape = float(np.median(relative[(learning_rate, num_leaves)]))
    print(
        f"{learning_rate:>13}  {num_leaves:>10}  {training['best_iteration']:>10,}  "
        f"{training['boosting_rounds']:>10,}  {fit_seconds:>5.0f}   "
        f"{training['validation_l1_log_price']:.5f}      {mdape:.4%}"
        + ("" if training["early_stopped"] else "  (budget bound)")
    )

committed = (settings["params"]["learning_rate"], settings["params"]["num_leaves"])
by_l1 = min(errors, key=lambda key: errors[key].mean())
by_mdape = min(relative, key=lambda key: np.median(relative[key]))
print(f"\ncommitted point {committed}; lowest validation L1 at {by_l1}, MdAPE at {by_mdape}")

# Paired, by seller group: a draw resamples the groups with replacement, which gives every row
# the number of times its group was drawn as a weight, and every point is scored on the same
# weights.
rng = np.random.default_rng(seed)
unique, inverse = np.unique(groups, return_inverse=True)
print(f"validation rows {len(groups):,}, {len(unique):,} seller groups, {BOOTSTRAP_DRAWS:,} draws")
others = [key for key in sorted(errors) if key != committed]
l1_differences = {key: [] for key in others}
mdape_differences = {key: [] for key in others}
for _ in range(BOOTSTRAP_DRAWS):
    drawn = rng.integers(0, len(unique), size=len(unique))
    weight = np.bincount(drawn, minlength=len(unique))[inverse]
    l1 = {key: float((weight * errors[key]).sum() / weight.sum()) for key in errors}
    mdape = {key: float(np.median(np.repeat(relative[key], weight))) for key in relative}
    for key in others:
        l1_differences[key].append(l1[key] - l1[committed])
        mdape_differences[key].append(mdape[key] - mdape[committed])

print("point minus committed: the difference, its 95 % interval, the share of draws it is better")
for key in others:
    l1_draws, mdape_draws = np.array(l1_differences[key]), np.array(mdape_differences[key])
    l1_low, l1_high = np.percentile(l1_draws, [2.5, 97.5])
    md_low, md_high = np.percentile(mdape_draws, [2.5, 97.5])
    print(
        f"  {key!s:12s} L1 {errors[key].mean() - errors[committed].mean():+.5f} "
        f"[{l1_low:+.5f}, {l1_high:+.5f}] better {np.mean(l1_draws < 0):6.1%}   "
        f"MdAPE {np.median(relative[key]) - np.median(relative[committed]):+.3%} "
        f"[{md_low:+.3%}, {md_high:+.3%}] better {np.mean(mdape_draws < 0):6.1%}"
    )
