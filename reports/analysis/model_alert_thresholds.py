"""The thresholds of the model alerts ModelErrorHigh and ModelCoverageLow (#230).

Question: how far does a window's MdAPE and range coverage stray from the test
values by chance alone, when its requests come from the test distribution?
The alerts fire beyond the 0.005 one-sided tail, NFR-11's false-alarm level
per window (docs/docs/monitoring.md#why-these-model-thresholds).

Draws 20,000 windows of n labelled answers with replacement from the test
set's errors and range outcomes of lgbm-basic, for n = 200 (the fewest the
drift job reports an MdAPE for), 300 (the fewest for a coverage), 500 (a
replay window's ranges) and 1,000, and prints the quantiles of the windows'
MdAPE and coverage beside the binomial standard error at the nominal 0.9.

Run from the repository root after `dvc pull train features`:
    python reports/analysis/model_alert_thresholds.py
Output: reports/analysis/model_alert_thresholds_results.txt
"""

from pathlib import Path

import numpy as np
from scipy import stats

from recommenditos.data.build_features import FeatureSpace
from recommenditos.modeling.model import load_model
from recommenditos.pipeline import read_frame

VARIANT = "lgbm-basic"
NOMINAL = 0.9
DRAWS = 20_000
SEED = 230

model = load_model(Path("models") / VARIANT)
features = Path("data/processed/features") / model.feature_set
space = FeatureSpace.load(features, name=f"features-{model.feature_set}")
test = read_frame(features / "test.parquet", space.schema)
price = test["price"].to_numpy()
ape = np.abs(price - model.predict_eur(test).to_numpy()) / price
interval = model.predict_interval_eur(test)
inside = (interval["lower_eur"].to_numpy() <= price) & (price <= interval["upper_eur"].to_numpy())
test_mdape = float(np.median(ape))
print(f"{VARIANT} test set: {len(ape)} rows, MdAPE {test_mdape:.4f}, coverage {inside.mean():.4f}")

rng = np.random.default_rng(SEED)
z = stats.norm.ppf(0.995)
print(f"one-sided 0.005: z = {z:.3f}")
print("n     MdAPE q99.5  x test  q99.9  x test | coverage q0.5  binomial SE  nominal - z SE")
for n in (200, 300, 500, 1000):
    draws = rng.integers(0, len(ape), size=(DRAWS, n))
    mdape = np.median(ape[draws], axis=1)
    coverage = inside[draws].mean(axis=1)
    se = np.sqrt(NOMINAL * (1 - NOMINAL) / n)
    q995, q999 = np.quantile(mdape, [0.995, 0.999])
    print(
        f"{n:<5} {q995:.4f}       {q995 / test_mdape:.3f}   {q999:.4f} {q999 / test_mdape:.3f}"
        f"  | {np.quantile(coverage, 0.005):.4f}         {se:.4f}       {NOMINAL - z * se:.4f}"
    )
