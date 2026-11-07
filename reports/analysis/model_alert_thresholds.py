"""The thresholds of the model alerts ModelErrorHigh and ModelCoverageLow (#230).

Question: how far does a window's MdAPE and range coverage stray from the test
values by chance alone, when its requests come from the test distribution?
The alerts fire beyond the 0.005 one-sided tail, NFR-11's false-alarm level
per window (docs/docs/monitoring.md#why-these-model-thresholds), at a ceiling
and a floor that depend on the window's labelled count n, recorded by
deploy/prometheus/rules/model-performance.yml:
    MdAPE ceiling  test MdAPE * (1 + C / sqrt(n)),   n = labelled answers
    coverage floor p - Z * sqrt(p (1 - p) / n) - 0.5 / n,   n = labelled ranges

MdAPE: draws 20,000 windows of n labelled answers with replacement from the
test set's errors of lgbm-basic, for n from 200 (the fewest the drift job
reports an MdAPE for) to 1,000 (a full window), and compares the 99.5th
percentile of their MdAPE with the ceiling; C is the smallest constant whose
ceiling is at or above that percentile at every n, so the level is at most
0.005 at every count.
Coverage: the exact one-sided binomial false-alarm rate at p = the nominal
0.9 for every n from 300 (the fewest for a coverage) to 1,000, with and
without the continuity correction 0.5 / n, and the 0.5th percentile of 20,000
windows drawn from the test set's range outcomes.
Then the family-wise rate of the three window alerts, and the rules applied to
the windows of reports/monitoring/laptop-es-replay-2026-11-04.md, which the model
before #303 answered, so their MdAPE is a multiple of that model's test MdAPE.

Since #306 (EDN-103) ModelErrorHigh judges a window against the test MdAPE of
its countries, above 1 + 3.55 sqrt(1 / n + 1 / m) times it, m the test rows of
that baseline; reports/analysis/market_baseline.py measures and calibrates that
rule. The MdAPE part here is the one-sample ceiling of the rule before, kept as
the record EDN-101 cites; for a window of the test mix m is the whole test set,
so the two ceilings differ by at most 0.004.

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
ALPHA = 0.005
DRAWS = 20_000
SEED = 230
C = 3.52  # the MdAPE ceiling's constant, as in the rules file
#: The test MdAPE of the lgbm-basic model that answered the replay of 2026-11-04
#: (afc3335e, before #303), the baseline its windows were compared with.
REPLAY_TEST_MDAPE = 0.0673
Z = float(stats.norm.ppf(1 - ALPHA))
MDAPE_COUNTS = (200, 250, 300, 400, 500, 600, 700, 800, 1000)
COVERAGE_COUNTS = (300, 400, 500, 600, 700, 800, 1000)


def ceiling(n: int) -> float:
    return 1 + C / np.sqrt(n)


def floor(n: int, correction: bool = True) -> float:
    return NOMINAL - Z * np.sqrt(NOMINAL * (1 - NOMINAL) / n) - (0.5 / n if correction else 0)


def exact_rate(n: int, correction: bool = True) -> float:
    """P(coverage < floor) for n ranges that each hold the price with p = NOMINAL."""
    below = int(np.ceil(n * floor(n, correction))) - 1  # the largest count under the floor
    return float(stats.binom.cdf(below, n, NOMINAL))


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
print(f"one-sided {ALPHA}: z = {Z:.3f}")
rng = np.random.default_rng(SEED)

print(f"\nMdAPE ceiling, multiples of the test MdAPE: 1 + {C} / sqrt(n), n = labelled answers")
print("n     bootstrap q99.5  ceiling  error   false-alarm rate at the ceiling")
needed, mdape_rates, multiples = [], [], {}
for n in MDAPE_COUNTS:
    multiple = np.median(ape[rng.integers(0, len(ape), size=(DRAWS, n))], axis=1) / test_mdape
    multiples[n] = multiple
    q = float(np.quantile(multiple, 1 - ALPHA))
    rate = float((multiple > ceiling(n)).mean())
    needed.append((q - 1) * np.sqrt(n))
    mdape_rates.append(rate)
    print(f"{n:<5} {q:.4f}           {ceiling(n):.4f}   {ceiling(n) - q:+.4f}  {rate:.5f}")
counts = np.array(MDAPE_COUNTS)
fitted = float(np.sum(np.array(needed) / counts) / np.sum(1 / counts))  # least squares of q - 1 on 1 / sqrt(n)
worst = max(float((multiples[n] > 1 + fitted / np.sqrt(n)).mean()) for n in MDAPE_COUNTS)
print(f"C each n needs: {', '.join(f'{c:.2f}' for c in needed)}; largest {max(needed):.3f}")
print(f"least-squares C {fitted:.3f}: its false-alarm rate reaches {worst:.5f}, so the rules take the largest")

print("\nCoverage floor: p - z sqrt(p (1 - p) / n) - 0.5 / n, n = labelled ranges, p = 0.9")
print("n     bootstrap q0.5  floor, no correction  rate   floor  exact rate")
for n in COVERAGE_COUNTS:
    q = float(np.quantile(inside[rng.integers(0, len(inside), size=(DRAWS, n))].mean(axis=1), ALPHA))
    print(
        f"{n:<5} {q:.4f}          {floor(n, False):.4f}                {exact_rate(n, False):.4f} "
        f"{floor(n):.4f} {exact_rate(n):.4f}"
    )
coverage_rates = [exact_rate(n) for n in range(300, 1001)]
uncorrected = [exact_rate(n, False) for n in range(300, 1001)]
print(
    f"exact rate for n = 300..1,000: {min(coverage_rates):.4f} to {max(coverage_rates):.4f}, "
    f"mean {np.mean(coverage_rates):.4f}; without the correction {min(uncorrected):.4f} to "
    f"{max(uncorrected):.4f}, mean {np.mean(uncorrected):.4f}"
)

family = ALPHA + max(mdape_rates) + max(coverage_rates)
print(
    f"\nFamily-wise, per window under the null: at most {ALPHA} (drift, Bonferroni) "
    f"+ {max(mdape_rates):.4f} (MdAPE) + {max(coverage_rates):.4f} (coverage) = {family:.4f}"
)

print("\nThe replay windows (laptop-es-replay-2026-11-04.md): value against its threshold")
for name, mdape, answers, coverage, ranges in (
    ("1 control", 0.0538, 1000, 0.968, 500),
    ("2 control", 0.0528, 1000, 0.954, 500),
    ("3 ES", 0.0941, 1000, 0.858, 500),
):
    multiple = mdape / REPLAY_TEST_MDAPE
    print(
        f"window {name:<9}  MdAPE {multiple:.2f} x vs {ceiling(answers):.4f} at {answers}: "
        f"{'fires' if multiple > ceiling(answers) else 'quiet'};  coverage {coverage:.3f} vs "
        f"{floor(ranges):.4f} at {ranges}: {'fires' if coverage < floor(ranges) else 'quiet'}"
    )
