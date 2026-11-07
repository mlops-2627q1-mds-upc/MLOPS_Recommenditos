"""Judge a window's error and coverage against its market's baseline? (#306)

ModelErrorHigh and ModelCoverageLow (#230) compare a window with the model's
overall test values: its MdAPE with the overall test MdAPE times
1 + C / sqrt(n), its coverage with the nominal 0.9 less Z binomial standard
errors and 0.5 / n (deploy/prometheus/rules/model-performance.yml). In the
FR-15 rehearsal (#231) both fired on an `ES` window at the model's own `ES`
error, because `ES` is above the average on the test set itself.

The options, each with the rule's form unchanged and only the baseline replaced:
    A  overall: the overall test MdAPE, and the nominal level for the coverage;
    B  market: the test MdAPE and coverage of the window's majority country;
    C  mix: the mean of the per-country test values, weighted by the countries
       of the window's labelled answers (MdAPE) and labelled ranges (coverage);
    D  mix with the baseline's own error: C's baseline, and a margin for the
       sampling error of both the window and the baseline, as a two-sample
       test: 1 + C sqrt(1 / n + 1 / m) and p - z sqrt(p (1 - p) (1 / n + 1 / m))
       - 0.5 / n, with 1 / m = sum of w_c^2 / m_c over the window's countries c,
       w_c its share and m_c the test rows of c's baseline (m_c = 0 terms for
       the nominal level, which is a promise, not an estimate).
A country with fewer than MIN_BASELINE_ROWS test rows (SC-04's minimum segment)
takes the overall values of A, and so does a country the test set lacks.

Method of model_alert_thresholds.py: windows of n labelled answers and n labelled
ranges drawn with replacement from the test set's errors and range outcomes of
lgbm-basic, for each market, the test mix (a control window), and a window
of 78 % `ES` (rehearsal window 11). Per window and option:
- the false-alarm rate, the share of windows that fire, against NFR-11's 0.005;
  "fixed" treats the test values as the truth, as the current thresholds do;
  "redrawn" also draws each baseline again from its own test rows, the
  expected rate over a baseline's own sampling error, which matters when a
  market has a few hundred test rows: one model keeps its estimate for every
  window, so a baseline that came out low makes all of its windows fire;
- the miss rate of a degraded window: every APE times 1.2 or 1.3, or the
  coverage 3 or 5 points below the group's own test coverage;
- the verdict of each option on the windows of the rehearsal
  (reports/monitoring/fr15-rehearsal.md), recomputed per request for the `ES`
  windows from reports/monitoring/fr15/paired-es-requests.csv.

Run from the repository root after `dvc pull train features`:
    python reports/analysis/market_baseline.py > reports/analysis/market_baseline_results.txt
Output: reports/analysis/market_baseline_results.txt
"""

from collections.abc import Callable, Iterator
import csv
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
from scipy import stats

from recommenditos.data.build_features import FeatureSpace
from recommenditos.modeling.model import load_model
from recommenditos.pipeline import read_frame

logger.remove()  # the model's and the reader's log lines would land in the results file

VARIANT = "lgbm-basic"
NOMINAL = 0.9
ALPHA = 0.005
DRAWS = 20_000
CHUNK = 500
SEED = 306
C = 3.52  # the MdAPE ceiling's constant, as in the rules file
Z = float(stats.norm.ppf(1 - ALPHA))
MIN_BASELINE_ROWS = 500  # params.yaml success_criteria.sc04_min_segment_rows
ANSWERS = (200, 500, 1000)  # labelled answers: the job's minimum, a half and a full window
RANGES = (300, 500, 1000)  # labelled ranges: the job's minimum, a replay window, all
REPLAY = (1000, 500)  # a replay window: 1,000 labelled answers, 500 of them ranges
ES_MIX = 0.78  # the `ES` share of rehearsal window 11
MDAPE_FACTORS = (1.2, 1.3)
COVERAGE_DROPS = (0.03, 0.05)
OPTIONS = ("A", "B", "C", "D")
#: The model before the promotion of #303 (afc3335e): its test MdAPE, and its
#: per-country test MdAPE in the segments.csv of ccb175e^, which has no `ES`
#: row, since all `ES` sellers were held out of its split. Its coverage per
#: country was never stored; the rehearsal's control parts take the promoted
#: model's (said in the output).
BEFORE_MDAPE = 0.06733805931326535
BEFORE_COUNTRY_MDAPE = {
    "DE": 0.06327326686869228,
    "IT": 0.0686580438177411,
    "NL": 0.0698088409354266,
    "BE": 0.06844616620761967,
    "AT": 0.07155128629892624,
    "FR": 0.08080920969180333,
    "LU": 0.054544597127415766,
}
BEFORE_COUNTRY_ROWS = {"DE": 8030, "IT": 4529, "NL": 3089, "BE": 1679, "AT": 1133, "FR": 1058, "LU": 147}
PAIRED = Path("reports/monitoring/fr15/paired-es-requests.csv")
SEGMENTS = Path("reports/metrics/segments.csv")


def ceiling(n: int) -> float:
    return 1 + C / np.sqrt(n)


def floor(p: np.ndarray | float, n: int) -> np.ndarray | float:
    return p - Z * np.sqrt(p * (1 - p) / n) - 0.5 / n


def two_sample_ceiling(n: int, inverse_m: np.ndarray | float) -> np.ndarray | float:
    return 1 + C * np.sqrt(1 / n + inverse_m)


def two_sample_floor(p: np.ndarray | float, n: int, inverse_m: np.ndarray | float) -> np.ndarray | float:
    return p - Z * np.sqrt(p * (1 - p) * (1 / n + inverse_m)) - 0.5 / n


# --------------------------------------------------------------------------
# The test set
# --------------------------------------------------------------------------

model = load_model(Path("models") / VARIANT)
features = Path("data/processed/features") / model.feature_set
space = FeatureSpace.load(features, name=f"features-{model.feature_set}")
test = read_frame(features / "test.parquet", space.schema)
price = test["price"].to_numpy()
ape = np.abs(price - model.predict_eur(test).to_numpy()) / price
interval = model.predict_interval_eur(test)
inside = (interval["lower_eur"].to_numpy() <= price) & (price <= interval["upper_eur"].to_numpy())
country = test["country_code"].astype(str).to_numpy()
markets = [str(m) for m in test["country_code"].astype(str).value_counts().index]
code = np.array([markets.index(c) for c in country])
K = len(markets)
overall_mdape, overall_coverage = float(np.median(ape)), float(inside.mean())
rows = np.bincount(code, minlength=K)
market_mdape = np.array([np.median(ape[code == k]) for k in range(K)])
market_coverage = np.array([inside[code == k].mean() for k in range(K)])
has_baseline = rows >= MIN_BASELINE_ROWS
#: The baseline each market gets under B and C: its own, or A's values.
base_mdape = np.where(has_baseline, market_mdape, overall_mdape)
base_coverage = np.where(has_baseline, market_coverage, NOMINAL)
#: 1 / m_c of D: a country without its own baseline takes the overall test rows
#: for the MdAPE, and the nominal level, without sampling error, for the coverage.
inverse_rows_mdape = np.where(has_baseline, 1 / np.maximum(rows, 1), 1 / len(ape))
inverse_rows_coverage = np.where(has_baseline, 1 / np.maximum(rows, 1), 0.0)
ES = markets.index("ES")

with SEGMENTS.open(newline="", encoding="utf-8") as table:
    stored = {
        r["level"]: float(r["mdape"])
        for r in csv.DictReader(table)
        if (r["variant"], r["segment"]) == (VARIANT, "country_code")
    }

print(f"{VARIANT} test set: {len(ape)} rows, MdAPE {overall_mdape:.4f}, coverage {overall_coverage:.4f}")
print(f"one-sided {ALPHA}: z = {Z:.3f}; ceiling 1 + {C} / sqrt(n); floor p - z sqrt(p (1 - p) / n) - 0.5 / n")
print(f"\nPer-country test values (a baseline needs {MIN_BASELINE_ROWS} test rows, else A's values)")
print("market  rows   MdAPE   x overall  segments.csv  coverage  baseline under B and C")
for k, m in enumerate(markets):
    print(
        f"{m:<7} {rows[k]:<6} {market_mdape[k]:.4f}  {market_mdape[k] / overall_mdape:.3f}      "
        f"{stored.get(m, float('nan')):.4f}        {market_coverage[k]:.4f}    "
        f"{'own' if has_baseline[k] else 'overall'}"
    )

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# Baselines drawn again from their own test rows, for "with baseline error"
# --------------------------------------------------------------------------


def bootstrap_medians(values: np.ndarray, draws: int) -> np.ndarray:
    out = np.empty(draws)
    for start in range(0, draws, CHUNK):
        size = min(CHUNK, draws - start)
        out[start : start + size] = np.median(
            values[rng.integers(0, len(values), size=(size, len(values)))], axis=1
        )
    return out


redrawn_overall = bootstrap_medians(ape, DRAWS)
redrawn_mdape = np.column_stack(
    [bootstrap_medians(ape[code == k], DRAWS) if has_baseline[k] else redrawn_overall for k in range(K)]
)
redrawn_coverage = np.column_stack(
    [
        rng.binomial(rows[k], market_coverage[k], DRAWS) / rows[k] if has_baseline[k] else np.full(DRAWS, NOMINAL)
        for k in range(K)
    ]
)
spread = np.std(redrawn_mdape, axis=0) / base_mdape
print("\nStandard error of each baseline MdAPE, relative (bootstrap of its own test rows):")
print("  " + ", ".join(f"{m} {s:.3f}" for m, s, b in zip(markets, spread, has_baseline, strict=True) if b))
print(f"  overall {np.std(redrawn_overall) / overall_mdape:.3f}; a window of 1,000 answers: see q99.5 - 1 over z")

# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Group:
    name: str
    #: Draws `size` windows of `n` rows: an array of test row indices.
    draw: Callable[[int, int], np.ndarray]


def pool(indices: np.ndarray) -> Callable[[int, int], np.ndarray]:
    return lambda size, n: indices[rng.integers(0, len(indices), size=(size, n))]


def mix(es_share: float) -> Callable[[int, int], np.ndarray]:
    es, rest = np.flatnonzero(code == ES), np.flatnonzero(code != ES)

    def draw(size: int, n: int) -> np.ndarray:
        k = round(es_share * n)
        return np.hstack([pool(es)(size, k), pool(rest)(size, n - k)])

    return draw


groups = [Group("control (test mix)", pool(np.arange(len(ape))))]
groups += [Group(m, pool(np.flatnonzero(code == k))) for k, m in enumerate(markets)]
groups += [Group(f"{ES_MIX:.0%} ES, rest test mix", mix(ES_MIX))]


def chunks(group: Group, n: int) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
    """Per chunk of windows: their slice of the draws, row indices and country counts."""
    for start in range(0, DRAWS, CHUNK):
        size = min(CHUNK, DRAWS - start)
        drawn = group.draw(size, n)
        counts = np.stack([(code[drawn] == k).sum(axis=1) for k in range(K)], axis=1)
        yield slice(start, start + size), drawn, counts


def mdape_fires(group: Group, n: int, factor: float = 1.0) -> dict[str, np.ndarray]:
    """Per option and draw: does ModelErrorHigh fire, with fixed and redrawn baselines?"""
    fired = {f"{o} {kind}": np.empty(DRAWS, bool) for o in OPTIONS for kind in ("fixed", "redrawn")}
    for part, drawn, counts in chunks(group, n):
        window = np.median(ape[drawn], axis=1) * factor
        majority = counts.argmax(axis=1)
        weights = counts / n
        baselines = {
            "A fixed": np.full(len(window), overall_mdape),
            "B fixed": base_mdape[majority],
            "C fixed": weights @ base_mdape,
            "A redrawn": redrawn_overall[part],
            "B redrawn": redrawn_mdape[part][np.arange(len(window)), majority],
            "C redrawn": (weights * redrawn_mdape[part]).sum(axis=1),
        }
        baselines["D fixed"], baselines["D redrawn"] = baselines["C fixed"], baselines["C redrawn"]
        wide = two_sample_ceiling(n, weights**2 @ inverse_rows_mdape)
        for key, baseline in baselines.items():
            fired[key][part] = window / baseline > (wide if key.startswith("D") else ceiling(n))
    return fired


def coverage_fires(group: Group, n: int, drop: float = 0.0) -> dict[str, np.ndarray]:
    """Per option and draw: does ModelCoverageLow fire? `drop` lowers the group's coverage."""
    fired = {f"{o} {kind}": np.empty(DRAWS, bool) for o in OPTIONS for kind in ("fixed", "redrawn")}
    for part, drawn, counts in chunks(group, n):
        held = inside[drawn]
        if drop:
            # Turn a share of the held ranges into misses: drop / p of them in expectation.
            p = held.mean()
            held = held & (rng.random(held.shape) >= drop / p)
        window = held.mean(axis=1)
        majority = counts.argmax(axis=1)
        weights = counts / n
        baselines = {
            "A fixed": np.full(len(window), NOMINAL),
            "B fixed": base_coverage[majority],
            "C fixed": weights @ base_coverage,
            "A redrawn": np.full(len(window), NOMINAL),
            "B redrawn": redrawn_coverage[part][np.arange(len(window)), majority],
            "C redrawn": (weights * redrawn_coverage[part]).sum(axis=1),
        }
        baselines["D fixed"], baselines["D redrawn"] = baselines["C fixed"], baselines["C redrawn"]
        inverse_m = weights**2 @ inverse_rows_coverage
        for key, baseline in baselines.items():
            low = two_sample_floor(baseline, n, inverse_m) if key.startswith("D") else floor(baseline, n)
            fired[key][part] = window < low
    return fired


def row(cells: dict[str, np.ndarray]) -> str:
    return "  ".join(f"{cells[f'{o} fixed'].mean():.4f}/{cells[f'{o} redrawn'].mean():.4f}" for o in OPTIONS)


HEADER = "A fixed/redrawn  B fixed/redrawn  C fixed/redrawn  D fixed/redrawn"
print(f"\nFalse-alarm rate per window, in-distribution windows; level {ALPHA}")
print("Each cell: fixed baseline / baseline redrawn from its test rows.")
print(f"\nModelErrorHigh, n labelled answers\ngroup                    n     {HEADER}")
for group in groups:
    for n in ANSWERS:
        print(f"{group.name:<24} {n:<5} {row(mdape_fires(group, n))}")
print(f"\nModelCoverageLow, n labelled ranges\ngroup                    n     {HEADER}")
for group in groups:
    for n in RANGES:
        print(f"{group.name:<24} {n:<5} {row(coverage_fires(group, n))}")

print(f"\nMiss rate of a degraded window (1 - power); a replay window has {REPLAY[0]} answers, {REPLAY[1]} ranges")
print("ModelErrorHigh: every APE of the window times the factor")
print(f"group                    n     factor  {HEADER}")
for group in groups:
    for n in (ANSWERS[0], REPLAY[0]):
        for factor in MDAPE_FACTORS:
            fired = mdape_fires(group, n, factor)
            print(f"{group.name:<24} {n:<5} {factor:<7} {row({k: ~v for k, v in fired.items()})}")
print("ModelCoverageLow: the window's coverage lowered by the drop, from the group's own test coverage")
print(f"group                    n     drop    {HEADER}")
for group in groups:
    for n in (RANGES[0], REPLAY[1]):
        for drop in COVERAGE_DROPS:
            fired = coverage_fires(group, n, drop)
            print(f"{group.name:<24} {n:<5} {drop:<7} {row({k: ~v for k, v in fired.items()})}")

# --------------------------------------------------------------------------
# The rehearsal's windows
# --------------------------------------------------------------------------

with PAIRED.open(newline="", encoding="utf-8") as table:
    paired = list(csv.DictReader(table))


def es_window(when: str, half: int) -> tuple[float, int, float, int]:
    """MdAPE, answers, coverage and ranges of the `ES` window `half` (0, 1) of a replay."""
    part = paired[half * 1000 : (half + 1) * 1000]
    errors = [float(r[f"ape_{when}"]) for r in part]
    ranged = [int(r[f"in_range_{when}"]) for r in part if r[f"in_range_{when}"] != ""]
    return float(np.median(errors)), len(errors), sum(ranged) / len(ranged), len(ranged)


# The control part of a window: the test mix without `ES` (the replay's control
# sample is drawn from the training data, of the same countries).
rest_share = np.bincount(code[code != ES], minlength=K) / (code != ES).sum()
before_mdape = np.array(
    [
        BEFORE_COUNTRY_MDAPE[m] if BEFORE_COUNTRY_ROWS.get(m, 0) >= MIN_BASELINE_ROWS else BEFORE_MDAPE
        for m in markets
    ]
)
# Before: no `ES` coverage baseline exists (A's nominal level); the other
# countries take the promoted model's coverage, which only the control part uses.
before_coverage = base_coverage.copy()
before_coverage[ES] = NOMINAL
before_rows = sum(BEFORE_COUNTRY_ROWS.values())
before_inverse_mdape = np.array(
    [
        1 / BEFORE_COUNTRY_ROWS[m] if BEFORE_COUNTRY_ROWS.get(m, 0) >= MIN_BASELINE_ROWS else 1 / before_rows
        for m in markets
    ]
)
before_inverse_coverage = inverse_rows_coverage.copy()
before_inverse_coverage[ES] = 0.0
print(
    "\nThe rehearsal's windows (fr15-rehearsal.md; 13, 14, 17, 18 recomputed per request from "
    "paired-es-requests.csv)"
)
print(
    "Before (11-14, afc3335e): no `ES` row in its test set, so `ES` takes the overall values; its "
    "per-country coverage was\nnot stored, so a control part takes the promoted model's (only windows "
    "11 and 12 have one, both far from any floor)."
)
rehearsal = [
    # window, ES share, MdAPE, answers, coverage, ranges, model
    (11, 0.78, 0.0776, 1000, 0.8798, 499, "before"),
    (12, 0.0, 0.0544, 1000, 0.9580, 500, "before"),
    (13, 1.0, *es_window("before", 0), "before"),
    (14, 1.0, *es_window("before", 1), "before"),
    (15, 0.04, 0.0576, 1000, 0.9680, 500, "after"),
    (16, 0.04, 0.0539, 1000, 0.9600, 500, "after"),
    (17, 1.0, *es_window("after", 0), "after"),
    (18, 1.0, *es_window("after", 1), "after"),
]
print("window ES    MdAPE   n     coverage n    option  MdAPE x base vs ceiling  coverage vs floor   fires")
for window, share, mdape, answers, coverage, ranges, when in rehearsal:
    mdapes = before_mdape if when == "before" else base_mdape
    coverages = before_coverage if when == "before" else base_coverage
    overall = BEFORE_MDAPE if when == "before" else overall_mdape
    weights = share * (np.arange(K) == ES) + (1 - share) * rest_share
    majority = int(weights.argmax())
    option_baselines = {
        "A": (overall, NOMINAL),
        "B": (mdapes[majority], coverages[majority]),
        "C": (float(weights @ mdapes), float(weights @ coverages)),
    }
    option_baselines["D"] = option_baselines["C"]
    inverse_mdape = before_inverse_mdape if when == "before" else inverse_rows_mdape
    inverse_coverage = before_inverse_coverage if when == "before" else inverse_rows_coverage
    for option, (b_mdape, b_coverage) in option_baselines.items():
        multiple = mdape / b_mdape
        if option == "D":
            high = float(two_sample_ceiling(answers, weights**2 @ inverse_mdape))
            low = float(two_sample_floor(b_coverage, ranges, weights**2 @ inverse_coverage))
        else:
            high, low = float(ceiling(answers)), float(floor(b_coverage, ranges))
        fires = [name for name, hit in (("error", multiple > high), ("coverage", coverage < low)) if hit]
        lead = f"{window:<6} {share:<5.0%} {mdape:.4f}  {answers:<5} {coverage:.4f}   {ranges:<4}" if option == "A" else " " * 41
        print(
            f"{lead} {option}       {multiple:.3f} vs {high:.3f}           "
            f"{coverage:.4f} vs {low:.4f}   {', '.join(fires) or 'quiet'}"
        )

# --------------------------------------------------------------------------
# D's constant, calibrated for the two-sample case (#306, EDN-103)
# --------------------------------------------------------------------------
# As model_alert_thresholds.py sets C: the smallest constant K whose ceiling
# 1 + K sqrt(1 / n + 1 / m) is at or above the 99.5th percentile of the
# window's multiple of its baseline at every count and in every group with a
# baseline of its own, here per draw as the 99.5th percentile of
# (multiple - 1) / sqrt(1 / n + 1 / m), once with the fixed baselines and once
# with the redrawn ones.

CALIBRATION_COUNTS = (200, 250, 300, 400, 500, 600, 700, 800, 1000)
calibrated = [g for g in groups if g.name != "LU"]


def standardised(group: Group, n: int, factor: float = 1.0) -> dict[str, np.ndarray]:
    """Per draw: (window / D's baseline - 1) / sqrt(1 / n + 1 / m), fixed and redrawn."""
    out = {kind: np.empty(DRAWS) for kind in ("fixed", "redrawn")}
    for part, drawn, counts in chunks(group, n):
        window = np.median(ape[drawn], axis=1) * factor
        weights = counts / n
        scale = np.sqrt(1 / n + weights**2 @ inverse_rows_mdape)
        out["fixed"][part] = (window / (weights @ base_mdape) - 1) / scale
        out["redrawn"][part] = (window / (weights * redrawn_mdape[part]).sum(axis=1) - 1) / scale
    return out


print("\nD's constant: the 99.5th percentile of (multiple - 1) / sqrt(1 / n + 1 / m), fixed / redrawn")
print(("group                    " + "  ".join(f"{n:<11}" for n in CALIBRATION_COUNTS)).rstrip())
needed: dict[str, list[float]] = {"fixed": [], "redrawn": []}
for group in calibrated:
    cells = []
    for n in CALIBRATION_COUNTS:
        z = standardised(group, n)
        q = {kind: float(np.quantile(values, 1 - ALPHA)) for kind, values in z.items()}
        for kind in needed:
            needed[kind].append(q[kind])
        cells.append(f"{q['fixed']:.2f}/{q['redrawn']:.2f}")
    print((f"{group.name:<24} " + "  ".join(f"{c:<11}" for c in cells)).rstrip())
constants = {kind: float(np.ceil(max(values) * 100) / 100) for kind, values in needed.items()}
print(f"smallest constant at or above every percentile: fixed {constants['fixed']:.2f}, redrawn {constants['redrawn']:.2f}")

print("\nD at each constant: false-alarm rate (fixed / redrawn) and miss rate of x1.2, x1.3 (fixed / redrawn)")
print("group                    n     K     false alarm      miss x1.2        miss x1.3")
for k_name, k in (("C", C), ("fixed", constants["fixed"]), ("redrawn", constants["redrawn"])):
    for group in calibrated:
        for n in (ANSWERS[0], REPLAY[0]):
            null, worse, worst = standardised(group, n), standardised(group, n, 1.2), standardised(group, n, 1.3)
            print(
                f"{group.name:<24} {n:<5} {k:<5.2f} "
                f"{(null['fixed'] > k).mean():.4f}/{(null['redrawn'] > k).mean():.4f}  "
                f"{(worse['fixed'] <= k).mean():.4f}/{(worse['redrawn'] <= k).mean():.4f}  "
                f"{(worst['fixed'] <= k).mean():.4f}/{(worst['redrawn'] <= k).mean():.4f}"
            )
