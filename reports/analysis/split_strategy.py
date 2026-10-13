"""How comparable the four sets are, and how much a reported number depends on the split (EDN-32).

The evidence behind issue #89, which asks whether the seller-grouped hash split should stay as it
is. It measures; it does not choose. In particular **no seed is selected from anything below**: the
seeds 1 to 20 are fixed here before any of them was run, they measure how much a figure moves when
other sellers land in the sets, and the committed seed stays the snapshot date it always was.

Seven blocks, numbered as the results file has them:

1. The committed split, set by set: rows, sellers, the share of every supported make, the price
   quantiles and the share of cars over 20 years, the segment SC-04 is missed on.
2. Why Mercedes-Benz is 8.67 % of the calibration set against 14.95 % of train, dealer by dealer.
3. The same quantities over 1,000 seeds, assignment only, so the committed draw can be read against
   what the hash split produces in general.
4. The split noise of the reported metrics: the real `split` and `features` stages and the `train`
   stage's own fit of the candidate, once per seed, in a temporary directory with tracking off, and
   the test MdAPE, the over-20 segment and SC-04's verdict of each. Beside it, on the validation
   rows only, the grid point EDN-73 found with the lowest validation MdAPE against the committed
   one, to ask whether that 0.08 pp survives another split.
5. Adversarial validation: one classifier that tries to tell train rows from test rows.
6. The stratified-group alternative, assignment only: how much better the make balance gets, and
   how many sellers move when the `ES` listings join the pool, which is FR-15's retrain.
7. Whether SC-05's interval coverage can be checked per make today.

The split seed and the model seed are separate here on purpose: only the split seed varies, and
every fit uses the committed `seed` and hyperparameters, so the spread is the split's and not the
fit's. Nothing under `data/`, `models/`, `metrics.json` or `dvc.lock` is written.

Run from the repository root after `dvc pull`, with tracking off (about 15 minutes on a laptop):

    MLFLOW_TRACKING_URI= uv run python reports/analysis/split_strategy.py \
        > reports/analysis/split_strategy_results.txt
"""

import json
from pathlib import Path
import sys
import tempfile
import time

from lightgbm import LGBMClassifier
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import yaml

from recommenditos.config import (
    INTERIM_DATA_DIR,
    METRICS_FILE,
    PARAMS_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from recommenditos.data import build_features, split_data
from recommenditos.data.build_features import age_years, read_supported_makes
from recommenditos.data.split_data import SPLIT_NAMES, assign_split
from recommenditos.modeling import evaluate
from recommenditos.modeling.model import Model, RequestContract, fit_variant
from recommenditos.modeling.train import read_matrices
from recommenditos.pipeline import load_params, read_frame

VARIANT = "lgbm-basic"
#: Fixed before any of them was run. They measure noise; none of them is a candidate split.
FIT_SEEDS = tuple(range(1, 21))
ASSIGNMENT_SEEDS = range(1, 1001)
#: Three, because one assignment takes about 45 s and they hardly differ from each other.
STRATIFIED_SEEDS = (1, 2, 3)
#: EDN-73's grid point with the lowest validation MdAPE, 6.88 % against the committed 6.96 %.
ALTERNATIVE_NUM_LEAVES = 255
#: Ten folds give the configured 60/10/10/20 as six, one, one and two of them.
FOLD_SETS = ("train",) * 6 + ("validation", "calibration") + ("test",) * 2
OVER_20_YEARS = 20
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

# The stages log through loguru to stdout; only their warnings are kept, on stderr.
logger.remove()
logger.add(sys.stderr, level="WARNING")

started_at = time.perf_counter()
params = load_params(PARAMS_FILE)
committed_seed = params["seed"]
ratios = params["split"]["ratios"]
group_key = params["split"]["group_key"]
settings = params["train"]["variants"][VARIANT]
feature_set = settings["feature_set"]
criteria = params["evaluate"]["success_criteria"]
supported = read_supported_makes(PROCESSED_DATA_DIR)

interim = pd.read_parquet(INTERIM_DATA_DIR / "listings.parquet")
is_holdout = interim["country_code"] == params["split"]["holdout_country"]
pool = interim[~is_holdout].reset_index(drop=True)
holdout = interim[is_holdout].reset_index(drop=True)
pool_make = pool["make"].where(pool["make"].isin(supported), "(unsupported)")
pool_over_20 = (age_years(pool["registration_date"], params["reference_date"]) >= OVER_20_YEARS).to_numpy()
pool_price = pool["price"].to_numpy()
sellers, seller_of_row = np.unique(pool[group_key].to_numpy(), return_inverse=True)
seller_keys = pd.Series(sellers)
makes = [*supported, "(unsupported)"]
make_of_row = pd.Categorical(pool_make, categories=makes).codes
pool_make_share = np.bincount(make_of_row, minlength=len(makes)) / len(pool)


def title(text: str) -> None:
    print(f"\n{'=' * 100}\n{text}\n{'=' * 100}")


def set_of_row(seed: int) -> np.ndarray:
    """The set index of every pool row under the stage's own hash assignment."""
    assigned = assign_split(seller_keys, ratios, seed)
    return pd.Categorical(assigned, categories=SPLIT_NAMES).codes[seller_of_row]


def describe(codes: np.ndarray) -> dict[str, np.ndarray]:
    """Per set: the make shares, the median price and the over-20 share, from set codes."""
    rows = np.bincount(codes, minlength=len(SPLIT_NAMES))
    by_make = np.bincount(codes * len(makes) + make_of_row, minlength=len(SPLIT_NAMES) * len(makes))
    by_make = by_make.reshape(len(SPLIT_NAMES), len(makes))
    return {
        "rows": rows,
        "make_share": by_make / rows[:, None],
        "median_price": np.array([np.median(pool_price[codes == index]) for index in range(len(SPLIT_NAMES))]),
        "over_20": np.bincount(codes, weights=pool_over_20, minlength=len(SPLIT_NAMES)) / rows,
    }


def worst_make_gap(make_share: np.ndarray) -> float:
    """The largest distance of any make's share in any set from its share of the pool."""
    return float(np.abs(make_share - pool_make_share).max())


print("# reports/analysis/split_strategy.py on the real snapshot")
print(f"pool {len(pool):,} rows in {len(sellers):,} seller groups after the "
      f"{params['split']['holdout_country']} holdout of {len(holdout):,}; committed seed {committed_seed}; "
      f"ratios {ratios}")
print("No seed is chosen from this file. The seeds below measure the noise of the split; "
      "the committed seed stays as it is.")

# ----------------------------------------------------------------------------------------------
title("1. The committed split, set by set (data/processed, every row the split wrote)")
frames = {name: pd.read_parquet(PROCESSED_DATA_DIR / f"{name}.parquet") for name in SPLIT_NAMES}
committed_codes = set_of_row(committed_seed)
committed = describe(committed_codes)
assert [len(frames[name]) for name in SPLIT_NAMES] == committed["rows"].tolist(), "not the committed split"

header = f"{'':24s}" + "".join(f"{name:>13s}" for name in SPLIT_NAMES) + f"{'pool':>13s}"
print(header)
print(f"{'rows':24s}" + "".join(f"{len(frames[name]):>13,}" for name in SPLIT_NAMES) + f"{len(pool):>13,}")
print(f"{'share of the pool':24s}" + "".join(f"{len(frames[name]) / len(pool):>13.2%}" for name in SPLIT_NAMES))
print(f"{'seller groups':24s}" + "".join(f"{frames[name][group_key].nunique():>13,}" for name in SPLIT_NAMES)
      + f"{len(sellers):>13,}")
print(f"{'rows of served makes':24s}"
      + "".join(f"{int(frames[name]['make'].isin(supported).sum()):>13,}" for name in SPLIT_NAMES)
      + f"{int(pool['make'].isin(supported).sum()):>13,}")
print("\nshare of the set's rows, per make")
for index, make in enumerate(makes):
    print(f"{make:24s}" + "".join(f"{committed['make_share'][row, index]:>13.2%}" for row in range(4))
          + f"{pool_make_share[index]:>13.2%}")
print(f"{'largest gap to the pool':24s}"
      + "".join(f"{np.abs(committed['make_share'][row] - pool_make_share).max():>12.2%} " for row in range(4)).rstrip())
print("\nprice in EUR")
for quantile in QUANTILES:
    print(f"{f'quantile {quantile:.2f}':24s}"
          + "".join(f"{frames[name]['price'].quantile(quantile):>13,.0f}" for name in SPLIT_NAMES)
          + f"{pool['price'].quantile(quantile):>13,.0f}")
print(f"\n{'over 20 years, rows':24s}"
      + "".join(f"{int(pool_over_20[committed_codes == row].sum()):>13,}" for row in range(4))
      + f"{int(pool_over_20.sum()):>13,}")
print(f"{'over 20 years, share':24s}" + "".join(f"{committed['over_20'][row]:>13.2%}" for row in range(4))
      + f"{pool_over_20.mean():>13.2%}")
test_matrix = pd.read_parquet(PROCESSED_DATA_DIR / "features" / feature_set / "test.parquet")
print(f"over 20 years among the {len(test_matrix):,} served test rows `evaluate` reads: "
      f"{int((test_matrix['age_years'] >= OVER_20_YEARS).sum())} (SC-04's segment)")

# ----------------------------------------------------------------------------------------------
title("2. Why Mercedes-Benz is under-represented in calibration: dealer by dealer")
MAKE = "Mercedes-Benz"
is_make = (pool_make == MAKE).to_numpy()
per_seller = pd.DataFrame({
    "rows": np.bincount(seller_of_row, minlength=len(sellers)),
    "make_rows": np.bincount(seller_of_row, weights=is_make, minlength=len(sellers)).astype(int),
    "set": assign_split(seller_keys, ratios, committed_seed).to_numpy(),
})
largest = per_seller["rows"].idxmax()
largest_row = int(np.flatnonzero(seller_of_row == largest)[0])
largest_mix = pool_make[seller_of_row == largest].value_counts().head(3)
print(f"the largest dealer: {per_seller.loc[largest, 'rows']:,} rows "
      f"({per_seller.loc[largest, 'rows'] / len(pool):.2%} of the pool), assigned to "
      f"{per_seller.loc[largest, 'set']}, {int(per_seller.loc[largest, 'make_rows'])} of them {MAKE}; "
      f"its makes: {', '.join(f'{name} {count:,}' for name, count in largest_mix.items())}")
without_largest = seller_of_row != largest
print(f"\n{MAKE} share" + " " * 9 + "".join(f"{name:>13s}" for name in SPLIT_NAMES))
print(f"{'as committed':28s}" + "".join(
    f"{is_make[committed_codes == row].mean():>13.2%}" for row in range(4)))
print(f"{'without the largest dealer':28s}" + "".join(
    f"{is_make[(committed_codes == row) & without_largest].mean():>13.2%}" for row in range(4)))

rest = is_make[without_largest]
print(f"{MAKE} share of the pool: {is_make.mean():.2%} with that dealer, {rest.mean():.2%} without it; the dealer "
      f"holds {per_seller.loc[largest, 'make_rows'] / is_make.sum():.1%} of all {MAKE} rows")

TOP = 20
top = per_seller.sort_values("make_rows", ascending=False).head(TOP)
in_top = np.isin(seller_of_row, top.index.to_numpy())
print(f"\nthe {TOP} dealers with the most {MAKE} listings hold {int(top['make_rows'].sum()):,} of the "
      f"{int(is_make.sum()):,} {MAKE} rows ({top['make_rows'].sum() / is_make.sum():.1%}); where they landed:")
print("  rank  rows  of them Mercedes-Benz  set")
for rank, (_, dealer) in enumerate(top.iterrows(), start=1):
    print(f"  {rank:>4}  {dealer['rows']:>4,}  {dealer['make_rows']:>21,}  {dealer['set']}")
landed = top.groupby("set")["make_rows"].agg(["count", "sum"]).reindex(SPLIT_NAMES).fillna(0).astype(int)
print(f"\n{'':36s}" + "".join(f"{name:>13s}" for name in SPLIT_NAMES))
print(f"{f'top-{TOP} dealers in the set':36s}" + "".join(f"{landed.loc[name, 'count']:>13,}" for name in SPLIT_NAMES))
print(f"{f'their {MAKE} rows':36s}" + "".join(f"{landed.loc[name, 'sum']:>13,}" for name in SPLIT_NAMES))
print(f"{'expected at the set ratio':36s}"
      + "".join(f"{top['make_rows'].sum() * ratios[name]:>13,.0f}" for name in SPLIT_NAMES))
print(f"{f'{MAKE} rows of all other dealers':36s}" + "".join(
    f"{int(is_make[(committed_codes == row) & ~in_top].sum()):>13,}" for row in range(4)))
print(f"{f'share without the top-{TOP} dealers':36s}" + "".join(
    f"{is_make[(committed_codes == row) & ~in_top].mean():>13.2%}" for row in range(4)))

# ----------------------------------------------------------------------------------------------
title(f"3. The hash split over {len(ASSIGNMENT_SEEDS):,} seeds, assignment only (no fit)")
draws, largest_in = [], []
for seed in ASSIGNMENT_SEEDS:
    codes = set_of_row(seed)
    draws.append(describe(codes))
    largest_in.append(SPLIT_NAMES[codes[largest_row]])
largest_in = np.array(largest_in)
gaps = np.array([worst_make_gap(draw["make_share"]) for draw in draws])
quantities = {
    f"{MAKE} share": (lambda d: d["make_share"][:, makes.index(MAKE)], "{:.2%}"),
    "BMW share": (lambda d: d["make_share"][:, makes.index("BMW")], "{:.2%}"),
    "median price EUR": (lambda d: d["median_price"], "{:,.0f}"),
    "over 20 years share": (lambda d: d["over_20"], "{:.2%}"),
}
print(f"{'':22s}{'set':13s}{'committed':>11s}{'mean':>10s}{'sd':>9s}{'2.5 %':>10s}{'97.5 %':>10s}"
      f"{'min':>10s}{'max':>10s}  committed is at percentile")
for label, (pick, fmt) in quantities.items():
    values = np.array([pick(draw) for draw in draws])
    for row, name in enumerate(SPLIT_NAMES):
        column = values[:, row]
        low, high = np.percentile(column, [2.5, 97.5])
        own = pick(committed)[row]
        cells = [fmt.format(value) for value in (own, column.mean(), column.std(ddof=1), low, high,
                                                 column.min(), column.max())]
        print(f"{label:22s}{name:13s}{cells[0]:>11s}{cells[1]:>10s}{cells[2]:>9s}{cells[3]:>10s}"
              f"{cells[4]:>10s}{cells[5]:>10s}{cells[6]:>10s}  {np.mean(column < own):.1%}")
own_gap = worst_make_gap(committed["make_share"])
print(f"\nlargest gap of any make's share in any set to its pool share: committed {own_gap:.2%}; over "
      f"the seeds mean {gaps.mean():.2%}, median {np.median(gaps):.2%}, 2.5 % to 97.5 % "
      f"{np.percentile(gaps, 2.5):.2%} to {np.percentile(gaps, 97.5):.2%}, max {gaps.max():.2%}; "
      f"{np.mean(gaps < own_gap):.1%} of the seeds are better balanced than the committed one")
print("the same gap by where the largest dealer landed:")
for name in SPLIT_NAMES:
    there = largest_in == name
    share = np.array([draw["make_share"][SPLIT_NAMES.index(name), makes.index(MAKE)] for draw in draws])[there]
    print(f"  in {name:12s} {there.mean():>6.1%} of the seeds, gap mean {gaps[there].mean():.2%}, "
          f"{MAKE} share of that set mean {share.mean():.2%}")

# ----------------------------------------------------------------------------------------------
title(f"4. Split noise of the reported metrics: {VARIANT} refitted on {len(FIT_SEEDS)} other splits")
segments = evaluate.criterion_segments(params)
alternative = {**settings, "params": {**settings["params"], "num_leaves": ALTERNATIVE_NUM_LEAVES}}
print(f"committed hyperparameters {settings['params']}, model seed {committed_seed} for every fit")
print(f"alternative, validation only: num_leaves {ALTERNATIVE_NUM_LEAVES}, everything else as committed")


def measure(seed: int) -> dict:
    """Split with `seed`, build the features and fit, all through the stages' own code."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        split_params = root / "params.yaml"
        split_params.write_text(yaml.safe_dump({**params, "seed": seed}), encoding="utf-8")
        tick = time.perf_counter()
        split_data.main(INTERIM_DATA_DIR / "listings.parquet", root / "processed", split_params)
        build_features.main(feature_set, root / "processed", root / "processed" / "features", PARAMS_FILE)
        # With the request contract, as the `train` stage reads them, so the candidate is fitted
        # with the blanked copies of its training rows it has had since EDN-82.
        data = read_matrices(root / "processed" / "features", feature_set, read_supported_makes(root / "processed"),
                             request=RequestContract.from_params(params))
        prepared = time.perf_counter() - tick
        tick = time.perf_counter()
        model = fit_variant(VARIANT, settings, data, seed=committed_seed, num_threads=params["train"]["num_threads"])
        fitted = time.perf_counter() - tick
        test = read_frame(root / "processed" / "features" / feature_set / "test.parquet", data.space.schema)
        other = fit_variant(VARIANT, alternative, data, seed=committed_seed,
                            num_threads=params["train"]["num_threads"])
    predicted = model.predict_eur(test)
    rows = evaluate.segment_rows(
        {name: evaluate.segment_level_column(test, name, params) for name in segments},
        test["price"], predicted, variant=VARIANT, min_rows=criteria["sc04_min_segment_rows"],
    )
    old = next(row for row in rows if row["segment"] == "age_bucket" and row["level"] == f"over {OVER_20_YEARS}")
    validation = data.validation
    return {
        "seed": seed,
        "largest_in": SPLIT_NAMES[set_of_row(seed)[largest_row]],
        "prepare_s": prepared,
        "fit_s": fitted,
        "n_test": len(test),
        **evaluate.point_metrics(test["price"], predicted),
        "over_20_n": old["n"],
        "over_20_mdape": old["mdape"],
        "sc04": evaluate.evaluate_sc04(rows, criteria, segments=segments),
        "trees": model.metadata["training"]["best_iteration"],
        "val_mdape": evaluate.point_metrics(validation["price"], model.predict_eur(validation))["mdape"],
        "val_l1": model.metadata["training"]["validation_l1_log_price"],
        "alt_val_mdape": evaluate.point_metrics(validation["price"], other.predict_eur(validation))["mdape"],
        "alt_val_l1": other.metadata["training"]["validation_l1_log_price"],
    }


def line(result: dict) -> str:
    worst = result["sc04"]["sc04_worst_segment"]
    return (f"{result['seed']:>9}  {result['n_test']:>6,}  {result['mdape']:>9.4%}  {result['within_20pct']:>9.2%}  "
            f"{result['over_20_n']:>6,}  {result['over_20_mdape']:>9.2%}  "
            f"{'pass' if result['sc04']['sc04_passed'] else 'miss':>5s}  "
            f"{worst['segment']}={worst['level']} {worst['mdape']:.2%} (n {worst['n']:,})  "
            f"{result['trees']:>6,}  {result['val_mdape']:>8.4%}  {result['alt_val_mdape'] - result['val_mdape']:>+8.3%}  "
            f"{result['alt_val_l1'] - result['val_l1']:>+9.5f}  {result['prepare_s']:>4.0f}  {result['fit_s']:>4.0f}  "
            f"{result['largest_in']}")


columns = ("     seed  test n  test MdAPE  within 20  o20 n  o20 MdAPE  SC-04  worst SC-04 level"
           "                        trees  val MdAPE  alt-com   alt-com L1  prep s  fit s  largest dealer in")
print(f"\n{columns}")
own = measure(committed_seed)
print(line(own) + "   <- the committed seed, as a check of the method")
recorded = json.loads(METRICS_FILE.read_text(encoding="utf-8"))["variants"][VARIANT]
print(f"metrics.json records MdAPE {recorded['mdape']:.4%} and within 20 % {recorded['within_20pct']:.2%} for "
      f"{VARIANT}; this script reproduces them to {abs(own['mdape'] - recorded['mdape']):.1e} and "
      f"{abs(own['within_20pct'] - recorded['within_20pct']):.1e}")
results = []
for seed in FIT_SEEDS:
    try:
        results.append(measure(seed))
    except ValueError as refused:
        print(f"{seed:>9}  refused by the stage: {str(refused)[:160]}")
        continue
    print(line(results[-1]))

print(f"\nover the {len(results)} other seeds{'':20s}{'committed':>10s}{'mean':>9s}{'sd':>8s}{'min':>9s}{'max':>9s}")
for label, key in (("test MdAPE", "mdape"), ("within 20 % (SC-02)", "within_20pct"),
                   ("over-20 MdAPE (SC-04)", "over_20_mdape"), ("validation MdAPE", "val_mdape")):
    values = np.array([result[key] for result in results])
    print(f"{label:39s}{own[key]:>10.3%}{values.mean():>9.3%}{values.std(ddof=1):>8.3%}"
          f"{values.min():>9.3%}{values.max():>9.3%}")
old_rows = np.array([result["over_20_n"] for result in results])
print(f"{'over-20 test rows':39s}{own['over_20_n']:>10,}{old_rows.mean():>9,.0f}{old_rows.std(ddof=1):>8,.0f}"
      f"{old_rows.min():>9,}{old_rows.max():>9,}")
passed = [result["seed"] for result in results if result["sc04"]["sc04_passed"]]
elsewhere = [result["seed"] for result in results
             if result["sc04"]["sc04_worst_segment"]["level"] != f"over {OVER_20_YEARS}"]
below = [result["seed"] for result in results if result["over_20_mdape"] <= criteria["sc04_segment_mdape_max"]]
print(f"SC-04 (bound {criteria['sc04_segment_mdape_max']:.0%}) passes on {len(passed)} of {len(results)} seeds"
      f"{f' ({passed})' if passed else ''}; the over-20 segment is at or under the bound on {len(below)}"
      f"{f' ({below})' if below else ''}; the worst level is another one on {len(elsewhere)}"
      f"{f' ({elsewhere})' if elsewhere else ''}")
for name in SPLIT_NAMES:
    there = [result for result in results if result["largest_in"] == name]
    if there:
        print(f"  largest dealer in {name:12s} {len(there):>2} seeds: test MdAPE mean "
              f"{np.mean([r['mdape'] for r in there]):.3%}, over-20 MdAPE mean "
              f"{np.mean([r['over_20_mdape'] for r in there]):.2%}")
print(f"SC-01 (MdAPE at most {criteria['sc01_mdape_max']:.0%}) and SC-02 (at least "
      f"{criteria['sc02_within_20pct_min']:.0%} within 20 %) hold on "
      f"{sum(r['mdape'] <= criteria['sc01_mdape_max'] and r['within_20pct'] >= criteria['sc02_within_20pct_min'] for r in results)}"
      f" of {len(results)} seeds")

validation_sd = np.std([result["val_mdape"] for result in results], ddof=1)
difference = np.array([result["alt_val_mdape"] - result["val_mdape"] for result in results])
difference_l1 = np.array([result["alt_val_l1"] - result["val_l1"] for result in results])
print("\nthe tuning grid of EDN-73 against this noise")
print(f"  the grid's validation MdAPE spans 6.88 % to 7.04 %, 0.16 pp; the validation MdAPE of the committed point "
      f"has a standard deviation of {validation_sd * 100:.2f} pp over the splits and a range of "
      f"{(max(r['val_mdape'] for r in results) - min(r['val_mdape'] for r in results)) * 100:.2f} pp")
print(f"  num_leaves {ALTERNATIVE_NUM_LEAVES} minus committed, same split, validation MdAPE: committed split "
      f"{(own['alt_val_mdape'] - own['val_mdape']) * 100:+.3f} pp; over the seeds mean {difference.mean() * 100:+.3f} pp, "
      f"sd {difference.std(ddof=1) * 100:.3f} pp, lower on {int((difference < 0).sum())} of {len(difference)}")
print(f"  the same in validation L1 (log price), which the tuning protocol decides by: committed split "
      f"{own['alt_val_l1'] - own['val_l1']:+.5f}; mean {difference_l1.mean():+.5f}, sd {difference_l1.std(ddof=1):.5f}, "
      f"lower on {int((difference_l1 < 0).sum())} of {len(difference_l1)}")

# ----------------------------------------------------------------------------------------------
title("5. Adversarial validation: can a classifier tell train rows from test rows?")
features_dir = PROCESSED_DATA_DIR / "features" / feature_set
matrices, groups = [], []
for name in ("train", "test"):
    matrix = pd.read_parquet(features_dir / f"{name}.parquet")
    served = frames[name][frames[name]["make"].isin(supported)].reset_index(drop=True)
    assert len(served) == len(matrix) and np.allclose(served["price"], matrix["price"]), name
    matrices.append(matrix.drop(columns="log_price").assign(is_test=int(name == "test")))
    groups.append(served[group_key])
both = pd.concat(matrices, ignore_index=True)
both_groups = pd.concat(groups, ignore_index=True).to_numpy()
rng = np.random.default_rng(committed_seed)
unique, inverse = np.unique(both_groups, return_inverse=True)
held_out = {
    "whole sellers held out": (rng.random(len(unique)) < 0.3)[inverse],
    "rows held out at random": rng.random(len(both)) < 0.3,
}
print(f"{len(both):,} rows of the `{feature_set}` matrices, {both['is_test'].mean():.1%} of them test; the "
      f"model's features and the price; one LightGBM classifier (200 trees, 31 leaves) per row below, "
      f"scored on the 30 % it did not see")
for label, mask in held_out.items():
    classifier = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=committed_seed,
                                n_jobs=params["train"]["num_threads"], deterministic=True, verbose=-1)
    classifier.fit(both.loc[~mask].drop(columns="is_test"), both.loc[~mask, "is_test"])
    score = classifier.predict_proba(both.loc[mask].drop(columns="is_test"))[:, 1]
    print(f"  {label:26s} AUC {roc_auc_score(both.loc[mask, 'is_test'], score):.3f} on {int(mask.sum()):,} rows")

# ----------------------------------------------------------------------------------------------
title("6. The stratified-group alternative (StratifiedGroupKFold on make), assignment only")
strata = make_of_row


def stratified(seed: int, labels: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """The set index of every row: ten stratified group folds, mapped six, one, one and two."""
    folds = StratifiedGroupKFold(n_splits=len(FOLD_SETS), shuffle=True, random_state=seed)
    codes = np.empty(len(labels), dtype=int)
    for fold, (_, rows) in enumerate(folds.split(labels, labels, group_ids)):
        codes[rows] = SPLIT_NAMES.index(FOLD_SETS[fold])
    return codes


tick = time.perf_counter()
stratified_draws = {seed: stratified(seed, strata, seller_of_row) for seed in STRATIFIED_SEEDS}
for codes in stratified_draws.values():
    assert (pd.Series(codes).groupby(seller_of_row).nunique() == 1).all(), "a seller is in two sets"
print(f"{len(STRATIFIED_SEEDS)} seeds, {time.perf_counter() - tick:.0f} s; no seller is in two sets under any of them")
described = [describe(codes) for codes in stratified_draws.values()]
print("the largest dealer lands in: "
      + ", ".join(f"{SPLIT_NAMES[codes[largest_row]]} (seed {seed})" for seed, codes in stratified_draws.items()))
stratified_gaps = np.array([worst_make_gap(draw["make_share"]) for draw in described])
print(f"\n{'':34s}{'hash, committed':>17s}{f'hash, {len(ASSIGNMENT_SEEDS):,} seeds':>19s}"
      f"{f'stratified, {len(STRATIFIED_SEEDS)} seeds':>22s}")
print(f"{'largest make-share gap to pool':34s}{own_gap:>17.2%}{f'{gaps.mean():.2%} mean':>19s}"
      f"{f'{stratified_gaps.mean():.2%} mean, {stratified_gaps.max():.2%} max':>22s}")
for label, pick in ((f"{MAKE} share, calibration", lambda d: d["make_share"][2, makes.index(MAKE)]),
                    (f"{MAKE} share, sd over sets", lambda d: d["make_share"][:, makes.index(MAKE)].std(ddof=1)),
                    ("median price, max-min over sets", lambda d: np.ptp(d["median_price"])),
                    ("over-20 share, max-min over sets", lambda d: np.ptp(d["over_20"])),
                    ("row share, worst gap to ratio",
                     lambda d: np.abs(d["rows"] / d["rows"].sum() - [ratios[n] for n in SPLIT_NAMES]).max())):
    fmt = "{:,.0f}" if "price" in label else "{:.2%}"
    print(f"{label:34s}{fmt.format(pick(committed)):>17s}"
          f"{fmt.format(np.mean([pick(draw) for draw in draws])) + ' mean':>19s}"
          f"{fmt.format(np.mean([pick(draw) for draw in described])) + ' mean':>22s}")

print(f"\nFR-15's retrain adds the {len(holdout):,} `{params['split']['holdout_country']}` rows to the pool. "
      "Sellers already in the pool that change set when it does:")
grown_groups = np.concatenate([pool[group_key].to_numpy(), holdout[group_key].to_numpy()])
grown_sellers, grown_seller_of_row = np.unique(grown_groups, return_inverse=True)
grown_make = pd.concat([pool["make"], holdout["make"]], ignore_index=True)
grown_strata = pd.Categorical(grown_make.where(grown_make.isin(supported), "(unsupported)"), categories=makes).codes
before_hash = assign_split(pd.Series(pool[group_key].to_numpy()), ratios, committed_seed).to_numpy()
after_hash = assign_split(pd.Series(grown_groups), ratios, committed_seed).to_numpy()[: len(pool)]
print(f"  hash split: {np.mean(before_hash != after_hash):.2%} of the pool's rows "
      f"({len(np.unique(grown_sellers)) - len(sellers):,} sellers are new)")
moved_rows, moved_sellers = [], []
for seed in STRATIFIED_SEEDS:
    after = stratified(seed, grown_strata, grown_seller_of_row)[: len(pool)]
    changed = after != stratified_draws[seed]
    moved_rows.append(changed.mean())
    moved_sellers.append(len(np.unique(seller_of_row[changed])) / len(sellers))
print(f"  stratified-group split, same seed before and after: {np.mean(moved_rows):.1%} of the pool's rows on "
      f"average ({min(moved_rows):.1%} to {max(moved_rows):.1%}), {np.mean(moved_sellers):.1%} of its sellers")
other_seed = stratified(STRATIFIED_SEEDS[0] + 1000, strata, seller_of_row) != stratified_draws[STRATIFIED_SEEDS[0]]
print(f"  for scale, a different seed on the same pool moves {other_seed.mean():.1%} of the rows, and four sets "
      f"at 60/10/10/20 drawn independently twice disagree on "
      f"{1 - sum(ratio ** 2 for ratio in ratios.values()):.1%}")

# ----------------------------------------------------------------------------------------------
title("7. Can the UC2 interval coverage be checked per make on the test set today?")
print(f"Model has `{evaluate.INTERVAL_METHOD}`: {hasattr(Model, evaluate.INTERVAL_METHOD)}")
summary = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
print(f"metrics.json: criteria_not_measured = {summary['criteria_not_measured']!r}, "
      f"deployable_variant = {summary['deployable_variant']!r}")
for name in params["train"]["variants"]:
    record = json.loads((REPORTS_DIR / "metrics" / f"{name}.json").read_text(encoding="utf-8"))
    print(f"  {name:14s} sc05_status {record['sc05_status']!r}, sc05_coverage {record['sc05_coverage']!r}")
print("No interval exists, so no coverage can be computed, per make or pooled. The calibration set is written "
      "by `split` and read by no model yet.")

print(f"\ntotal run time {(time.perf_counter() - started_at) / 60:.1f} min")
