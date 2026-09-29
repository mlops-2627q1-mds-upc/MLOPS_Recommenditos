"""NFR-11's control clause against the significance level it is derived from.

The drift job flags a window when any feature's p-value falls below a Bonferroni threshold of
`P_VAL / n_features`. With `P_VAL = 0.05` the family-wise false-alarm rate is 5 % *by
construction*, which is exactly the "at most 1 of 20 control windows" NFR-11 promises. The
requirement therefore asks the test to perform at its own theoretical bound with no margin, so any
estimation error or mild non-exchangeability in the control sample breaks it. Two runs on slightly
different scopes measured 0.7 and 2.3 flagged windows of 20, which is that instability showing.

This run separates the two things that were confounded:

1. **Stability.** Three independent i.i.d. splits, so the false-alarm rate is not read off one
   lucky draw.
2. **The significance level.** p-values do not depend on the threshold, so each trial's p-values
   are computed once and evaluated at several values of `P_VAL`. A lower level buys margin against
   the promise; the question is whether the `ES` replay still gets flagged.

    PYTHONPATH=reports/analysis uv run python reports/analysis/nfr11_alpha_sweep.py <cars.csv>
"""

import json
import sys

import numpy as np

from nfr11_check import FEATURES, SEED, drift, load

REFERENCE_SIZE = 10_000
WINDOWS = [100, 1000]
N_TRIALS = 100
N_SPLITS = 3
CONTROL_FRACTION = 0.2
ALPHAS = [0.05, 0.01, 0.005, 0.001]
# `model` is excluded for the reason measured in EDN-26: 349 levels against these window sizes
# leaves well under one expected observation per chi-square cell.
DRIFT_FEATURES = [f for f in FEATURES if f != "model"]


def collect(ref_df, pool, window, rng, n_trials=N_TRIALS):
    """Return one p-value vector per trial, so any threshold can be evaluated afterwards."""
    out = []
    for _ in range(n_trials):
        win = pool.iloc[rng.choice(len(pool), size=window, replace=False)]
        ps = drift(ref_df, win, DRIFT_FEATURES)
        out.append({f: p for f, (p, _) in ps.items()})
    return out


def rates(trials, alpha):
    thr = alpha / len(DRIFT_FEATURES)
    flagged, counts = [], []
    for ps in trials:
        drifted = [f for f, p in ps.items() if p < thr]
        flagged.append(len(drifted) > 0)
        counts.append(sum(1 for f in drifted if f != "country_code"))
    counts = np.array(counts)
    return {
        "flag_rate": float(np.mean(flagged)),
        "features_excl_country_min": int(counts.min()),
        "at_least_three": float(np.mean(counts >= 3)),
    }


def main():
    df, counts = load()
    es = df[df["country_code"] == "ES"]
    rest = df[df["country_code"] != "ES"]
    makes = rest["make"].value_counts()
    supported = set(makes[makes >= 300].index)
    es_api = es[es["make"].isin(supported)]
    rest_api = rest[rest["make"].isin(supported)]

    print(json.dumps(counts))
    print(f"features={len(DRIFT_FEATURES)} (`model` excluded)  splits={N_SPLITS}  trials={N_TRIALS}")

    results = {"counts": counts, "features": len(DRIFT_FEATURES), "splits": {}}
    for split in range(N_SPLITS):
        rng = np.random.default_rng(SEED + 100 * split)
        perm = rng.permutation(len(rest_api))
        n_control = int(CONTROL_FRACTION * len(rest_api))
        control = rest_api.iloc[perm[:n_control]]
        train_pool = rest_api.iloc[perm[n_control:]]
        reference = train_pool.iloc[
            rng.choice(len(train_pool), size=REFERENCE_SIZE, replace=False)
        ]
        for window in WINDOWS:
            es_trials = collect(reference, es_api, window, np.random.default_rng(SEED + 1 + split))
            ct_trials = collect(reference, control, window, np.random.default_rng(SEED + 2 + split))
            for alpha in ALPHAS:
                e, c = rates(es_trials, alpha), rates(ct_trials, alpha)
                results["splits"][f"s{split}@{window}@{alpha}"] = {"es": e, "control": c}
                print(
                    f"  split {split}  window={window:>4}  alpha={alpha:<6} "
                    f"ES flagged={e['flag_rate']:.3f} (min {e['features_excl_country_min']} "
                    f"features, >=3 in {e['at_least_three']:.2f})  "
                    f"control={c['flag_rate']:.3f} = {c['flag_rate'] * 20:.1f}/20"
                )

    print("\nControl false-alarm rate across splits, as flagged windows of 20:")
    for window in WINDOWS:
        for alpha in ALPHAS:
            vals = [
                results["splits"][f"s{s}@{window}@{alpha}"]["control"]["flag_rate"] * 20
                for s in range(N_SPLITS)
            ]
            es_ok = all(
                results["splits"][f"s{s}@{window}@{alpha}"]["es"]["flag_rate"] == 1.0
                and results["splits"][f"s{s}@{window}@{alpha}"]["es"]["at_least_three"] == 1.0
                for s in range(N_SPLITS)
            )
            print(
                f"  window={window:>4}  alpha={alpha:<6} "
                f"min={min(vals):.1f}  max={max(vals):.1f}  mean={np.mean(vals):.1f}"
                f"   ES clause holds in every split: {'yes' if es_ok else 'no'}"
            )

    with open("reports/analysis/nfr11_alpha_sweep_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwritten: reports/analysis/nfr11_alpha_sweep_results.json")


if __name__ == "__main__":
    main()
