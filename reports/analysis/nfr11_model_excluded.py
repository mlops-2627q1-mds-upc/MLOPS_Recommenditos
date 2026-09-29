"""NFR-11 control clause: is `model` the reason the control fails at window 100?

EDN-14 fixed NFR-11 on two runs with different settings: the `ES` half came from
`nfr11_check.py` (window 100, full reference), the control half from `nfr11_diag.py`
(window 1000, reference capped at 10,000). Stated together as one requirement, the control
clause ("of 20 control windows, at most 1 is flagged") was never measured at the window the
requirement names.

This run measures both halves in one configuration, at the reference size NFR-11 prescribes,
and asks whether the false alarms come from `model` specifically: measured here, `model` has
349 levels in the reference, so a chi-square against a 100-row window puts well under one
expected observation in most cells. That is an under-powered test rather than a small window,
and it is worth separating from the window size before the window is changed.

Control is drawn i.i.d. from held-out listings, as EDN-14 requires; the seller-grouped
control it rejected is not repeated here.

    uv run python reports/analysis/nfr11_model_excluded.py <path-to-cars.csv>
"""

import json
import sys

import numpy as np

from nfr11_check import FEATURES, P_VAL, SEED, drift, load

REFERENCE_SIZE = 10_000
WINDOWS = [100, 1000]
N_TRIALS = 200
CONTROL_FRACTION = 0.2

FEATURE_SETS = {
    "all": FEATURES,
    "no_model": [f for f in FEATURES if f != "model"],
}


def trial_counts(ref_df, pool, features, window, rng, n_trials=N_TRIALS):
    """Per trial: was anything flagged, and how many features other than country_code."""
    thr = P_VAL / len(features)
    flagged, counts_excl_country, per_feature = [], [], {f: 0 for f in features}
    for _ in range(n_trials):
        win = pool.iloc[rng.choice(len(pool), size=window, replace=False)]
        drifted = [f for f, (p, _) in drift(ref_df, win, features).items() if p < thr]
        flagged.append(len(drifted) > 0)
        counts_excl_country.append(sum(1 for f in drifted if f != "country_code"))
        for f in drifted:
            per_feature[f] += 1
    counts = np.array(counts_excl_country)
    return {
        "threshold": thr,
        "flag_rate": float(np.mean(flagged)),
        "features_excl_country_min": int(counts.min()),
        "features_excl_country_median": float(np.median(counts)),
        "trials_with_at_least_three": float(np.mean(counts >= 3)),
        "per_feature_rate": {f: per_feature[f] / n_trials for f in features},
    }


def main():
    df, counts = load()
    es = df[df["country_code"] == "ES"]
    rest = df[df["country_code"] != "ES"]
    makes = rest["make"].value_counts()
    supported = set(makes[makes >= 300].index)
    es_api = es[es["make"].isin(supported)]
    rest_api = rest[rest["make"].isin(supported)]

    # i.i.d. holdout, as EDN-14 option D defines the control
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(rest_api))
    n_control = int(CONTROL_FRACTION * len(rest_api))
    control = rest_api.iloc[perm[:n_control]]
    train_pool = rest_api.iloc[perm[n_control:]]
    reference = train_pool.iloc[
        rng.choice(len(train_pool), size=REFERENCE_SIZE, replace=False)
    ]

    print(json.dumps(counts))
    print(
        f"supported makes={len(supported)}  reference={len(reference)} "
        f"(sampled from {len(train_pool)})  control pool={len(control)}  ES accepted={len(es_api)}"
    )
    print(f"distinct `model` values in the reference: {reference['model'].nunique()}")

    results = {"counts": counts, "reference_size": REFERENCE_SIZE, "runs": {}}
    for set_name, features in FEATURE_SETS.items():
        for window in WINDOWS:
            es_res = trial_counts(
                reference, es_api, features, window, np.random.default_rng(SEED + 1)
            )
            ct_res = trial_counts(
                reference, control, features, window, np.random.default_rng(SEED + 2)
            )
            results["runs"][f"{set_name}@{window}"] = {
                "features": len(features),
                "es": es_res,
                "control": ct_res,
            }
            print(
                f"\n[{set_name:>8}] window={window:>4}  features={len(features)}\n"
                f"  ES      flagged={es_res['flag_rate']:.3f}  "
                f"other-than-country min={es_res['features_excl_country_min']} "
                f"median={es_res['features_excl_country_median']:.0f}  "
                f">=3 in {es_res['trials_with_at_least_three']:.3f} of trials\n"
                f"  control flagged={ct_res['flag_rate']:.3f}  "
                f"= {ct_res['flag_rate'] * 20:.1f} of 20 windows"
            )
            worst = sorted(
                ct_res["per_feature_rate"].items(), key=lambda kv: -kv[1]
            )[:4]
            print(
                "  control false alarms by feature: "
                + ", ".join(f"{f}={r:.3f}" for f, r in worst if r > 0)
            )

    with open("reports/analysis/nfr11_model_excluded_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwritten: reports/analysis/nfr11_model_excluded_results.json")


if __name__ == "__main__":
    main()
