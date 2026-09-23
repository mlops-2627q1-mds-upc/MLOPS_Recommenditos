"""NFR-11 feasibility check.

Question: does an `ES` replay window get flagged as input drift against the training
reference, and does at least one feature other than `country_code` drift?
Control: an equal-sized window drawn from the held-out (non-ES) test split must NOT
be flagged.

Emulates alibi-detect TabularDrift with scipy: KS two-sample for numerical features,
chi-square for categorical features, Bonferroni correction over all features.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

CSV = sys.argv[1]
SNAPSHOT = pd.Timestamp("2025-11-08")
P_VAL = 0.05
N_TRIALS = 200
WINDOWS = [100, 250, 500, 1000]
SEED = 20260923

USECOLS = [
    "make",
    "model",
    "model_version",
    "body_type",
    "registration_date",
    "mileage_km_raw",
    "nr_prev_owners",
    "power_kw",
    "fuel_category",
    "transmission",
    "drive_train",
    "gears",
    "cylinders_volume_cc",
    "nr_seats",
    "nr_doors",
    "country_code",
    "seller_type",
    "price",
    "offer_type",
    "is_preregistered",
    "vehicle_type",
    "seller_company_name",
]
NUM = [
    "age_years",
    "mileage_km_raw",
    "power_kw",
    "nr_prev_owners",
    "gears",
    "cylinders_volume_cc",
    "nr_seats",
    "nr_doors",
]
CAT = [
    "make",
    "model",
    "body_type",
    "fuel_category",
    "transmission",
    "drive_train",
    "seller_type",
    "country_code",
]
# alibi-detect does no NaN handling at all (a NaN p-value silently reads as "no drift"),
# so a real pipeline has to encode missingness. Numerical columns with missing values get
# an explicit missing-indicator feature, tested with chi-square like any categorical.
MISS_IND = [f"{c}__missing" for c in NUM]
FEATURES = NUM + CAT + MISS_IND
MISSING = "__MISSING__"


def load():
    df = pd.read_csv(CSV, usecols=USECOLS, low_memory=False)
    raw_rows = len(df)
    # Scope (EDN-04) + training price range (problem spec section 2)
    df = df[
        (df["offer_type"] == "U")
        & (~df["is_preregistered"].astype("boolean").fillna(False))
        & (df["vehicle_type"] == "Car")
    ]
    df = df[(df["price"] >= 500) & (df["price"] <= 2_000_000)]
    scoped_rows = len(df)
    # Deduplicate before any split (project brief 3.3)
    df = df.drop_duplicates(
        subset=[
            "make",
            "model",
            "model_version",
            "mileage_km_raw",
            "registration_date",
            "price",
            "power_kw",
        ]
    )
    dedup_rows = len(df)
    reg = pd.to_datetime(df["registration_date"], errors="coerce")
    df["age_years"] = (SNAPSHOT - reg).dt.days / 365.25
    # Grouped split key: hashed dealer name, private sellers are their own group
    seller = df["seller_company_name"].fillna("")
    df["group"] = [
        hashlib.sha1(s.encode()).hexdigest() if s else f"private-{i}"
        for i, s in zip(df.index, seller)
    ]
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}__missing"] = (
            df[c].isna().map({True: "missing", False: "present"}).astype("string")
        )
    for c in CAT:
        df[c] = df[c].astype("string").fillna(MISSING)
    return df, {"raw_rows": raw_rows, "scoped_rows": scoped_rows, "dedup_rows": dedup_rows}


def ks_p(ref, win):
    a = ref[~np.isnan(ref)]
    b = win[~np.isnan(win)]
    if len(a) < 2 or len(b) < 2:
        return 1.0, 0.0
    # alibi-detect v0.13 calls ks_2samp(..., mode='asymp')
    r = stats.ks_2samp(a, b, method="asymp")
    return float(r.pvalue), float(r.statistic)


def chi2_p(ref, win):
    cats = pd.Index(sorted(set(ref.unique()) | set(win.unique())))
    a = ref.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    b = win.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    keep = (a + b) > 0
    a, b = a[keep], b[keep]
    if len(a) < 2:
        return 1.0, 0.0
    chi2, p, _, _ = stats.chi2_contingency(np.vstack([a, b]))
    return float(p), float(chi2)


def drift(ref_df, win_df, features):
    out = {}
    for f in features:
        if f in NUM:
            out[f] = ks_p(ref_df[f].to_numpy(dtype=float), win_df[f].to_numpy(dtype=float))
        else:
            out[f] = chi2_p(ref_df[f], win_df[f])
    return out


def summarise(ref_df, pool, features, label, rng, n_trials=N_TRIALS):
    thr = P_VAL / len(features)
    res = {}
    for n in WINDOWS:
        if len(pool) < n:
            continue
        flags, per_feat, per_feat_nc = [], {f: 0 for f in features}, []
        for _ in range(n_trials):
            idx = rng.choice(len(pool), size=n, replace=False)
            win = pool.iloc[idx]
            ps = drift(ref_df, win, features)
            drifted = [f for f, (p, _) in ps.items() if p < thr]
            flags.append(len(drifted) > 0)
            for f in drifted:
                per_feat[f] += 1
            per_feat_nc.append(any(f != "country_code" for f in drifted))
        res[n] = {
            "detection_rate": float(np.mean(flags)),
            "detection_rate_excl_country": float(np.mean(per_feat_nc)),
            "per_feature_rate": {f: per_feat[f] / n_trials for f in features},
        }
        print(
            f"  [{label}] window={n:>5}  flagged={np.mean(flags):.3f}  "
            f"flagged without country_code={np.mean(per_feat_nc):.3f}"
        )
    return res


def main():
    df, counts = load()
    es = df[df["country_code"] == "ES"]
    rest = df[df["country_code"] != "ES"]
    makes = rest["make"].value_counts()
    supported = set(makes[makes >= 300].index)
    # The API rejects unsupported makes (FR-04), so they never reach the prediction log
    es_api = es[es["make"].isin(supported)]
    rest_api = rest[rest["make"].isin(supported)]

    # Grouped split of the non-ES data: 80 % reference (training), 20 % control (test)
    rng = np.random.default_rng(SEED)
    groups = rest_api["group"].unique()
    rng.shuffle(groups)
    cut = int(0.8 * len(groups))
    ref_groups = set(groups[:cut])
    ref = rest_api[rest_api["group"].isin(ref_groups)]
    control = rest_api[~rest_api["group"].isin(ref_groups)]

    print(json.dumps(counts))
    print(f"supported makes: {len(supported)} -> {sorted(supported)}")
    print(
        f"reference(train, non-ES, supported)={len(ref)}  control(test)={len(control)}  "
        f"ES total={len(es)}  ES accepted by API={len(es_api)}"
    )

    thr = P_VAL / len(FEATURES)
    print(f"\nBonferroni threshold: {P_VAL}/{len(FEATURES)} = {thr:.5f}\n")

    print("Per-feature test on the FULL ES set vs the full reference:")
    full = drift(ref, es_api, FEATURES)
    rows = []
    for f, (p, stat) in sorted(full.items(), key=lambda kv: kv[1][0]):
        rows.append({"feature": f, "p": p, "stat": stat, "drift": p < thr})
        print(f"  {f:<22} p={p:.3e}  stat={stat:10.4f}  {'DRIFT' if p < thr else '-'}")

    print("\nEffect sizes (reference vs ES), numerical means / categorical top shares:")
    for f in NUM:
        print(
            f"  {f:<22} ref mean={ref[f].mean():9.2f}  ES mean={es_api[f].mean():9.2f}  "
            f"ref missing={ref[f].isna().mean():.1%}  ES missing={es_api[f].isna().mean():.1%}"
        )
    for f in CAT:
        top = ref[f].value_counts(normalize=True).head(3)
        es_share = es_api[f].value_counts(normalize=True)
        parts = ", ".join(f"{k}: {v:.1%} vs {es_share.get(k, 0):.1%}" for k, v in top.items())
        print(f"  {f:<22} {parts}")

    print("\nReplay windows (ES), 200 trials each:")
    es_res = summarise(ref, es_api, FEATURES, "ES", np.random.default_rng(SEED + 1))
    print("\nControl windows (held-out non-ES test split), 200 trials each:")
    ctl_res = summarise(ref, control, FEATURES, "control", np.random.default_rng(SEED + 2))

    print("\nPer-feature detection rate at window=1000 (ES):")
    for f, r in sorted(es_res[1000]["per_feature_rate"].items(), key=lambda kv: -kv[1]):
        print(f"  {f:<22} {r:.3f}")
    print("\nPer-feature false-alarm rate at window=1000 (control):")
    for f, r in sorted(ctl_res[1000]["per_feature_rate"].items(), key=lambda kv: -kv[1]):
        print(f"  {f:<22} {r:.3f}")

    with open("nfr11_results.json", "w") as fh:
        json.dump({"counts": counts, "full": rows, "es": es_res, "control": ctl_res}, fh, indent=2)


if __name__ == "__main__":
    main()
