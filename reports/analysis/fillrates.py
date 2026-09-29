"""FR-01 / SC-06 fill-rate check (EDN-15).

Question: which of the fields FR-01 leaves optional are so often filled in the training data
that the model would never see them missing, and would therefore have no learned behaviour for
a request that omits them?

Scope matches the evaluation protocol: used cars only (EDN-04), the training price range,
deduplicated before any split, `ES` held out (EDN-03), and only makes with at least 300
listings (EDN-05). The raw dataset is not in the repo (NFR-08, EDN-07), see the README.
"""

import sys

import pandas as pd

CSV = sys.argv[1]
MIN_LISTINGS_PER_MAKE = 300

REQUIRED = [
    "make",
    "model",
    "registration_date",
    "mileage_km_raw",
    "power_kw",
    "fuel_category",
    "transmission",
    "country_code",
    "body_type",
    "seller_type",
]
OPTIONAL = [
    "nr_prev_owners",
    "drive_train",
    "gears",
    "cylinders_volume_cc",
    "nr_seats",
    "nr_doors",
]
SCOPE_COLS = ["model_version", "price", "offer_type", "is_preregistered", "vehicle_type"]


def load(path):
    df = pd.read_csv(path, usecols=sorted(set(REQUIRED + OPTIONAL + SCOPE_COLS)), low_memory=False)
    raw_rows = len(df)
    df = df[
        (df["offer_type"] == "U")
        & (~df["is_preregistered"].astype("boolean").fillna(False))
        & (df["vehicle_type"] == "Car")
    ]
    df = df[(df["price"] >= 500) & (df["price"] <= 2_000_000)]
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
    return df, raw_rows


def filled_pct(col):
    values = col.replace(r"^\s*$", None, regex=True) if col.dtype == object else col
    return values.notna().mean() * 100


def main():
    df, raw_rows = load(CSV)
    print(f"raw rows: {raw_rows:,}")
    print(f"scoped and deduplicated: {len(df):,}")

    train = df[df["country_code"] != "ES"]
    counts = train["make"].value_counts()
    makes = counts[counts >= MIN_LISTINGS_PER_MAKE].index
    train = train[train["make"].isin(makes)]
    es = df[(df["country_code"] == "ES") & (df["make"].isin(makes))]
    print(f"training scope (non-ES, {len(makes)} supported makes): {len(train):,}")
    print(f"ES holdout, as the API would accept it: {len(es):,}\n")

    print(f"{'feature':24}{'train %':>9}{'ES %':>9}  role in FR-01")
    for group, role in ((REQUIRED, "required"), (OPTIONAL, "optional")):
        for name in group:
            print(f"{name:24}{filled_pct(train[name]):9.1f}{filled_pct(es[name]):9.1f}  {role}")
        print()

    missing = train[OPTIONAL].isna().sum(axis=1)
    print(f"missing optional fields per training row (of {len(OPTIONAL)}):")
    for count, rows in missing.value_counts().sort_index().items():
        print(f"  {count}: {rows:>8,}  ({rows / len(train) * 100:5.1f} %)")
    print(f"\nrows with every optional field present: {(missing == 0).mean() * 100:.1f} %")
    print(f"rows with every optional field absent:  {(missing == len(OPTIONAL)).sum():,}")


if __name__ == "__main__":
    main()
