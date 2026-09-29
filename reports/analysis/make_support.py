"""SC-04 coverage per supported make (model card, bias and limitations).

Question: EDN-05 admits a make into scope at 300 listings, but SC-04 only checks a segment once
it holds 500 test rows. Which supported makes are therefore never checked by the quality gate?

Scope matches the evaluation protocol, problem specification sections 2 and 5: used cars only
(EDN-04), listings registered after the age reference date dropped (EDN-22), the training price
range, deduplicated before any split, `ES` held out (EDN-03), and only makes with at least 300
listings (EDN-05). The raw dataset comes from DVC (`dvc pull`), see the README.
"""

import sys

import pandas as pd

CSV = sys.argv[1]
MIN_LISTINGS_PER_MAKE = 300  # EDN-05
MIN_SEGMENT_TEST_ROWS = 500  # SC-04
TEST_SHARE = 0.2  # the exploratory run's 80/20 split, problem specification section 8
PRICE_MIN = 500  # training price range, problem specification section 2
PRICE_MAX = 2_000_000

SCOPE_COLS = [
    "make",
    "model",
    "model_version",
    "mileage_km_raw",
    "registration_date",
    "price",
    "power_kw",
    "offer_type",
    "is_preregistered",
    "vehicle_type",
    "country_code",
]
SNAPSHOT = pd.Timestamp("2025-11-08")  # age reference date, problem specification section 4


def load(path):
    df = pd.read_csv(path, usecols=SCOPE_COLS, low_memory=False)
    df = df[
        (df["offer_type"] == "U")
        & (~df["is_preregistered"].astype("boolean").fillna(False))
        & (df["vehicle_type"] == "Car")
    ]
    # EDN-22: a registration date after the reference date makes age negative
    reg = pd.to_datetime(df["registration_date"], errors="coerce")
    df = df[~(reg > SNAPSHOT)]
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
    return df.drop_duplicates(
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


def main():
    train = load(CSV)
    train = train[train["country_code"] != "ES"]
    counts = train["make"].value_counts()
    supported = counts[counts >= MIN_LISTINGS_PER_MAKE]
    breakeven = MIN_SEGMENT_TEST_ROWS / TEST_SHARE

    print(f"training scope (non-ES, {len(supported)} supported makes): {supported.sum():,}")
    print(f"SC-04 needs {MIN_SEGMENT_TEST_ROWS} test rows, at a {TEST_SHARE:.0%} test share")
    print(f"so a make needs {breakeven:,.0f} listings in scope to be checked at all\n")

    print(f"{'make':16}{'listings':>10}{'est. test rows':>16}  checked by SC-04?")
    for make, n in supported.items():
        checked = "yes" if n * TEST_SHARE >= MIN_SEGMENT_TEST_ROWS else "NO"
        print(f"{make:16}{n:>10,}{n * TEST_SHARE:>16,.0f}  {checked}")

    unchecked = supported[supported * TEST_SHARE < MIN_SEGMENT_TEST_ROWS]
    print(
        f"\nsupported but never checked by SC-04: {len(unchecked)} of {len(supported)} makes, "
        f"{unchecked.sum():,} listings "
        f"({unchecked.sum() / supported.sum() * 100:.1f} % of the training scope)"
    )
    print(f"they are: {', '.join(unchecked.index)}")


if __name__ == "__main__":
    main()
