"""The `preprocess` row funnel on the real snapshot (issue #34).

Question: how many listings does each row rule of the `preprocess` stage remove from the
published AutoScout24 file? The report cites the funnel, and the pipeline itself only ever logs
it for whatever data it was run on, which is the synthetic fixture until #33 lands the real
acquisition.

It calls the stage's own functions instead of re-implementing any of its steps, so these numbers
cannot drift from what the pipeline does. The rule order is the one issue #34 fixes and is load-bearing:
each rule sees only what the rule above it left. The raw dataset is not in the repo (NFR-08,
EDN-07), see the README.

Usage: uv run python reports/analysis/preprocess_funnel.py <the raw CSV>
"""

import sys

import pandas as pd

from recommenditos.config import PARAMS_FILE
from recommenditos.data.preprocess import (
    add_the_target,
    apply_row_rules,
    derive_group_key_and_drop_pii,
)
from recommenditos.pipeline import load_params
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA

CSV = sys.argv[1]


def main():
    params = load_params(PARAMS_FILE)
    rules = params["preprocess"]
    reference_date = pd.Timestamp(params["reference_date"])

    raw = RAW_SCHEMA.conform(pd.read_csv(CSV, low_memory=False))

    # What the dataset card measured on the raw file, before any rule runs. The funnel below
    # reports how many of each reach the rule that removes them, which is the smaller number:
    # the rules above delete some of them first.
    registered = pd.to_datetime(raw["registration_date"])
    print(f"raw file: {len(raw):,} listings")
    print(f"  registered after {reference_date.date()}: {(registered > reference_date).sum():,}")
    print(f"  latest registration date: {registered.max().date()}")
    print(
        f"  duplicates on the {len(rules['dedup_key'])}-column key: "
        f"{raw.duplicated(subset=rules['dedup_key']).sum():,}"
    )
    in_scope_offer = raw["offer_type"] == rules["offer_type"]
    print(
        f"  is_used = False while offer_type = {rules['offer_type']}: "
        f"{(in_scope_offer & ~raw['is_used']).sum():,} of {in_scope_offer.sum():,}"
    )
    print()

    frame = derive_group_key_and_drop_pii(raw, params)
    kept, funnel = apply_row_rules(frame, params)
    print("row funnel, each rule applied to what the rule above it left:")
    print(funnel.render())
    print()

    # The same call the stage makes before writing, so this also answers whether the real file
    # produces a contract-valid interim frame and not only the synthetic fixture does.
    interim = INTERIM_SCHEMA.conform(add_the_target(kept))
    print(
        f"interim frame: {len(interim):,} listings, {len(interim.columns)} columns, "
        f"{interim['seller_group_id'].nunique():,} seller groups, "
        f"valid against the {INTERIM_SCHEMA.name!r} contract"
    )
    print(
        f"  of which country_code = ES (the EDN-03 holdout): "
        f"{(interim['country_code'] == 'ES').sum():,}"
    )
    dropped = set(RAW_SCHEMA.names) - set(interim.columns)
    print(
        f"  {len(dropped)} raw columns dropped, PII among them: "
        f"{sorted(set(rules['pii_columns']) & dropped)}"
    )


if __name__ == "__main__":
    main()
