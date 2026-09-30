"""`preprocess` stage: raw listings to the interim contract.

STUB. It does only what the interim contract cannot be satisfied without: hash
the seller into the split's group key *before* the PII columns are dropped,
parse the registration date, build the target, and conform - which drops every
column the contract does not name, PII included.

Issue #34 adds the row rules, in this order: scope to used passenger cars
(EDN-04), drop listings registered after the reference date (EDN-22), restrict
the price range, and deduplicate on the 7-column key. The supported-make filter
is deliberately NOT here: EDN-05 counts support after the `ES` holdout, which
happens in `split`.

`hash_seller_group` is the one function the API will reuse, so a seller hashed
at training time and one hashed at serving time cannot drift apart.
"""

import hashlib
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from recommenditos.config import INTERIM_DATA_DIR, PARAMS_FILE, RAW_DATA_DIR
from recommenditos.pipeline import load_params, read_frame, write_frame
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA

app = typer.Typer()


def hash_seller_group(frame: pd.DataFrame) -> pd.Series:
    """The split's grouping key, derived before `seller_company_name` is dropped.

    A dealer groups by its company name. A private seller has none, so it
    groups by location instead, which is what problem-spec section 5 means by
    "location for private sellers". The hash is one-way: the interim frame
    carries no way back to the name (NFR-08).

    Two deliberate over-groupings, both in the safe direction: listings that
    end up in one group cannot be split across two sets, which is the leak the
    grouping exists to prevent.

    - A private seller with no location at all (15 such rows in the raw file)
      falls into one shared group.
    - A dealer groups by its company name alone, so two unrelated dealers of
      the same name in different countries merge. The location is available
      and deliberately not used, because a dealer with branches in two cities
      is one seller and splitting it would leak.
    """
    private = frame["seller_company_name"].isna()
    location = (
        frame["country_code"].fillna("??").astype("str")
        + "|"
        + frame["zip"].fillna("?????").astype("str")
        + "|"
        + frame["city"].fillna("?").astype("str")
    )
    raw_key = frame["seller_company_name"].where(~private, "private|" + location)
    return raw_key.map(lambda key: hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16])


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "listings.parquet",
    output_path: Path = INTERIM_DATA_DIR / "listings.parquet",
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    frame = read_frame(input_path, RAW_SCHEMA)

    # Before the PII columns go, because it is derived from one of them.
    frame["seller_group_id"] = hash_seller_group(frame)
    frame["registration_date"] = pd.to_datetime(frame["registration_date"])
    frame["log_price"] = np.log(frame["price"])

    logger.warning(
        "STUB: the scope, date, price-range and deduplication rules of issue #34 are "
        "not applied yet, so every raw row reaches the interim frame."
    )
    logger.info(f"Reference date for the age feature: {params['reference_date']}.")

    # `conform` selects the contract's columns, so the PII, leakage and
    # identifier columns cannot reach the interim artefact even by accident.
    write_frame(frame, output_path, INTERIM_SCHEMA)


if __name__ == "__main__":
    app()
